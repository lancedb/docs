// Finds the links in MDX texts for scripts/compose.py, parsed as Mint parses pages
// (coreRemark in @mintlify/common), so that composition changes only what Mint renders as a link.
import remarkFrontmatter from 'remark-frontmatter'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import remarkMdx from 'remark-mdx'
import remarkParse from 'remark-parse'
import {unified} from 'unified'

const FENCE = /^(`{3,}|~{3,})/
const HEADING_ID = /^( {0,3}#{1,6}\s+.+?\s*)(\{\s*#[^}]+?\s*\})\s*$/
const MODULES = new Set(['ImportDeclaration', 'ExportAllDeclaration', 'ExportNamedDeclaration'])

let destinations = []
const processor = unified()
  .use(remarkParse)
  .use(remarkMdx)
  .use(remarkGfm)
  .use(remarkFrontmatter, ['yaml', 'toml'])
  .use(remarkMath)
  .use(function () {
    const data = this.data()
    ;(data.fromMarkdownExtensions ||= []).push({
      afterExit(token) {
        if (token.type === 'resourceDestinationString' || token.type === 'definitionDestinationString') {
          destinations.push([token.start.offset, token.end.offset, this.stack.at(-1).url])
        }
      }
    })
  })

process.stdin.setEncoding('utf8')
let input = ''
for await (const chunk of process.stdin) input += chunk
const {attributes, texts} = JSON.parse(input)
const linkAttributes = new Set(attributes)
process.stdout.write(JSON.stringify(texts.map(find)))

function find(text) {
  const bom = text.startsWith('\uFEFF') ? 1 : 0
  const source = withoutHeadingIds(text.slice(bom))
  const at = codePoints(source, bom)
  destinations = []
  let tree
  try {
    tree = processor.parse(source)
  } catch (error) {
    if (error?.reason === undefined) throw error
    return {error: [error.line ?? null, error.reason]}
  }
  const found = {links: [], imports: [], computed: [], bare: [], code: []}
  for (const [start, end, url] of destinations) found.links.push([at(start), at(end), url, null])
  visit(tree)
  return found

  function visit(node) {
    if (node.type === 'link' && source[node.position.start.offset] !== '[') {
      found.bare.push([at(node.position.start.offset), node.url])
    } else if (node.type === 'code' || node.type === 'inlineCode') {
      found.code.push([at(node.position.start.offset), node.value])
    } else if (node.data?.estree) {
      script(node.data.estree)
    }
    node.attributes?.forEach(attribute)
    node.children?.forEach(visit)
  }

  function attribute(node) {
    const value = node.value
    if (node.type !== 'mdxJsxAttribute' || !linkAttributes.has(node.name)) {
      script(node.data?.estree ?? value?.data?.estree)
    } else if (typeof value === 'string') {
      const end = node.position.end.offset - 1
      found.links.push([at(source.lastIndexOf(source[end], end - 1) + 1), at(end), value, node.name])
    } else if (value) {
      link(node.name, value.data.estree.body[0]?.expression, node.position.start.offset)
    }
  }

  function link(name, expression, fallback) {
    if (expression?.type === 'Literal' && typeof expression.value === 'string') {
      found.links.push([at(expression.start + 1), at(expression.end - 1), expression.value, name])
    } else if (expression?.type === 'TemplateLiteral' && !expression.expressions.length) {
      const [quasi] = expression.quasis
      found.links.push([at(quasi.start), at(quasi.end), quasi.value.cooked, name])
    } else {
      found.computed.push([at(expression?.start ?? fallback), name])
    }
  }

  function script(node) {
    if (Array.isArray(node)) return node.forEach(script)
    if (typeof node?.type !== 'string') return
    if (node.type === 'JSXAttribute') return jsxAttribute(node)
    if (node.type === 'Literal' && typeof node.value === 'string') found.code.push([at(node.start), node.value])
    if (node.type === 'TemplateElement') found.code.push([at(node.start), node.value.cooked ?? node.value.raw])
    const module = MODULES.has(node.type) ? node.source : null
    if (module) found.imports.push([at(module.start + 1), at(module.end - 1), module.value])
    for (const key in node) if (key !== 'loc' && key !== 'range' && node[key] !== module) script(node[key])
  }

  function jsxAttribute(node) {
    const name = node.name.type === 'JSXIdentifier' ? node.name.name : null
    const value = node.value
    if (!linkAttributes.has(name)) {
      if (value?.type !== 'Literal') script(value)
    } else if (value?.type === 'Literal') {
      found.links.push([at(value.start + 1), at(value.end - 1), value.value, name])
    } else if (value?.type === 'JSXExpressionContainer') {
      link(name, value.expression, value.start)
    } else {
      script(value)
    }
  }
}

// Mint turns `## Title {#id}` into a heading element before parsing (preprocessCustomHeadingIds);
// blanking the id instead keeps every offset.
function withoutHeadingIds(text) {
  let fence
  return text
    .split('\n')
    .map((line) => {
      const stripped = line.trimStart()
      const run = FENCE.exec(stripped)?.[1]
      if (fence) {
        if (run?.[0] === fence[0] && run.length >= fence.length && /^[ \t\r]*$/.test(stripped.slice(run.length))) {
          fence = undefined
        }
        return line
      }
      if (run) {
        fence = run
        return line
      }
      const found = HEADING_ID.exec(line)
      if (!found) return line
      return found[1] + found[2].replace(/./gsu, ' ') + line.slice(found[1].length + found[2].length)
    })
    .join('\n')
}

// Python indexes the text in code points, and micromark drops a byte order mark.
function codePoints(text, bom) {
  if (!/[\uD800-\uDFFF]/.test(text)) return (offset) => offset + bom
  const index = new Uint32Array(text.length + 1)
  let points = bom
  for (let i = 0; i < text.length; i++) {
    index[i] = points
    const unit = text.charCodeAt(i)
    if (unit < 0xdc00 || unit > 0xdfff) points++
  }
  index[text.length] = points
  return (offset) => index[offset]
}
