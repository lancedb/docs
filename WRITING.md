# Writing guide

This is a documentation site built in [Mintlify](https://www.mintlify.com/docs). Writing in Mintlify is similar to `README.md`
docs you may be used to writing in markdown -- the main difference is that Mintlify uses MDX files (Markdown + JSX) files
instead of regular markdown files.

It's worth going through the key sections in the Mintlify [docs](https://www.mintlify.com/docs) before you begin writing.
To begin writing, create a new MDX file in the appropriate location and follow the steps below.

## 1. Create the MDX file

As far as possible, docs are organized at the top level by concept, in the `/docs/` directory. In certain cases,
an additional level of nesting into an inner subdirectory is okay to avoid cluttering in the sidebar. However, it's
recommended to avoid nesting more than 2 levels deep as this affects the flow and discoverability from a user
perspective.

## 2. Update `docs.json`

The sitemap is defined in JSON format in `docs.json`. You can organize new content into tab groups and pages, as per the
structure shown in the existing `docs.json`. To view the new page in the sidebar and in the local preview, it must be
referenced in `docs.json` in the appropriate location.

## 3. Add frontmatter

Frontmatter is written in YAML, and is compulsory for all MDX files that contain documentation. It's recommended to _always_
have at least the first three keys (`title`, `sidebarTitle` and `description`) for readability and SEO on a given docs page.
Specifying an `icon` helps readers associate a familiar image with the page title in the sidebar. For searchability within
the docs, you can optionally specify the `keywords` field and pass in a list of keyword strings. When a user searches for
those strings, the page is prioritized in the search box.

Here's an example:

```yml
---
title: "Lance format"
sidebarTitle: "Lance format"
description: "Open-source lakehouse format for multimodal AI."
icon: "/static/assets/logo/lance-logo-gray.svg"
keywords: ["lance"]
---
```

> ![NOTE]
> The example above showed a custom SVG icon in the `/static/assets/` directory of this repo, but you can pick stock
> icons from [fontawesome.com](https://fontawesome.com/icons) by searching for a high-level concept by name.

## 4. Begin writing

Writing in Mintlify is similar to conventional markdown, except that you have access to JSX-based (React) components that
make it much simple to add documentation-friendly functionality and aesthetics to the docs page. Components are a very powerful
addition to the writing experience, and are covered in detail on the [Mintlify docs](https://www.mintlify.com/docs/components/accordions).

Below is an example of a `Card`, which emphasizes content, while providing a clickable URL out of the given page.
```jsx
<Card
    title="Quickstart"
    icon="rocket"
    href="/quickstart"
>
Get started with LanceDB in minutes.
</Card>
```

The best part about components is that they are composable. You can embed one component inside another and achieve the functionality of both. The example below shows an `Card` at the top level, with an `Accordion` inside it.

```mdx
<Card
    title="Quickstart"
    icon="rocket"
    href="/quickstart"
>
Get started with LanceDB in minutes.

<Accordion>
Collapsible text content here....
</Accordion>

</Card>
```

## 5. Mathematical equations

Math equations are supported via standard KaTeX plugins. You can write any LaTeX-style equation and get it rendered on the
page by enclosing it in `$$` symbols.

```
$$
E = mc^2
$$
```

## 6. Code snippets

Code snippets are where Mintlify probably differs the most from markdown. There are several ways to write code snippets, but
this section describes how we do it specifically in these LanceDB docs.

### Option 1: `CodeGroup` components

The preferred way to include a code snippet is to enter it within <CodeGroup> tags, as follows:

```mdx
<CodeGroup>
```python Python icon="python"
import lancedb
```

```typescript TypeScript icon="square-js"
import * as lancedb from "@lancedb/lancedb";
```

```rust Rust icon="rust"
use lancedb::connect;
```
</CodeGroup>
```

This will allow you to include code snippets from multiple languages, grouped together on the docs page so that the user
can click on their language of choice via tabs.

### Option 2: `CodeBlock` components within `CodeGroup`

As engineers, we may want to write a testable snippet in code in the `tests/py`, `tests/ts`, or `tests/rs` directory.
These directories contain test files in each language that contain valid, tested code, which are fenced within comment markers
so that they can be parsed by a [snippet generation script](./scripts/mdx_snippets_gen.py).

The snippet generation script is run to extract the relevant snippets from the file (based on the fenced comment markers
indicating `start` and `end` in each test file).

Here's how you'd call the snippet into a code block in the MDX file:

```mdx
import { PyConnect, TsConnect, RsConnect } from '/snippets/connection.mdx';

<CodeGroup >
    <CodeBlock filename="Python" language="Python" icon="python">
    {PyConnect}
    </CodeBlock>

    <CodeBlock filename="TypeScript" language="TypeScript" icon="square-js">
    {TsConnect}
    </CodeBlock>

    <CodeBlock filename="Rust" language="Rust" icon="rust">
    { "// Rust imports go here\n" }
    {RsConnect}
    </CodeBlock>
</CodeGroup >

```

### Option 3: Vanilla backticks

This is the least preferred approach, as it doesn't let you group together code snippets from multiple languages effectively.
Note that Mintlify offers some additional features compared to traditional markdown even when using triple backticks.

In the example below, we may have a long code snippet that we want to collapse (to show a few lines in the rendered page).
This is useful for example code or data snippets that are quite long.

```json camelot.json icon="brackets-curly" expandable=true
[
  {
    "id": 1,
    "name": "King Arthur",
    "role": "King of Camelot",
    "description": "The legendary ruler of Camelot, wielder of Excalibur, and leader of the Knights of the Round Table.",
    "vector": [0.72, -0.28, 0.60, 0.86],
    "stats": { "strength": 2, "courage": 5, "magic": 1, "wisdom": 4 }
  }
]
```

Using vanilla backticks is okay when the code snippets like JSON blobs can get really long, and we only want to show a
preview to the reader. Enabling `expandable=true` allows readers to see the whole block when they click on the "expand"
button on the page.

## 7. Run local deployment

After you update the `docs.json` page with the path to the new MDX file, you can debug the site on the local deployment.

```bash
# cd to the docs/ directory
cd docs
# Run local server
mint dev
```
This will run a local deployment on `localhost:3000`, which is useful for debugging and testing purposes.

You can check for broken lines in the site by running the following command.

```bash
mint broken-links
```

> ![NOTE]
> The broken link checker **only** checks for internal (relative) links to other pages within this docs repo.
> It cannot check external site links.

## 8. Commit to the docs repo

Once you've finished writing and reviewing the content yourself, submit a PR to the [repo](https://github.com/lancedb/docs)
for review. If you're an external contributor, we thank you for your contribution to LanceDB!