# LanceDB documentation style

Write for developers who want to understand LanceDB and get something done.
Be direct, approachable and technically precise. Give readers enough context
to make a decision, then show them how to act on it.

This guide covers prose on docs.lancedb.com: the pages in `lancedb/lancedb`
`docs/web`, the public Enterprise pages in `lancedb/sophon` `docs/web`, and the
Geneva, dataset and REST pages in this repository. [WRITING.md](WRITING.md)
covers Mintlify mechanics.

## Start with the reader's goal

Open with what the feature does and when to use it. Keep the introduction
short. Put prerequisites before the instructions that need them, and explain
limitations beside the choice or operation they affect.

Show the common path before advanced options. A guide should get readers to a
useful result before asking them to compare every configuration. This is a
default, not a required outline for every page.

## Give each page a purpose

A concept page explains a few central ideas and how they relate. A guide helps
readers complete a task. A reference describes exact syntax, options and
behavior. Link to related material with descriptive link text instead of
making every page cover all three roles.

Keep each section focused on the page's purpose. Don't add a conclusion or a
next-steps section unless it helps the reader decide what to do next.

## Teach through examples

Use small datasets, meaningful names and examples that build on each other.
Include the imports, setup and prerequisites readers need, or clearly link to
shared setup. State when successive blocks belong to one program. Show the
expected result when it helps readers understand the operation.

Where equivalent SDK support exists, show Python, TypeScript and Rust
consistently. Use the same task and sample data across languages, with
idiomatic code in each. If support differs, explain the difference beside the
example; don't invent an equivalent API. Prefer fresh example data and make
any destructive behavior explicit.

Explain what an example demonstrates or why an option matters. Avoid narrating
every line when the code already makes its meaning clear.

## Explain choices and consequences

Tell readers when to choose an option and what changes when they do. State the
relevant tradeoff, such as latency versus recall, rather than listing settings
without guidance. Use a table when it makes alternatives easier to compare.

Name the product or component whose behavior you describe. Use LanceDB
Enterprise when you mean the offering; use deployment when you mean a
particular installation or its configuration. Describe concrete capabilities
and benefits without unsupported performance claims or superlatives.

## Say it directly

Address the reader as "you" and prefer active verbs and familiar words.
Introduce unfamiliar terms when readers first need them. Prefer "use" to
"utilize" and "to" to "in order to". Keep technical terms precise: "GPU
utilization", for example, has a specific meaning and does not need replacing.

Describe the subject, not the page or the writing. Instead of "This page is the
concept; the SQL reference is the specification", explain the concept and
link to the reference. Instead of opening with "That matters", state the
consequence. Avoid "simply" and "obviously" when they assume a task is easy for
the reader. These are writing guidelines, not blanket bans on individual words.

## Headings

A heading names the task or topic of its section. "Then what" doesn't; "Choose
an output" does. Prefer specific headings such as "Data durability" to broad
questions such as "What survives?". Use sentence case, preserving product names
and identifiers.

## Names

Write product names exactly: LanceDB, LanceDB OSS, LanceDB Cloud and LanceDB
Enterprise. Write other names as their owners do, such as Hugging Face,
LangChain, TypeScript and Voyage AI.

Package, module, class and repository names are code, so put them in code
font: `lancedb`, `pyarrow`, `HuggingFaceEmbeddings`.

## Punctuation

Don't use em dashes in prose, or a double hyphen or spaced en dash in place of
one. Use a comma, a colon, parentheses or two sentences. An en dash in a range,
such as 1–3, is fine, and so is a table cell that holds only an em dash to mean
"none".

## Automated checks and review

[Vale](https://vale.sh) checks a small subset of these conventions: product
names, dashes, repeated words, a few vague phrases and headings, and a few
plain-language substitutions. Findings are advisory and appear on changed
lines; they never fail the check. Installation or configuration failures can
still fail it. See the [Vale package guide](vale/README.md) for usage and tests.

Vale cannot verify technical claims, example correctness, SDK coverage or a
page's organization. Review those using implementation evidence, tests and
feature-owner input. Confirm availability and release status with the owner;
do not infer a release promise from an implementation or a passing test.
Reader-visible component titles and captions in MDX attributes are also left
to review because Vale does not inspect those attributes.

Use these guidelines as defaults. Reviewers can make exceptions when they
improve clarity or accuracy.

## Exceptions

When a sentence must break an automated rule, such as a quoted error message,
switch off that rule around it and say why in the pull request:

```mdx
{/* vale LanceDB.Dashes = NO */}
The server returns "Table not found — check the name".
{/* vale LanceDB.Dashes = YES */}
```
