# LanceDB documentation style

This guide covers prose on docs.lancedb.com: the pages in `lancedb/lancedb`
`docs/web`, the public Enterprise pages in `lancedb/sophon` `docs/web`, and the
Geneva, dataset and REST pages in this repository. [WRITING.md](WRITING.md)
covers Mintlify mechanics.

[Vale](https://vale.sh) checks the rules below, with the shared package in
[`vale/`](vale/README.md). It reports suggestions on the lines a pull request
changes and never blocks a merge. Everything else in this guide is left to
review.

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

Vale also flags a repeated word, such as "the the".

## Headings

A heading names the task or topic of its section. "Then what" doesn't; "Choose
an output" does.

## Say it directly

Describe the subject, not the page or the writing. Instead of "This page is the
concept; the SQL reference is the specification", say what the reader learns
here and link to the reference. Drop filler openers such as "That matters" and
state the consequence.

## Left to review

Vale checks wording, not meaning, so these stay with reviewers:

- **Word choice that depends on meaning.** "Deployment" is the right word for
  infrastructure and in identifiers such as `/enterprise/deployment`. Whether
  it is the right word for an offering is a review question.
- **Product claims.** Availability, Enterprise behavior and release status come
  from the feature owner and the editor, not from a linter or a test.
- **Coverage and organization.** Whether a change needs a page, which related
  guides it affects, and how the site reads as a whole. Gatekeeper suggests
  affected pages and skills on pull requests; Brendan owns editorial decisions.

## Exceptions

When a sentence must break a rule, such as a quoted error message, switch off
that rule around it and say why in the pull request:

```mdx
{/* vale LanceDB.Dashes = NO */}
The server returns "Table not found — check the name".
{/* vale LanceDB.Dashes = YES */}
```
