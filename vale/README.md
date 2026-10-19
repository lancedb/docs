# Vale package for LanceDB documentation

`LanceDB/` is a [Vale](https://vale.sh) package: the rules in
[STYLE.md](../STYLE.md), shared by every repository that holds documentation
pages. It needs Vale 3.24.0 or later, which lints the text inside MDX
components and runs the tests below.

| Rule | Level | What it reports |
| --- | --- | --- |
| `LanceDB.Dashes` | warning | An em dash in prose, or a double hyphen or spaced en dash used as one |
| `LanceDB.ProductNames` | warning | A misspelled LanceDB product name or one of a few third-party names |
| `LanceDB.VagueHeadings` | suggestion | A heading that names no topic, such as "Then what" |
| `LanceDB.VaguePhrases` | suggestion | A sentence about the page or the writing instead of the subject |
| `LanceDB.PlainLanguage` | suggestion | A few wordy expressions with simpler alternatives; technical nouns such as GPU utilization are left alone |
| `Vale.Repetition` | warning | A repeated word |

Code, inline code, imports, MDX expressions and comments, URLs and component
attributes are never linted. Component attributes include reader-visible
titles, such as `<Card title="...">`, so review those by eye.

## Lint pages

A repository that uses the package has a `.vale.ini` at its root naming a
release of it. Install Vale, then from the repository root:

```bash
vale sync
vale docs/web/indexing/vector-index.mdx
```

To lint only the pages your branch changes:

```bash
vale $(git diff --name-only origin/main -- '*.mdx')
```

Pull requests run the same rules with the
[Vale action](https://github.com/vale-cli/vale-action), which reports only the
lines the pull request adds or changes. Editing a line makes it yours, so an
older problem on that line is reported too. Findings are advisory and never
fail the check; a failure to install Vale, sync the package or read the
configuration still fails the job.

In this repository, `.vale.ini` uses the working copy of the package, so a
rule change applies to this repository's pages as soon as it is synced.

## Change a rule

Each rule carries its own test cases, and `tests/` holds whole Mintlify pages
linted through the package configuration. Run them from `vale/LanceDB`:

```bash
vale test --coverage styles ../tests
```

Keep the rule set small. Add a rule only for a mechanical mistake that review
keeps finding, with a case that it catches and a case that it leaves alone.

## Release

Consumers pin a package release by URL. To release updated rules, change
`version` in `LanceDB/meta.json` in the same pull request as the package changes.
Use a new stable `major.minor.patch` version for each release.

When that version change merges into `main`, the **Release Vale package**
workflow tests the package, builds `LanceDB.zip` from that exact commit and
publishes `vale-lancedb-v<version>`. The archive uses the commit timestamp so
repeated builds from that commit produce identical bytes. No local ZIP build
or personal access token is needed; the workflow uses GitHub's repository token.

Adding or changing the release workflow also starts it on `main`. Its initial
merge publishes the current `0.1.0` package if that tag does not already exist.
The workflow can also be started from the Actions page on `main` to retry a
failure before publication. It refuses to replace an existing tag or release;
a run that already published successfully does not need retrying. If a failed
run left a tag or release behind, inspect it before deciding how to recover.

After publication, update the release URL in each consumer's `.vale.ini` and
check its CI before merging. Consumer updates remain explicit; publishing a
new package does not silently change another repository's rules.

This workflow publishes only the Vale package. It does not publish staging or
production documentation, deploy Gatekeeper, or update consumer repositories.
