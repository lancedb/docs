# LanceDB Documentation

Home of the [LanceDB](https://lancedb.com/) documentation. Built using [Mintlify](https://www.mintlify.com/).

## Development

The published site is assembled from three roots, listed in `assemble.yaml`:
the open-source pages in `lancedb/lancedb`, the Enterprise documentation,
both under `docs/web`, and this repository's `docs/`.

This repository's `docs/` holds the pages it still owns: the Geneva pages, the
dataset cards and the REST API reference. The four Geneva snippets
(`docs/snippets/geneva_*.mdx`) are frozen copies from `deploy-freeze`; the tests
that generated them need the `geneva` package and are not in this repository,
so the snippets are not regenerated.

Install the [Mintlify CLI](https://www.npmjs.com/package/mint) at the version CI
uses:

```bash
npm i -g mint@4.2.888
```

With the three repositories checked out side by side, build and preview the
site:

```bash
make assemble
cd build/site && mint dev
```

To build from other checkouts or worktrees, point `LANCEDB_DOCS_ROOT` and
`SOPHON_DOCS_ROOT` at their `docs/web` directories. The assembler prints the
commit it read each root from, so every build names its inputs.

Check the assembled site the way CI does, and run the assembler's own tests:

```bash
(cd build/site && mint validate && mint broken-links --check-anchors --check-redirects)
make test-assemble
```

`cd docs && mint dev` previews only this repository's pages. Links into the
open-source and Enterprise pages do not resolve there, so check links on the
assembled site.

### Availability

A page in any root can declare in its frontmatter whether LanceDB OSS and
LanceDB Enterprise have what it describes. The Documentation section of
lancedb's `CONTRIBUTING.md` describes the `availability` key. For each page
that declares it, the assembler:

- puts a label, the offerings' badges and the page's one-sentence summary,
  directly after the frontmatter, ahead of any imports;
- adds a sidebar `tag` when only one offering has it;
- leaves `availability` out of the published frontmatter.

A line `{/* availability-comparison */}` in a page becomes the pages that
declare availability, in navigation order: for each navigation section -- the
outermost group holding a page, or its tab -- a `###` heading and a table of
that section's pages, so the line belongs under a `##` heading. Pages outside
the navigation come last, under "Other pages". Declarations are read from
the page the site publishes, so an overlay's frontmatter replaces its reference
page's. A malformed declaration, a `tag` set by hand, a page that opens with a
badge of its own, or a misplaced comparison line stops the build. Pages that
declare nothing are copied unchanged.

The `oss` and `enterprise` values must be `available` or `unavailable`. They
describe product support. Setup switches, permissions, supported SDK or server
versions, and narrower subfeature limits belong in the summary or page body;
they do not make a supported feature `varies`. Unknown or legacy `varies`
values stop assembly rather than being silently converted to support claims.
Availability does not decide whether a page is published; see
[Writing docs before release](#writing-docs-before-release).

When migrating older sources, update their declarations and qualifications
before using this stricter assembler with them. Binary declarations also work
with the previous assembler. CI reads producer `main` by default; explicit
ref overrides must point at migrated sources. An older producer ref containing
`varies` intentionally fails with a page and offering in the error, so update
that ref before assembling. Keep availability source-owned; do not add a second
feature map to the generator.

`make test-assemble` also compiles its labelled test pages with the MDX
compiler Mint uses when `MDX_COMPILER` names the `index.js` of the
`@mdx-js/mdx` package installed with Mint and `node` is on the path; otherwise
those tests are skipped.

### Writing docs before release

Documentation can be written before the feature it describes is released,
without presenting it as available. Use Mintlify's own controls and ordinary
edits. They change what the documentation shows, not what a server does, and
they do not keep anything confidential: drafts stay in the source and can
appear in the built site's page data.

- **Draft pages.** List the page in `.mintignore`, beside `docs.json` in
  lancedb's `docs/web`; the assembled site uses that file for pages from every
  root. Keep the page out of the navigation and out of redirects, and do not
  link to it from published pages. `mint validate` reports navigation to an
  ignored page, and `mint broken-links --check-redirects` reports links and
  redirects to one.
- **Draft sections.** Wrap the section in an MDX comment whose first line names
  the release:

  ```mdx
  {/*
  Draft for the Feature X release.

  ## Feature X {#feature-x}

  Feature X does ...
  */}
  ```

  Give a comment lines of its own, with a blank line before and after it, or
  start it at the end of a line of text. A comment on a line of its own in the
  middle of a paragraph splits the paragraph. Inside a component, indent the
  comment with the component's children. A comment ends at the first `*/`.
- **Tables and lists.** A comment between table rows breaks the table, and one
  between list items breaks the list. Put draft rows or items in a comment
  below the table or list, and say where they go:

  ```mdx
  | Option | Effect |
  | --- | --- |
  | `a` | Available now. |

  {/*
  Draft for the Feature X release. Add this row to the table above:
  | `x` | Turns on Feature X. |
  */}
  ```
- **Preview.** `mint dev` and assembled builds show what readers will see. To
  check how a draft renders, remove its comment or its `.mintignore` entry in a
  disposable copy, or in a local change you do not commit, then build and
  preview as above.
- **Release.** In the pull request that publishes the documentation, remove the
  comments and their notes, remove the `.mintignore` entries, add the pages to
  the navigation, restore their links and redirects, and run the checks above.

## Publishing

Merging publishes nothing. Every pull request and every push to `main` runs the
Assemble workflow, which builds the site from the three roots, checks it as
above, and keeps the checked tree, hidden files such as `.mintignore` included,
as the run's `candidate` artifact, with a record of the commit each root was
read from and a SHA-256 for every file. A checksum over those hashes names the
candidate, and the run's summary shows it.
That workflow's token can only read.

Publishing is a separate, manual step: run the Publish workflow with the ID of a
successful Assemble run and its candidate's checksum.

- `staging` puts the candidate on the `staging` branch, for a Mintlify preview
  of the combined site. A preview of any other branch of this repository is not
  one: it holds only `docs/`.
- `production` puts it on `assembled`. It takes only a candidate built on `main`
  from both producers' `main`, runs only from `main`, and pushes with the
  `production` environment's deploy key. It waits for approval only if that
  environment requires reviewers.

Either way the published files are the candidate's, checked against its record
and the checksum, and the record must name exactly the three source commits and
both producers' refs. Nothing is rebuilt, so newer source commits cannot slip in.
Each publication is a new commit on its branch that names the run, the checksum
and the source commits, and nothing is force-pushed. To roll back, publish an
earlier candidate again: its run ID and checksum are in its commit on
`assembled`. Once its artifact has expired, after 90 days, that earlier commit's
files are published again, after they are checked against the checksum.

`scripts/candidate.py` records and publishes; `make test-assemble` runs its
tests against disposable local repositories.

Publishing relies on settings outside this repository:

- A `production` environment with required reviewers, deployments limited to
  `main`, and an `ASSEMBLED_DEPLOY_KEY` secret: the private half of the only
  deploy key with write access.
- A ruleset on `assembled` that restricts updates and deletions and blocks force
  pushes, with deploy keys as its only bypass. Without it, any workflow token
  that can write could change `assembled`.
- A repository variable `MINTLIFY_CONTENT_DIR`: the path Mintlify's Git settings
  read `docs.json` from, such as `/docs`, or `/` for the root. Publishing
  refuses to guess.
- Mintlify's Git settings: repository `lancedb/docs`, branch `assembled`, and
  the content directory above. Mintlify serves the branch its settings name, so
  publishing to `assembled` reaches production only once that is the branch.
- The LanceDB Docs Reader GitHub App installed on the Enterprise documentation repository only, with
  Contents: read and the required Metadata: read permission. Set its Client ID
  in the `SOPHON_DOCS_APP_CLIENT_ID` repository variable and its PEM private key
  in the `SOPHON_DOCS_APP_PRIVATE_KEY` repository secret. The Assemble and Docs
  Check workflows generate a short-lived token scoped to that repository with
  Contents: read. Checkout does not persist it, and the token is revoked when
  the job ends. Fork pull requests receive no App secret and cannot run the
  combined build; the assembler guard tests and the saved-edition check still
  run without Enterprise repository credentials.

Of these, the Publish workflow checks only that production runs from `main`,
that the environment supplies the deploy key and that `MINTLIFY_CONTENT_DIR` is
set. It cannot see whether the environment requires reviewers and admits only
`main`, or whether the ruleset refuses every other writer: verify those
separately before the first production publication.

## Saved editions

Each published documentation edition is saved in this repository: the site the
assembler built from one product release's own sources, in
`editions/<product>/<version>/`, with a release record in
`releases/<product>-<version>.yaml` that names every source it was read from.
New documentation is still written in `lancedb/lancedb` and the Enterprise
documentation repository.
[VERSIONING.md](VERSIONING.md) describes the design. Composing the saved
editions into the site, and publishing them, are later steps; for now the site
is assembled and published as above.

An edition's content comes from two kinds of commit, and its record names both:

- **Product pages** come from the release: an OSS edition's from its
  `lancedb/lancedb` release tag, and an Enterprise edition's from its
  Enterprise release tag and the LanceDB commit that release bundles.
- **Docs-owned content**, the Geneva pages, the dataset cards and the REST API
  specification, comes from the docs input commit: the commit of this
  repository's `main` that preparation read, recorded as `sources.build.commit`.
  It is `main` as it was when the edition was prepared, not when the product
  was released.

Both are fixed when the edition is prepared. Later changes to `main`, or to a
product's sources, never reach a saved edition; only a reviewed correction
changes one.

### Preparing an edition

Fetch the release tags into the LanceDB and Enterprise checkouts. Then run the
command in a clone of this repository whose `origin` is
`https://github.com/lancedb/docs.git`, or its SSH form, rather than a fork, on
a branch whose `scripts/` are those of `main`:

```bash
uv run --no-project --with pyyaml scripts/edition.py prepare enterprise 1.10.0 --lancedb ../lancedb --enterprise-repo ../enterprise
uv run --no-project --with pyyaml scripts/edition.py prepare oss 0.41.0 --enterprise v1.10.0 --lancedb ../lancedb --enterprise-repo ../enterprise
```

- **Enterprise** reads the Enterprise release tag `v1.10.0`, which must have an
  annotated `release-qualified/v1.10.0` tag on the same commit. The shared
  pages come from the LanceDB commit that release bundles at `src/lancedb`,
  which may be one no OSS release tags: fetch it with `git -C ../lancedb fetch
  origin <commit>`. `--bundled-lancedb <commit>`, given the full 40-character
  hash, stops preparation if the release bundles a different one.
- **OSS** reads lancedb's tag `v0.41.0`. The Enterprise pages come from the
  Enterprise tag or full commit hash named with `--enterprise`; a branch name
  is not accepted, so the record pins an exact commit. Pinning a commit does
  not decide that it may be published with that OSS release, which is not
  settled yet; see [Open questions](VERSIONING.md#open-questions).
- The LanceDB and Enterprise roots are read from Git at their commits, never
  from a working tree, and only from `docs/web`.
- Preparation fetches `main` from `origin` itself, and reads this
  repository's `docs/` and `assemble.yaml` from Git at its head, or at an
  earlier commit of `main` named with `--docs`. It never reads them from your
  branch, your working tree or an `origin/main` fetched before. A fork's `main`
  can hold commits `lancedb/docs` never has, so an `origin` that is not
  `lancedb/docs` stops preparation. Squash merges give a branch's commits new
  hashes, so a record names only a commit of `main`, and editions prepared one
  after another in the same pull request do not depend on its unmerged
  commits. The assembler that runs is your checkout's `scripts/`, so they must
  be that commit's.
- Only stable versions, such as `1.2.3`, have editions.

Preparation saves nothing, and says why, when a tag or commit is missing, a
release has no `docs/web`, an Enterprise release is not qualified or bundles
another LanceDB, a root holds a symlink, the Enterprise root holds a hidden
file, the assembler stops, `.gitignore` would leave out one of the edition's
files, `origin` is not `lancedb/docs`, the docs input commit is not on its
`main`, or your `scripts/` are not that commit's.

Preparing a saved release again changes nothing when its release commits and
qualification tag match the record, even after the edition has been corrected,
and stops when they differ. It never replaces a saved edition.

Check the new edition as a site, check every edition against its record and
against `main`, and commit the edition and its record together:

```bash
(cd editions/enterprise/1.10.0 && mint validate && mint broken-links --check-anchors --check-redirects)
uv run --no-project --with pyyaml scripts/edition.py check --base refs/remotes/origin/main
```

A record looks like this. It is illustrative: the Enterprise repository is a
placeholder and the commits and checksums are shortened, so `check` would
refuse it as written.

```yaml
format: 1
product: enterprise
version: 1.10.0
edition: editions/enterprise/1.10.0
release:
  repository: <enterprise-repository>
  tag: v1.10.0
  commit: 5a1f0c9e…
  qualification:
    tag: release-qualified/v1.10.0
    object: 9c3e27d4…
  bundled_lancedb: 7d0b4a12…
sources:
  lancedb:
    repository: lancedb/lancedb
    path: docs/web
    commit: 7d0b4a12…
  enterprise:
    repository: <enterprise-repository>
    path: docs/web
    commit: 5a1f0c9e…
  build:
    repository: lancedb/docs
    path: docs
    commit: e2c4b8f0…
  openapi:
    repository: lance-format/lance-namespace
    release: v0.12.0
    path: docs/src/spec.yaml
    sha256: 41ab93c7…
prepared:
  files: 312
  sha256: b07d5e61…
corrections: []
```

`release` and `sources` say what the edition was built from, and `prepared` is
the snapshot as first saved. An OSS record has `enterprise_pages`, the
`--enterprise` ref and its commit, in place of `qualification` and
`bundled_lancedb`. `sources.openapi` names the specification release that
`assemble.yaml` pinned at the docs input commit, and the checksum of the copy
saved in the edition.

### What `check` enforces

`edition.py check` works offline, on this checkout:

- Every record has the fields `prepare` writes, in their formats, and each
  source commit agrees with the release field that selected it, such as an
  Enterprise record's `sources.lancedb.commit` and `release.bundled_lancedb`.
- Every record has its edition, and every edition its record.
- Each edition matches the checksum its record last accepted:
  `prepared.sha256`, or the latest correction's `sha256`.

With `--base COMMIT`, it also compares the editions and records already on
that commit, by their parsed values rather than their formatting. Each must
still be there. Every field of a record except `corrections` must be
unchanged, and its existing corrections must be unchanged and in order, with
new ones only after them. A new edition is checked by the rules above.

The test job of the Assemble workflow runs `make check-editions` with a base:
for a pull request, the base commit its event names, and for a push to `main`,
the commit `main` pointed at before. The job first fetches that commit by hash
with its own GitHub token; `edition.py check` then runs offline and needs no
Enterprise repository credentials. A manual run of the workflow has no base,
and checks each edition against its record only.

This keeps saved editions only while the check must pass before a pull request
merges, and while changes to the checker and its workflow need enforced review,
such as required code-owner review of `scripts/`, the `Makefile` and
`.github/workflows/`, since a pull request runs its own copy of the check. Both
are repository rules set up separately. The check on a push to `main` reports
a violation in that push but cannot undo it, and a later push does not find it,
because its base already includes the change.

### Correcting a saved edition

A saved edition can be corrected after review, at the same URL:

1. Edit the affected files in each edition that has the error, and in no other.
2. Run `edition.py check`. It fails, because the edition no longer matches its
   record, and prints the corrected edition's checksum:

   ```text
   edition: editions/enterprise/1.10.0 has changed since it was prepared; its files now have checksum 3f9c2a7e…. For a reviewed correction, add an entry under `corrections` in releases/enterprise-1.10.0.yaml with `sha256: 3f9c2a7e…`
   ```

3. In each edition's record, add an entry at the end of `corrections` with a
   one-line summary, the files it changes, and that checksum; for the first
   correction, replace `corrections: []`. Leave the rest of the record, and
   any earlier corrections, as they are.

   ```yaml
   corrections:
   - summary: Fix the default retry count in the merge-insert example.
     files: [tables/merge-insert.mdx]
     sha256: 3f9c2a7e…
   ```

4. Run the checks above, with `--base refs/remotes/origin/main` after fetching `main`.
   `edition.py check` now passes: each edition matches the checksum of its
   latest correction. Any later change to the edition, including a deleted
   file, fails again until it is recorded the same way, as a further entry.
   `--base` refuses a change to an earlier entry or to the rest of the
   record, as the pull request's check does. Check a corrected example the way
   that example needs; a typo needs no test run.
5. If the error is still in the current source, in `lancedb/lancedb`,
   the Enterprise documentation repository or `docs/` here, fix it there as well.
6. Open one pull request that names every edition it corrects. Reviewers decide
   whether each correction is right; the checksum shows only that the record
   describes the files.

Preparing that release again later keeps the edition and its corrections.

## Code snippets

The code examples on the open-source pages are tested programs in
`lancedb/lancedb`, under `docs/web-tests/{py,ts,rs}`. Their snippets are
generated there, into `docs/web/snippets/`, and committed beside the tests they
come from; this repository generates none. To change an example, edit its test
in a lancedb checkout and regenerate from that repository's root:

```bash
uv run docs/web-tests/mdx_snippets_gen.py -s docs/web-tests/py -s docs/web-tests/ts -s docs/web-tests/rs -o docs/web/snippets
```

The Documentation section of lancedb's `CONTRIBUTING.md` describes the same
workflow. Prefer an example in a test over code written into a page.

The only snippets in this repository are the four Geneva ones,
`docs/snippets/geneva_*.mdx`. They are frozen copies, as published at
`deploy-freeze`, and stay unchanged for as long as the Geneva pages are
published. `make snippets` stops and prints these instructions.

The assembler writes each example into the page that shows it, as an ordinary
fenced code block with the same title and icon, so the page's Markdown export
carries the code itself rather than the name of a snippet constant. Snippet
modules are not published once nothing imports them.

## Sync Hugging Face dataset pages

The `Datasets` tab is populated from [`lance-format/lance-huggingface`](https://github.com/lance-format/lance-huggingface),
the master repository where each Lance dataset published under the [`lance-format`](https://huggingface.co/lance-format)
Hugging Face organization has its own directory with an `HF_DATASET_CARD.md`. That same file is what gets pushed to
the Hub as the dataset's `README.md` via the `hf` CLI, so the GitHub repo is the single source of truth for the
content of every dataset card.

To avoid maintaining the same content in two places, the per-dataset MDX pages under `docs/datasets/` are
generated from those upstream cards via `scripts/sync_hf_datasets.py`. The script:

1. Reads `scripts/hf_datasets.yaml`, which lists every dataset to publish and maps the upstream directory name,
   the URL slug, the HF Hub repo, and the human-readable title.
2. Fetches each `HF_DATASET_CARD.md` from `lance-format/lance-huggingface` on GitHub.
3. Rewrites the frontmatter for Mintlify (sets `title`, `sidebarTitle`, `description`), strips the upstream H1,
   injects a "View on Hugging Face" card at the top, and sanitizes known MDX hazards (bibtex citations outside
   code fences, literal `<>` in prose).
4. Writes `docs/datasets/<slug>.mdx`, regenerates the card grid in `docs/datasets/index.mdx` between the
   `HF_SYNC:START` / `HF_SYNC:END` markers, and updates the `Datasets` tab in `docs/docs.nav.json` to keep the
   sidebar in sync.

Run it from the repo root:

```bash
make hf-sync
```

### Adding a new dataset

1. Author the new dataset's `HF_DATASET_CARD.md` upstream in `lance-format/lance-huggingface` (and push it to the
   Hub as usual).
2. Add a single line for the dataset under the appropriate category in `scripts/hf_datasets.yaml`. The four
   fields (`dir`, `slug`, `hf`, `title`) are explicit because the GitHub directory name, the HF Hub repo slug,
   and the desired URL slug don't follow a derivable convention.
3. Run `make hf-sync`. The script will fetch the new card, generate `docs/datasets/<slug>.mdx`, refresh the
   landing-page card grid, and add the new page to the `Datasets` tab in `docs/docs.nav.json`.
4. Run `make assemble`, then preview with `cd build/site && mint dev` (see [Development](#development)
   for the required source checkouts). Commit the MDX page, regenerated `index.mdx`, updated
   `docs.nav.json`, and new yaml entry.

If you remove a dataset from the yaml, the next `make hf-sync` will delete its MDX file and drop the sidebar
entry. The script hard-fails on any fetch error — partial regeneration would be worse than a clear error.
