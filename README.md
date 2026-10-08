# LanceDB Documentation

Home of the [LanceDB](https://lancedb.com/) documentation. Built using [Mintlify](https://www.mintlify.com/).

## Development

The published site is assembled from three roots, listed in `assemble.yaml`:
the open-source pages in `lancedb/lancedb` and the Enterprise pages in
`lancedb/sophon`, both under `docs/web`, and this repository's `docs/`.

This repository's `docs/` holds the pages it still owns: the Geneva pages, the
dataset cards and the REST API reference. The Geneva pages stay published until the official Function
launch, because the Function pages do not replace their APIs. They and their
snippets (`docs/snippets/geneva_*.mdx`) are kept as published at
`deploy-freeze`; the tests that generated those snippets need the `geneva`
package and are not in this repository, so the snippets are not regenerated.

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
  above the page's content, after its imports;
- adds a sidebar `tag` when only one offering has it;
- leaves `availability` out of the published frontmatter.

A line `{/* availability-comparison */}` in a page becomes a table of every
page that declares availability, in navigation order. Declarations are read from
the page the site publishes, so an overlay's frontmatter replaces its reference
page's. A malformed declaration, a badge of the page's own above a label, or a
misplaced comparison line stops the build. Pages that declare nothing are copied
unchanged.

## Publishing

Merging publishes nothing. Every pull request and every push to `main` runs the
Assemble workflow, which builds the site from the three roots, checks it as
above, and keeps the checked tree as the run's `candidate` artifact, with a
record of the commit each root was read from and a SHA-256 for every file. A
checksum over those hashes names the candidate, and the run's summary shows it.
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
- The LanceDB Docs Reader GitHub App installed on `lancedb/sophon` only, with
  Contents: read and the required Metadata: read permission. Set its Client ID
  in the `SOPHON_DOCS_APP_CLIENT_ID` repository variable and its PEM private key
  in the `SOPHON_DOCS_APP_PRIVATE_KEY` repository secret. The Assemble and Docs
  Check workflows generate a short-lived token scoped to Sophon with
  Contents: read. Checkout does not persist it, and the token is revoked when
  the job ends. Fork pull requests receive no App secret and cannot run the
  combined build; the assembler guard tests still run without credentials.

Of these, the Publish workflow checks only that production runs from `main`,
that the environment supplies the deploy key and that `MINTLIFY_CONTENT_DIR` is
set. It cannot see whether the environment requires reviewers and admits only
`main`, or whether the ruleset refuses every other writer: verify those
separately before the first production publication.

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
`deploy-freeze`, and stay unchanged until the Geneva pages are retired at the
official Function launch. `make snippets` stops and prints these instructions.

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
