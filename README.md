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

Merging publishes nothing by itself. Two workflows build and check a site on
every pull request and every push to `main`, and keep the checked tree, hidden
files such as `.mintignore` included, as the run's `candidate` artifact, with a
record of the commits it was built from and a SHA-256 for every file. A
checksum over those hashes names the candidate, and the run's summary shows it.
Neither workflow can write to the repository.

- **Compose** builds the versioned site from the saved editions, as in
  [Composing the site](#composing-the-site), after checking that the saved
  editions match their records and keep those of the commit before. It needs
  no credentials. With no saved editions it composes nothing and records no
  candidate, so nothing can be published from that run.
- **Assemble** builds the assembled site from the three roots, as in
  [Development](#development): today's site, from the producers' `main`.

The Publish workflow publishes a candidate exactly as it was checked; it never
rebuilds. Each publication is a new commit on its branch that names the
candidate's checksum, its run and the commits it was built from, and nothing is
force-pushed. `scripts/candidate.py` records and publishes; `make
test-assemble` runs its tests against disposable local repositories.

### Staging

Staging is the `staging` branch, which Mintlify serves as the shared staging
preview; a preview of any other branch of this repository shows only `docs/`.
To put a candidate there, run Publish with target `staging`, the ID of a
successful Compose or Assemble run, and its candidate's checksum. A versioned
candidate must be built from a commit that `main` contains. Publish reads
`refs/heads/main` itself, for production too, so a run of a tag named `main`
is refused. An assembled candidate can come from a manual run on another
branch, for a preview before anything merges.

Once staging shows the versioned site, it follows `main` by itself: after each
successful Compose run on `main`, Publish puts the candidate of `main`'s newest
commit on staging, unless staging already shows the same files. It reads
`main` itself, so an older or repeated run never puts an older candidate
there. Until then, as today, staging changes only by hand. Publishing the first
versioned candidate there is the deliberate step that switches the shared
staging site to the versioned site; publishing an assembled candidate there by
hand stops it following `main`. Publications to staging and production run one
at a time. Pending publications queue rather than replace earlier ones, up to
GitHub’s limit of 100 pending jobs.

A push is not a deployment. Mintlify deploys the commit afterwards: the
"Mintlify Deployment" check on the commit says when, and the deployment
Mintlify records for the commit names the address it went live at. To see which
build staging shows, open the latest commit on `staging`: its message names the
candidate's checksum, its run and the docs commit, and its Mintlify check says
whether the preview has been deployed from it. Until that check succeeds, the
preview may still show the build before. Each version's "About this version"
page shows its release and sources.

### Production

Production is the `assembled` branch, and it takes only the versioned site.
After reviewing staging, run Publish with target `production` and the same run
ID and checksum. The job waits for a reviewer to approve it in the
`production` environment. Then it publishes the files staging shows, and only
if staging still shows that candidate, Mintlify's newest deployment of that
staging commit is live at the staging address, and the run's artifact shows
that the run built those files. If a newer candidate has reached staging since
the review, it stops and changes nothing: review the new one and promote that
instead. The run also reads production before it waits for approval, and
publishes nothing if production has changed by the time it is approved, so an
old run left waiting cannot replace a newer publication.

Production and staging publications share one concurrency group, so staging
cannot change through Publish between production's check of it and its push.
Staging publications wait while a production run is in the group, which can
include its wait for approval, so reject or cancel a production run you no
longer want. A push to `staging` from outside Publish is not ordered with it:
publish to `staging` only through Publish.

- The staging publication is recorded automatically, as its commit and the
  Publish run's summary.
- Mintlify records its deployment of staging, as the commit's check and a
  deployment naming the address, and production checks the deployment.
- The review of the hosted staging site is recorded by hand: it is the
  reviewer's approval of the production job, which GitHub keeps with the run.
  Say in the approval's comment what you checked.

Once Mintlify serves production from `assembled`, the "Mintlify Deployment"
check on the new `assembled` commit says when production shows it.

### Rolling back

- To go back to an earlier release, change the preferred edition in
  `compose.yaml` in a pull request. It goes through staging and approval like
  any other change, and every version's URL keeps working.
- To restore an earlier production site as a whole, run Publish with target
  `production` and the run ID and checksum named in that publication's commit
  on `assembled`. It needs approval but not staging. It removes the versions
  published since, and their URLs stop working, so decide it deliberately.

Candidate artifacts are kept for 90 days, or 7 for pull requests. A rollback
reads the earlier commit on `assembled` and checks its files against the
checksum, so it works after the artifact has expired. Promotion from staging
needs the artifact of the run that staging shows, so promote within 90 days of
that build, or first publish a newer candidate to staging by hand. Publishing
to staging by hand also falls back to an earlier staging publication of the
same candidate.

### Settings outside this repository

Publishing relies on settings outside this repository:

- A `production` environment with required reviewers, deployments limited to
  `main`, and an `ASSEMBLED_DEPLOY_KEY` secret: the private half of the only
  deploy key with write access.
- A ruleset on `assembled` that restricts updates and deletions and blocks force
  pushes, with deploy keys as its only bypass. Without it, any workflow token
  that can write could change `assembled`, and production could not trust its
  own history for a rollback.
- A repository variable `MINTLIFY_CONTENT_DIR`: the path Mintlify's Git settings
  read `docs.json` from, such as `/docs`, or `/` for the root. Publishing
  refuses to guess.
- A repository variable `MINTLIFY_STAGING_URL`: the address Mintlify serves the
  staging preview at, exactly as its deployments name it, such as
  `https://<project>-staging.mintlify.site`, with no trailing slash.
  Production takes only a staging commit that Mintlify put live there.
- Mintlify's Git settings: repository `lancedb/docs`, branch `assembled`, and
  the content directory above. Mintlify serves the branch its settings name, so
  publishing to `assembled` reaches production only once that is the branch.
- A Mintlify preview of the `staging` branch that records each deployment as a
  deployment of the commit, with a status naming that address, including for
  commits the Publish workflow pushes. Without that record, nothing can be
  promoted.
- The LanceDB Docs Reader GitHub App installed on the Enterprise documentation repository only, with
  Contents: read and the required Metadata: read permission. Set its Client ID
  in the `SOPHON_DOCS_APP_CLIENT_ID` repository variable and its PEM private key
  in the `SOPHON_DOCS_APP_PRIVATE_KEY` repository secret. The Assemble, Docs
  Check and Discover releases workflows generate a short-lived token scoped to
  that repository with Contents: read. Assemble and Docs Check do not persist
  it. Discover releases keeps it in the configuration of its own blob-less copy
  of that repository, so preparation can fetch the documentation files it
  reads. The token is revoked when the job ends. Fork pull requests receive no
  App secret and cannot run the combined build; the assembler guard tests and
  the Compose workflow still run without Enterprise repository credentials.

Of these, the Publish workflow checks only that production runs from `main`,
that the environment supplies the deploy key, that `MINTLIFY_CONTENT_DIR` and
`MINTLIFY_STAGING_URL` are set, and Mintlify's deployment of staging. It cannot
see whether the environment requires reviewers and admits only `main`, or
whether the ruleset refuses every other writer: verify those separately before
the first production publication.

Release discovery's pull requests need two more, which a dry run does not: the
repository variable `RELEASE_DISCOVERY` set to `open-prs`, and the Actions
setting that allows workflows to create pull requests. That setting also lets
workflows approve pull requests, so enable it only once `main` requires review
that a workflow cannot give, such as code-owner review.

## Saved editions

Each published documentation edition is saved in this repository: the site the
assembler built from one product release's own sources, in
`editions/<product>/<version>/`, with a release record in
`releases/<product>-<version>.yaml` that names every source it was read from.
New documentation is still written in `lancedb/lancedb` and the Enterprise
documentation repository.
[VERSIONING.md](VERSIONING.md) describes the design. `make compose` composes
the saved editions into one versioned site; see
[Composing the site](#composing-the-site), and [Publishing](#publishing) for how
it reaches staging and production.

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
  pages come from the LanceDB commit that release bundles,
  which may be one no OSS release tags: fetch it with `git -C ../lancedb fetch
  origin <commit>`. `--bundled-lancedb <commit>`, given the full 40-character
  hash, stops preparation if the release bundles a different one.
- **OSS** reads lancedb's tag `v0.41.0`. The Enterprise pages come from the
  Enterprise tag or full commit hash named with `--enterprise`; a branch name
  is not accepted, so the record pins an exact commit. The named Enterprise
  release supplies the Enterprise pages in that OSS edition; see
  [What goes into an edition](VERSIONING.md#what-goes-into-an-edition).
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

Before saving a new edition, preparation checks the release-pinned pages and
MDX snippets with the same authoring rules as composition. Write component
attributes explicitly and use literal paths for links. Property spreads such
as `<a {...props}>` and calculated links such as `href={target}` stop
preparation with a file and line in the assembled page; that line can differ
from the source because assembly expands examples. Fix the source before the
release is tagged. If a tagged release fails this check, leave it without an
edition and fix the source for a later release.
Code examples and unpublished drafts are excluded from this rule. Preparation
needs Node.js and npm and installs the pinned parser packages on first use.

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

### Discovering releases

`scripts/discover.py` finds the releases that should get an edition, prepares
each one as above, and opens a pull request for it. People review and merge
those pull requests; discovery never merges or publishes.

- **Enterprise:** a release is eligible once its `vX.Y.Z` tag and an annotated
  `release-qualified/vX.Y.Z` tag name the same commit. A qualification tag
  without a release tag is not eligible; a release tag without a qualification
  tag stops for review. Discovery reads both tags from the Enterprise
  repository on every run, so a withdrawn qualification stops a release even
  in a clone that still has the old tag.
- **OSS:** a release is eligible once `lancedb/lancedb` has a published GitHub
  release for its `vX.Y.Z` tag that is not marked as a prerelease. Its
  Enterprise pages come from the preferred Enterprise edition, at the release
  its record names, so OSS releases wait until `compose.yaml` prefers one.
- Betas and release candidates never get editions. Discovery counts them, and
  Enterprise releases that are qualified but not yet published, without naming
  them.
- GitHub's "latest release" is not used: it can be a maintenance release of an
  older line.

`discover.yaml` configures it. Each product's `from` is the first release
discovery prepares; it never prepares an earlier one, so there is no backfill,
and a product with no `from` is off. `skip` lists releases to leave alone, each
with its reason, and `holds` lists publication holds that every edition pull
request repeats.

For each release from `from` on, discovery looks at `main` and at the branch
`editions/<product>-<version>`:

- **Saved on `main`:** nothing to do, corrected or not, while its tags still
  name the commit and qualification tag its record names. If they do not, it
  stops for review: an edition is never replaced.
- **An open pull request from the branch:** kept as it is, with any commits
  reviewers have pushed. Discovery starts each check workflow that has not run
  on its head, and never reruns one that has.
- **A pull request closed without merging:** left closed. Reopen it to prepare
  the release again.
- **The branch without a pull request,** as a run that stopped after pushing
  leaves it: discovery opens the pull request from it.
- **None of these:** discovery prepares the edition with `edition.py prepare`
  on a branch from `main`, runs `edition.py check` against `main`, pushes the
  branch and opens its pull request. For a product's first edition it also sets
  `preferred` in `compose.yaml`, since composing needs one. It never changes a
  preferred edition otherwise, so a maintenance or late release cannot replace a
  newer one; the pull request says whether the release is newer than the
  preferred edition, for its reviewers to decide.

Discovery creates a branch only if it does not exist yet, so two runs never
open two pull requests or overwrite each other's work. It prepares a release
only from its tags as it checked them, and checks the prepared record before
pushing, so a tag that moves while it runs stops that release. A release whose
inputs are missing or contradictory stops for review with the reason, and
leaves `main`, the saved editions and the other releases as they are. Any stop
fails the run, until the release is fixed or listed in `skip`.
If a tagged release fails the authoring check, list it in `skip`; it gets no
edition. Fix the source for a later release instead of moving the tag.

A dry run reports what discovery would do: each eligible release's exact
inputs, and why the others are waiting, skipped or stopped. It fetches release
tags and commits into the two checkouts and reads GitHub with `gh`, but pushes,
opens and starts nothing. Run it in a clone set up as for preparing, with
`--config` to try a configuration before committing it:

```bash
uv run --no-project --with pyyaml scripts/discover.py --lancedb ../lancedb --enterprise-repo ../enterprise
```

The Discover releases workflow runs it when started by hand, as a dry run by
default; nothing starts it on a schedule yet. Its `open-prs` mode prepares
editions and opens their pull requests. It runs only on `main`, once the
repository variable `RELEASE_DISCOVERY` is `open-prs`. It pushes and opens pull
requests with the workflow's own token, whose pushes start no workflows, so it
starts Compose and Assemble on each branch itself. The run's summary is as
public as this repository.

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

The Compose workflow runs `make check-editions` with a base: for a pull
request, the base commit its event names, and for a push to `main`, the commit
`main` pointed at before. The job first fetches that commit by hash
with its own GitHub token; `edition.py check` then runs offline and needs no
Enterprise repository credentials. A manual run, as release discovery starts
on its branches, checks out the branch rather than a merge, so its base is the
commit of `main` the branch was made from.

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

Preparing that release again later checks its release inputs and saved files,
then keeps the edition and its corrections. It does not fetch docs main or
require matching assembler scripts unless it is building a new edition.

### Composing the site

`make compose` composes every saved edition into one site in
`build/versioned`, with Mintlify's own product and version selectors. Like
Mint, it needs Node.js and npm: the first run installs the pinned packages of
`scripts/mdx` into its `node_modules`. Check and preview the site as you would
the assembled site:

```bash
make compose
(cd build/versioned && mint validate && mint broken-links --check-anchors --check-redirects)
cd build/versioned && mint dev
```

Each edition is published as it was saved, under
`/<product>/v<major>-<minor>-<patch>/`, such as
`/enterprise/v1-10-0/quickstart`: its pages, assets, navigation, availability
badges, redirects and drafts. Nothing is rebuilt from product sources, and an
edition that does not match its record is not composed. Each edition also gets
an "About this version" page, made from its record and listed last in its first
navigation group, with its release, the commits its pages come from and its
corrections.

`compose.yaml` lists the products in selector order, with their labels, and
names each product's preferred edition. A product's versions are listed newest
first, each under its version number, such as 1.10.0, and the preferred one is
the default, tagged Latest. Readers land on the first product's preferred
edition, the site's default edition. A preferred edition changes only when
`compose.yaml` does, in a reviewed pull request; preparing a newer release, or
a maintenance release of an older line, never changes it. A product with saved
editions must name one.

Use explicit paths for internal documentation links, such as `/guide`, so they
stay within the selected version. Code examples are preserved as written.

Composition reads each page and snippet module with the MDX parser Mint uses,
pinned in `scripts/mdx`, and changes only the link destinations it finds:

- Root-relative Markdown links, images and link definitions, the `href`, `src`
  and icon paths of components, including components that pages and snippets
  export, frontmatter, navigation and imports get the edition's path. Dot
  segments are normalized first.
- A link to `docs.lancedb.com` becomes a link to the page it lands on in the
  same edition, after that edition's redirects.
- Same-site redirect destinations stay within the edition, including absolute URLs.
- Relative links stay as written.
- Code blocks, inline code, JavaScript strings and comments are never
  rewritten.

A link attribute that JavaScript calculates, such as `href={page.url}` or
``href={`/guides/${name}`}``, cannot be pointed into the edition, so composition
stops there with the page and line: write the path itself, as `href="/guide"`
or `href={"/guide"}`. Component property spreads such as `<a {...props}>` are
also refused: write the attributes explicitly instead. This applies to
rendered components, including exported components, but not to code examples.
Composition also stops, naming the page and line, at a
link into another edition, a path that leaves its edition, a root-relative or
`docs.lancedb.com` link written with escapes or character references, a
published page or snippet that is not valid MDX, or a bare `docs.lancedb.com`
URL in the text, which Mintlify would link to the default edition; write it as
a Markdown link. A relative link in a snippet cannot climb with `..`, since it
is relative to each page that imports the snippet. A link to something the
edition does not publish is a warning, and `mint broken-links` checks the
composed site.

Redirect sources must be plain paths or contain one `:name*` segment. Other
patterns, including partial wildcards such as `/article-*`, stop composition
with an error; use explicit paths instead. Redirect checks follow redirects
before pages, and reject loops and chains longer than 100 hops.

An example that downloads a file from `docs.lancedb.com` keeps the URL it
shows, and the file is also served there, at its unversioned path. Every
edition whose code downloads it must hold the same file, so a newer release
cannot change it beneath an older example. If they differ, composition stops:
give the newer example its own URL in its source.

When changing the composer or these authoring rules, compose all saved editions
before merging. Do not silently rewrite an older edition to satisfy a new rule;
any necessary content change uses the reviewed correction process.

Site-wide settings live in `site/settings.json`: the theme, banner, logos,
navbar, footer and integrations. Edit that file to change the shared site
without creating a release or correcting an edition. `compose.yaml` still
chooses the preferred editions; changing that choice does not change these
settings.

Keep shared logos, fonts, stylesheets, scripts and their dependencies under
`site/static/`. Composition copies that whole folder to `/static/` unchanged,
so local stylesheet paths keep working. For example, `site/static/style.css`
can use `url("/static/hero.svg")` when the image is saved at
`site/static/hero.svg`. Anything in this folder is public. Pages do not belong
there, and symbolic links are refused.

Mintlify loads every stylesheet and script on every page, so the composer
uses only the copies in `site/static/`, not those saved in editions. Saved
editions still supply their pages, page assets and version navigation.
`navigation` and `redirects` cannot be set in `site/settings.json`; composition
generates them. A shared asset cannot replace a different file used by an
edition's unversioned example download. The composed candidate records the
docs commit containing these settings and assets, so promotion and rollback
reuse the exact shared files too.

These settings apply to the versioned site. The existing unversioned Assemble
workflow continues to take its settings from the product source.

Each edition's redirects stay within it. Every unversioned path that some
edition publishes, as a page or a redirect, redirects to the same page in the
default edition, and `/` goes there too. Those redirects are temporary, so
readers follow a change of preferred edition. A path the default edition does
not have gets no redirect, because a catch-all would hide the missing page:
`make compose` lists it for review. Each edition's `.mintignore` entries are
pointed at its own folder, so its drafts stay hidden there and nowhere else.

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
