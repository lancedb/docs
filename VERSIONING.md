# Versioned documentation

Readers should find the documentation for the product version they use.
Engineers keep writing documentation beside the product code, in
`lancedb/lancedb` and the Enterprise documentation repository. Release
automation prepares each version's documentation, and people review it before
it is merged and before it reaches production.

This is the design that work follows. It is built in four steps:

1. **Release records and saved editions.** A record names one release's
   sources, and a manual command saves the edition built from them. This step
   is in place: see [Saved editions](README.md#saved-editions).
2. **Composing editions into the site:** version paths, links, navigation,
   selectors, redirects and an "About this version" page for each edition.
3. **Supervised publication:** merged editions and corrections go to staging
   automatically, and a person approves promoting that same candidate to
   production.
4. **Release discovery:** each product's own qualification signal starts the
   same preparation, and a bot opens the documentation pull request.

Until steps 2 and 3 are in place and a cutover is approved, the site is built
and published as it is today, and production stays where it is.

## Decisions

| Topic | Decision |
| --- | --- |
| Hosting | Mintlify, with its native product and version selectors. |
| Default | The latest approved, published Enterprise edition. |
| Unversioned links | Resolve to the same page in that edition. Versioned links stay in their edition. |
| Navigation | Every edition keeps the full documentation set, with OSS and Enterprise badges. |
| Release cadence | OSS and Enterprise editions publish independently. An Enterprise release needs no new OSS release. |
| Enterprise sources | An Enterprise edition uses its own pages and the shared pages of the exact LanceDB it bundles, which may be an older release or a commit no release tags. |
| History | No backfill. Editions start with releases that have their documentation sources. |
| Storage | One folder per edition, all on one branch of this repository. |
| Corrections | A reviewed edit to a saved edition is allowed for a historical correction. Fix the current source too when the error is still there. |
| Automation | Preparation, checks, pull requests and staging are automated. People review before merging and approve production promotion. |
| Missing page on version switch | Mintlify's native fallback, with no custom switching logic. |

"Latest" means the approved preferred edition of a product, not whichever job
finished last. A later maintenance release must not replace a newer preferred
release line. Selecting the latest Enterprise edition does not select the
latest OSS edition: the LanceDB it bundles can be older.

## Components

```mermaid
flowchart TD
    L["lancedb/lancedb<br/>Shared pages and examples"] --> B
    S["Enterprise source<br/>Enterprise pages and bundled LanceDB"] --> B
    D["This repository<br/>Docs-owned pages"] --> B
    R["Release record<br/>Version and exact source commits"] --> B
    B["Edition builder<br/>Saves a new edition"] --> E
    E["Saved editions"]
    C["Reviewed correction"] --> E
    E --> A["Site composition<br/>Version paths, links and navigation"]
    N["Shared site configuration<br/>Branding and preferred editions"] --> A
    A --> V["Checked site candidate"]
    V --> T["Staging"]
    T --> H["Production approval"]
    H --> P["Promote the same candidate"]
```

These are parts of the existing tooling, not new services.

- **Edition builder** (`scripts/edition.py`) reads the recorded commits,
  builds the edition with the existing assembler, and saves it. The assembler
  stays where pages are composed: overlays, examples written into pages,
  availability badges and native drafts all come from it.
- **Saved edition**: the pages, assets, navigation and draft controls of one
  release. Later builds never fetch fresh content into it.
- **Site composition** gives each saved edition its version path and builds
  the navigation and redirects. It does not rebuild editions.
- **Publisher** puts the checked candidate on staging and, after approval, the
  same candidate on production.

Global branding and Mintlify's renderer change independently of editions, so
a saved edition keeps its content, not its exact pixels.

## Editions and their records

```text
releases/
  oss-<version>.yaml
  enterprise-<version>.yaml
editions/
  oss/<version>/
  enterprise/<version>/
```

A record names the product and version, the release tag and commit, the
Enterprise qualification tag, the LanceDB an Enterprise release bundles, the
commit of every root the assembler read, and the specification's upstream
release.

An edition's sources are fixed when it is prepared:

- **Product pages** come from the release commits: the OSS release, or the
  Enterprise release and the LanceDB it bundles.
- **Docs-owned content**, the Geneva pages, the dataset cards and the REST API
  specification, comes from the docs input commit: the commit of this
  repository's `main` that preparation read, which the record names with the
  others. It is `main` as it was when the edition was prepared, not when the
  product was released.
- Later changes to `main`, or to a product's sources, never reach a saved
  edition, so older editions never inherit today's pages.

Saved editions are kept by default but are not immutable. A reviewed
correction changes an edition's files at the same URL. The record keeps the
original sources and the checksum of the snapshot as prepared, and lists each
correction with the checksum of the edition it accepted; the saved edition
must match the latest one. Git history holds the correction pull requests.
Publication checksums describe the corrected candidate.

A pull request may add editions, and add corrections at the end of a record.
It may not remove an edition or a record, change a record's release, sources
or prepared snapshot, or change an earlier correction. The Assemble workflow's
test job checks this offline, on every pull request against its base commit
and on every push to `main`, by comparing the records' parsed values. That
protects saved editions only while the check is required before merging, a
repository rule set up separately. On a push to `main` it can report a
violation but not undo it, and later pushes no longer see it.

Preparation never overwrites a saved edition:

- The same release prepared again from the same release commits and
  qualification tag changes nothing, even after the edition has been
  corrected. A later commit of this repository's `main` does not make it a
  different release.
- Different inputs for a saved release, such as a moved tag, another
  Enterprise source, or a qualification tag deleted and created again, stop
  preparation for review. A recreated qualification tag is new qualification
  evidence, not a duplicate.
- A failed preparation leaves no edition and no record.

## Preparing a release

The manual command and later automatic discovery run the same preparation.
Nobody assembles files by hand.

**Enterprise.** Preparation takes the Enterprise release tag, which must
be qualified, reads the LanceDB commit that release bundles, and takes the
shared pages from that commit. A standalone OSS edition is not needed.

**OSS.** Preparation takes the `lancedb/lancedb` release tag. The full
navigation includes the Enterprise pages, so the Enterprise tag or commit
they come from is named explicitly. Naming it pins the source; it does not
decide that the source may be published with that OSS release, which is still
open. Nothing reads the Enterprise repository's `main` or infers a
compatibility relationship.

**This repository.** The docs-owned pages and `assemble.yaml` are read from a
commit of `main`, never from a branch: squash merges give a branch's commits
new hashes, so only a commit of `main` stays recoverable. Preparation fetches
`main` itself, from a clone whose origin is `lancedb/docs`, because a fork's
`main` can hold commits `lancedb/docs` never has. The assembler that runs must
be that commit's.

A missing tag, absent documentation, an unqualified or inconsistently tagged
Enterprise release, or private-source files that may not be published stop
preparation with the reason. Nothing is made up from today's sources.

The bundled LanceDB is not a client SDK compatibility matrix. Client minimum
versions belong beside the examples they apply to, when evidence supports
them, and the tested client version is not a claimed minimum.

## Corrections

```mermaid
flowchart LR
    E["Error in a published edition"] --> F["Edit the saved files"]
    F --> PR["Correction PR names the editions"]
    PR --> C["Checks and review"]
    C --> S["Corrected candidate on staging"]
    S --> A["Approve production promotion"]
    F --> U["Fix the current source too<br/>if the error is still there"]
```

A correction to a saved Enterprise edition edits its folder and keeps its
version and URL. It needs no product release and no historical source branch.

Corrections are explicit. If three editions are wrong, the pull request names
and edits all three. Checking a corrected example fits the example; a typo
needs no SDK test suite. Preparing a release again never rebuilds a corrected
edition from its original sources.

## Reader behavior

Site composition will provide this:

- First-time readers land on the latest published Enterprise edition.
- Unversioned links go to the same page in that edition. Redirects come from
  the actual page map and keep valid anchors; paths are not blindly prefixed.
- Versioned links stay in their edition.
- Readers can choose the newest OSS edition independently.
- Selectors use short labels. An "About this version" page shows the full
  release identity and its sources.
- Version switching is Mintlify's own. In a hosted trial, switching from a
  page that only a newer version has landed on another page of the older
  version, without explanation. That is Mintlify's fallback, not a promise.
- Test versions and probe pages never ship as editions.

An old unversioned URL with no matching page still needs a policy. A
catch-all redirect that hides missing content is not one.

## Publication and discovery

At first the preparation bot opens pull requests and does not merge them. A
merged edition or correction goes to staging. Production needs a person's
approval of the candidate that was reviewed. A stale approval cannot promote a
different candidate, and an older queued run cannot overwrite a newer
approved publication or change the preferred edition. Promotion reuses the
staged artifact.

Missing qualification, absent sources, ambiguous inputs or failed checks stop
the workflow with a report. A duplicate event, for a release already saved
from the same commits and qualification tag, is harmless.

Rolling back changes the preferred edition and keeps published folders.
Restoring an earlier whole-site build would remove newer editions, so it needs
a deliberate decision.

Merging and production promotion can later become automatic, separately.
That changes approval policy, not the builder or the publisher.

## Open questions

These are settled where they first matter, before production:

1. Each product's qualification signal for discovery. Enterprise releases
   carry a qualification tag, which preparation already requires. For OSS,
   the stable-release completion signal, beta and release-candidate handling,
   and the release owner's approval are still to be confirmed.
2. Which Enterprise source supplies the Enterprise pages of an OSS edition.
   Preparation takes it explicitly until this is decided.
3. Missing targets for legacy URLs.
4. Approval enforcement in the repository, and the supported Mintlify staging
   setup.
5. Preferred-release order across stable and maintenance lines.

**Enterprise pages without placeholders.** The assembler currently needs an
OSS page at the same path for every Enterprise page. Removing that
requirement, with its navigation and draft handling, is a separate change that
must land before the first real editions. Editions save whatever the
assembler builds, so they need no change for it. Once it lands, prepare test
editions again and check Enterprise-only pages, intentional overlays,
navigation and hidden drafts together.
