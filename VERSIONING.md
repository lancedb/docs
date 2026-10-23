# Versioned documentation

Readers should be able to find the docs for the version of LanceDB they run.
The docs are still written next to the code, in `lancedb/lancedb` and the
Enterprise documentation repository. Each release gets a saved edition that
combines its product pages, the selected shared or Enterprise pages, and pages
owned by this repository. These come from separately recorded commits. Readers
can switch between editions on the site.

Release records, saved editions, site composition, publishing and release
discovery are in place; see
[Discovering releases](README.md#discovering-releases).
Until a cutover is approved, the site is built and published as it is today.

## How it works

Each edition is the site the assembler built from those sources, stored in
`editions/oss/<version>/` or
`editions/enterprise/<version>/`. A record in `releases/` lists the commits it
came from. An edition is built once. Later changes to `main` or to a product's
docs don't reach it, so older versions keep the pages they shipped with.

The published site is put together from the saved editions. Each edition gets
its own path, such as `/enterprise/v1-10-0/`, and readers switch products and
versions with Mintlify's built-in selectors. Each version also has an About
this version page listing its release and sources. Branding, the banner, the
footer and integrations are shared across versions.

A composed site goes to staging first, and production gets the same build after
someone approves it.

New editions arrive as pull requests. They can be prepared by hand with
`scripts/edition.py prepare`, and a discovery job prepares one for each new
qualified release. People review and merge every one.

## What goes into an edition

An Enterprise edition has the Enterprise release's own pages, plus the shared
pages from the exact LanceDB commit that release bundles. That commit can be
older than the latest OSS release, or one that no OSS release tags.

An OSS edition has the pages from its `lancedb/lancedb` release tag. The
navigation also covers the Enterprise pages, so an OSS edition takes them from
an Enterprise release that's named when the edition is prepared. Discovery uses
the preferred Enterprise edition.

Pages owned by this repository, like the Geneva pages, dataset cards and REST
API reference, come from `main` as it was when the edition was prepared.

Every edition keeps the full navigation, with badges showing what's available
in OSS and in Enterprise. OSS and Enterprise editions ship on their own
schedules, and an Enterprise release doesn't need a matching OSS release.

Only stable releases get editions. An Enterprise release counts once its
release tag and its `release-qualified/` tag point at the same commit. An OSS
release counts once its GitHub release is published and isn't marked as a
prerelease. Betas and release candidates are skipped, and older releases aren't
backfilled.

## Which version readers see

Each product has a preferred edition, which readers get by default. New
visitors land on the preferred Enterprise edition, and links without a version
go to the same page there. Links that include a version stay in that version.

The preferred edition is labelled Latest. Discovery proposes a newer version
as the preferred edition in the same pull request that adds its docs. It takes
effect when that pull request merges. A patch to an older release line leaves
a newer preferred version alone.

If an old link has no matching page in the default edition, it gets no
redirect. Composition lists those links so someone can add the page or a
specific redirect.

When a reader switches to a version that doesn't have the page they're on,
Mintlify decides where they land.

## Corrections

A saved edition can be corrected through a reviewed pull request that edits its
files and records the correction. The version and its URLs stay the same. If
the mistake is still in the current docs, fix it there too. Preparing the same
release again keeps the correction.

## Publishing

Merged editions and corrections go to staging on their own. Production gets the
build staging shows, after approval in the `production` GitHub environment, and
promotion publishes that build without rebuilding it. An older run can't
replace a newer approved one.

To roll back, change the preferred edition. Published versions stay where they
are.

Merging and production promotion could become automatic later, each as its own
decision.
