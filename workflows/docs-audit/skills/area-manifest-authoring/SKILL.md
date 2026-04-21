---
name: area-manifest-authoring
description: Use when creating or updating a docs audit area manifest in this repo, especially for a new docs domain or when you need to discover likely source files across the watched repos and draft manifests/<area>.toml.
---

# Area Manifest Authoring

Use this skill to create or refresh `manifests/<area>.toml` for the docs-audit workspace.

This repo does not vendor the watched repos. The job is to map a docs domain to compact, user-facing evidence in the external repos.

## Read first

Before editing a manifest, read:

- `README.md`
- `AGENTS.md`
- `config.toml`
- the current manifest if one already exists at `manifests/<area>.toml`

## Goal

Produce a manifest that:

- defines the docs pages in the area
- maps each page to a small set of relevant evidence files
- prefers user-facing surfaces over deep implementation internals
- is easy to maintain when repos evolve

## Workflow

### 1. Fix the area boundary

Identify:

- the manifest filename and area name
- the docs section or path prefix in the docs repo
- the pages that should be audited together

If the boundary is fuzzy, prefer a narrower area. Smaller manifests are easier to keep accurate.

### 2. Start from the docs pages, not from code

For the target docs domain:

- list the public docs pages that belong in the area
- capture one stable `id` per page
- extract a few page-specific keywords from titles, headings, callouts, and API names

Keep page IDs short and stable. They become artifact filenames and state keys.

### 3. Discover candidate evidence across repos

Search all watched repos for user-facing signals tied to the area.

Start with targeted searches:

- docs repo: snippets, tests, nearby pages
- lancedb repo: public API files, doc comments, tests, lifecycle code
- sophon repo: request schemas, UI copy, config, integration tests

Useful commands:

```bash
rg -n "keyword1|keyword2|keyword3" <docs-repo> <lancedb-repo> <sophon-repo>
rg --files <docs-repo> <lancedb-repo> <sophon-repo> | rg "search|index|storage"
```

Resolve those repo roots from `config.toml` before you search.

When searching for future drift or newly added files:

- search by feature nouns and API verbs, not just the current filename
- search neighboring directories to the current sources
- look for tests and snippets added after the original manifest was written
- look for new UI or config surfaces that expose user-visible behavior

Do not try to include every related file. Include only the files that are likely to expose missing documentation.

### 4. Select compact evidence

Prefer files that reveal stable, user-visible behavior:

- tested snippets
- integration tests
- request and config schemas
- public API surfaces and doc comments
- UI copy and operational config

Avoid bloating the manifest with:

- large internal implementation files that expose little user-facing behavior
- duplicate files that repeat the same concepts
- broad file lists when one or two narrower files would carry the same signal

In general, a source block should stay compact. Split sources by surface type when that keeps the manifest easier to reason about.

### 5. Draft the manifest

Create or update `manifests/<area>.toml` with:

- top-level metadata:
  - `name`
  - `description`
  - `docs_repo`
  - `rotation_unit`
  - `keywords`
- one `[[pages]]` block per docs page
- one or more `[[sources]]` blocks that map evidence to the right page IDs

Use `applies_to` deliberately. That is the main control for keeping bundles relevant.

## Mechanical checks

Before finishing, verify:

- the manifest filename matches the intended CLI area
- every `repo` exists in `config.toml`
- every page `id` is unique
- every `applies_to` entry points to a real page `id` or `*`
- every `path` exists in the referenced repo
- keywords are concise and domain-specific rather than a long dump of terms

If you create a new area, also update `enabled_areas` in `config.toml` if the automation or operators expect it.

## Validation loop

After drafting the manifest, run:

```bash
uv run python scripts/run_audit.py prepare --area <area>
```

Inspect:

- `artifacts/runs/<run_id>/metadata.json`
- `artifacts/runs/<run_id>/selected_pages.json`
- `artifacts/runs/<run_id>/page_bundles/*.json`

If the bundles are noisy, tighten:

- page keywords
- source `extract_keywords`
- `applies_to`
- source file selection

If the bundles are sparse, add the smallest missing user-facing source that improves evidence quality.

## Output contract

When using this skill, the final result should:

- create or update `manifests/<area>.toml`
- mention the main assumptions made about area boundaries
- call out any likely blind spots or missing repos/files that should be reviewed later

## Request template

Use a request like:

```text
Use skills/area-manifest-authoring/SKILL.md to create or refresh manifests/<area>.toml for the <domain> docs. Scan the watched repos for likely user-facing evidence, keep the source list compact, and validate the manifest by running the prepare step.
```
