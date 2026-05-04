# Docs Audit Workspace

This workspace orchestrates a weekly documentation-gap audit across three local repositories:

- `../../../lancedb`
- `../..`
- `../../../sophon`

The goal is to find what is missing from the docs, especially conceptual and imperative guidance that exists in code, tests, UI copy, request schemas, config comments, or integration scenarios but is not conveyed clearly in the public docs.

This is a research workflow, not a production service. The design favors:

- compact deterministic preprocessing
- page-scoped LLM work inside the Codex app
- saved local artifacts for inspection and reuse
- simple extension through manifests

## Non-goals

This workspace does not:

- clone or vendor source code from the watched repos
- attempt to enforce a hard token quota in Codex
- produce doc fixes automatically
- behave like a production CI system

## Watched Repos

The watched repos live outside this workspace on purpose.

- `lancedb` is the main SDK/core codebase.
- `docs` is the Mintlify docs repo and includes tested snippets.
- `sophon` is the private product repo and contributes user-facing operational truth that may not appear in the public SDK alone.

The audit runner only reads those repos and records refresh status. This workspace stores orchestration code, manifests, prompts, state, and artifacts.

## High-Level Workflow

Each weekly run follows the same sequence:

1. Refresh watched repos with safe fast-forward pulls.
2. Read the enabled area manifests.
3. Build deterministic evidence bundles for each page in the selected area.
4. Compare current evidence fingerprints to the last completed run.
5. Select pages to audit:
   - always include pages whose mapped evidence changed
   - then include one rotating extra page for broader coverage
   - if no pages changed, the rotating extra page becomes the only selected page
   - the rotation walks through the pages in manifest order and advances one slot after each completed run
6. Use Codex LLM passes on the selected page bundles to extract:
   - code claims
   - doc claims
   - candidate gaps and final markdown observations
7. Save artifacts under a timestamped run directory.
8. Mark the run complete and update state.
9. Surface the final markdown report through a Codex inbox item.

## Workspace Layout

- `config.toml`: repo paths, enabled areas, selection rules, and output paths
- `manifests/`: docs-area manifests
- `prompts/`: reusable Codex prompt templates
- `scripts/`: deterministic extraction, refresh, selection, and state utilities
- `state/`: lightweight run state and rotation cursor
- `artifacts/`: per-run evidence bundles, LLM outputs, and reports
- `README.md`: maintainer-oriented workflow and extension guide
- `AGENTS.md`: instructions for future coding agents

## Deterministic Layer

The deterministic layer is responsible for the parts that should not require semantic interpretation:

- refreshing repos with `git pull --ff-only`
- reading manifests
- selecting source files per page
- hashing file contents and detecting changed pages
- extracting compact raw signals from docs pages and code-side surfaces
- writing page-scoped evidence bundles
- selecting changed pages plus one rotating extra page
- updating local state after a completed run

The deterministic layer intentionally keeps evidence compact so the LLM does not need to read entire files or repos.

## LLM-Assisted Layer

The semantic layer runs inside Codex through the automation prompt. For each selected page bundle, the LLM should:

1. infer normalized code claims from the evidence bundle
2. infer normalized doc claims from the docs bundle
3. identify missing documentation only
4. write concise markdown observations grouped by page or subsection

The saved artifacts should include:

- normalized code claims
- normalized doc claims
- candidate gaps
- final markdown report

## Running a Manual Audit

From this workspace root:

```bash
uv run python scripts/run_audit.py prepare --area indexing --refresh
```

`--area` is the manifest name, not a hardcoded value in the script. The runner loads:

`manifests/<area>.toml`

So `--area indexing` maps to `manifests/indexing.toml`. If you add `manifests/search.toml`, you would run:

```bash
uv run python scripts/run_audit.py prepare --area search --refresh
```

This creates a pending run directory under `artifacts/pending/<run_id>/` and prints a JSON summary to stdout.

After the LLM phase writes the expected outputs into that pending run directory, complete the run with:

```bash
uv run python scripts/run_audit.py complete --run-id <run_id>
```

Completion publishes the directory to `artifacts/runs/<run_id>/`. Directories under `artifacts/runs/`
are completed audit artifacts and should contain `report.md`.

To clean up old generated run artifacts, use:

```bash
uv run python scripts/run_audit.py cleanup --days 30
```

The retention window is configurable with `--days`, and you can preview deletions without removing anything:

```bash
uv run python scripts/run_audit.py cleanup --days 14 --dry-run
```

For manual testing of the fallback path, you can simulate an unrefreshable repo:

```bash
uv run python scripts/run_audit.py prepare \
  --area indexing \
  --refresh \
  --simulate-refresh-failure docs
```

## Inspecting Artifacts

Each completed run directory under `artifacts/runs/<run_id>/` contains:

- `metadata.json`: run-level metadata, repo refresh results, selection decisions
- `page_bundles/*.json`: deterministic evidence bundles per page
- `selected_pages.json`: the pages chosen for the semantic pass
- `llm_outputs/`: normalized claims, candidate gaps, and other semantic outputs
- `report.md`: final human-readable report

`artifacts/latest_run.json` points to the most recently completed run.

Pending run directories under `artifacts/pending/<run_id>/` are working directories from `prepare`.
They are used for manifest validation and LLM drafting, and are not considered completed artifacts
until `complete` publishes them.

## Using and Updating Area Manifests

The manifest is the only thing you usually need to change when you want to audit another docs domain. Treat it as a mapping file:

- which docs pages belong to this area
- which keywords help extract useful evidence
- which code, tests, snippets, or product surfaces should be compared against those docs pages

You do not need to understand every line in `manifests/indexing.toml` to create another area. Copy it, rename it for the new domain, then update only these parts.

If you want an LLM to help draft or refresh a manifest, use the repo-local skill at [skills/area-manifest-authoring/SKILL.md](skills/area-manifest-authoring/SKILL.md). It tells the agent how to discover candidate files across the watched repos, keep the evidence compact, and validate the generated manifest with `prepare`.

### 1. Pick the manifest filename and area name

The filename is the CLI value for `--area`.

- `manifests/indexing.toml` -> `--area indexing`
- `manifests/search.toml` -> `--area search`
- `manifests/storage.toml` -> `--area storage`

Set the top-level fields first:

```toml
name = "search"
description = "Audit the search docs against user-facing evidence from lancedb, docs snippets/tests, and sophon."
docs_repo = "docs"
rotation_unit = "page"
keywords = ["search", "filter", "rerank"]
```

What these fields mean:

- `name`: logical name for the area; keep it aligned with the filename
- `description`: short human-readable summary for metadata and operators
- `docs_repo`: usually `docs`; this is where the docs pages live
- `rotation_unit`: currently `page`; leave this alone unless the runner changes
- `keywords`: broad terms used across the whole area to find evidence

### 2. Define the docs pages in scope

Each `[[pages]]` block describes one public docs page you want audited.

```toml
[[pages]]
id = "overview"
title = "Search Overview"
path = "docs/search/index.mdx"
keywords = ["search", "full text search", "hybrid"]
```

Guidance:

- `id`: short stable identifier used in artifacts and source mappings
- `title`: readable label for reports
- `path`: path inside the docs repo
- `keywords`: page-specific terms that sharpen extraction for this page

Keep page IDs short and stable. If you rename them later, the state history for that page will no longer line up cleanly.

### 3. Map code-side evidence to the right pages

Each `[[sources]]` block says where supporting evidence should come from.

```toml
[[sources]]
id = "docs-snippets-tests"
repo = "docs"
kind = "snippets_and_tests"
applies_to = ["overview", "hybrid-search"]
paths = [
  "docs/snippets/search.mdx",
  "tests/py/test_search.py",
]
extract_keywords = ["search", "filter", "rerank"]
```

What to update:

- `id`: stable source label used in page bundles
- `repo`: one of the repos defined in `config.toml`
- `kind`: short category label for operators and prompts
- `applies_to`: page IDs that should receive this evidence
- `paths`: files inside that repo to scan
- `extract_keywords`: extra terms for this source only

The important field here is `applies_to`. It is how you decide which evidence should inform which docs page.

### 4. Prefer compact, user-facing evidence

Good source files:

- tested snippets
- request or config schemas
- doc comments on public APIs
- integration tests
- UI copy and user-facing config

Less useful source files:

- large internal implementation files with no public behavior signals
- broad directories when one or two targeted files would do
- duplicated files that say the same thing

The goal is not to prove how the system works internally. The goal is to capture user-visible behavior the docs may be missing.

### 5. Create a new area by copying, then trimming

A practical workflow:

1. Copy `manifests/indexing.toml` to `manifests/<new-area>.toml`.
2. Rename `name`, `description`, and the page paths.
3. Delete pages that do not belong to the new domain.
4. Replace the keywords with terms that actually appear in that domain.
5. Replace each source block with a small set of relevant files.
6. Make sure every `applies_to` entry refers to a real page `id`.
7. Add the area to `enabled_areas` in `config.toml` if your automation depends on that list.
8. Run `prepare --area <new-area>` and inspect the generated `page_bundles/*.json` in the printed pending `run_dir`.

### 6. Sanity-check the manifest before using it weekly

After adding a new manifest, run:

```bash
uv run python scripts/run_audit.py prepare --area <new-area>
```

Then inspect:

- the `run_dir` printed by `prepare` (normally `artifacts/pending/<run_id>`)
- `<run_dir>/metadata.json`
- `<run_dir>/selected_pages.json`
- `<run_dir>/page_bundles/*.json`

If the bundles look noisy, the fix is usually one of:

- fewer source files
- better page keywords
- better `extract_keywords`
- tighter `applies_to` mappings

The runner is designed so new docs areas should generally require a new manifest, not new orchestration code.

## Weekly Automation

The weekly Codex automation should use this workspace as its cwd and follow `prompts/weekly_automation.md`.

The automation should:

- review each enabled area manifest before running the audit
- use `skills/area-manifest-authoring/SKILL.md` to detect docs-page drift and newly relevant evidence files in the watched repos
- update a manifest when the area boundary or source mapping has materially changed
- run the deterministic prepare step
- inspect the generated selected page bundles
- perform the page-scoped LLM passes
- write outputs under the run directory
- keep `report.md` limited to the missing-doc summary itself, not routine workflow or refresh-status narration
- call the completion step
- return a concise markdown summary for the inbox item

## Maintainer Notes

- Keep reports focused on what is missing in the docs, not on implementation summaries or fix proposals.
- Do not spend report tokens on routine success status such as clean repo refreshes.
- Prefer evidence from doc comments, tested snippets, request schemas, UI copy, config comments, and integration tests over deep implementation internals.
- If a new feature lands and the docs area should notice it, update the manifest first.
- If the semantic pass grows too expensive, reduce weekly selection breadth before shrinking evidence quality.
