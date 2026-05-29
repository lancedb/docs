# Docs Audit Workspace

This workspace orchestrates a weekly documentation-gap audit across three local repositories:

- `../../../lancedb`
- `../..`
- `../../../sophon`

The goal is to find what is missing from the docs, especially conceptual and imperative guidance that exists in code, tests, UI copy, request schemas, config comments, or integration scenarios but is not conveyed clearly in the public docs.

This is a research workflow with a scheduled cloud runner. The design favors:

- compact deterministic preprocessing
- page-scoped LLM work through the OpenAI API
- saved local artifacts for inspection and reuse
- durable storage of completed reports and parsed findings
- simple extension through manifests

## Non-goals

This workspace does not:

- clone or vendor source code from the watched repos
- attempt to enforce a hard token quota in the agent runtime
- produce doc fixes automatically
- automatically author or rewrite area manifests during scheduled runs

## Watched Repos

The watched repos live outside this workspace on purpose.

- `lancedb` is the main SDK/core codebase.
- `docs` is the Mintlify docs repo and includes tested snippets.
- `sophon` is the private product repo and contributes user-facing operational truth that may not appear in the public SDK alone.

The audit runner only reads those repos and records refresh status. This workspace stores orchestration code, manifests, prompts, state, and artifacts.

## High-Level Workflow

Each weekly run follows the same sequence:

1. Refresh watched repos with safe fast-forward pulls.
2. Select a bounded set of enabled area manifests for the weekly run.
3. Read the selected area manifests.
4. Build deterministic evidence bundles for each page in the selected area.
5. Compare current evidence fingerprints to the last completed run.
6. Select pages to audit:
   - always include pages whose mapped evidence changed
   - then include rotating extra pages for broader coverage
   - if no pages changed, the rotating extra pages become the selected pages
   - the rotation walks through the pages in manifest order and advances as rotating pages are added
7. Use OpenAI API-driven page-scoped LLM passes on the selected page bundles to extract:
   - code claims
   - doc claims
   - candidate gaps and final markdown observations
8. Save artifacts under a timestamped run directory.
9. Mark the run complete and update state.
10. Parse the completed `report.md` into durable findings.
11. Embed each public finding with OpenAI embeddings.
12. Store the run and findings in LanceDB Enterprise under `db://docs-audit`.
13. Surface a concise summary from the filtered public findings.

## Workspace Layout

- `config.toml`: repo paths, enabled areas, selection rules, and output paths
- `manifests/`: docs-area manifests
- `prompts/`: reusable agent prompt templates
- `docs_audit/`: deterministic runner, OpenAI helpers, report parser, and Enterprise storage code
- `scripts/run_weekly_audit.py`: user-facing weekly audit entrypoint for local and EC2 cron runs
- `state/`: lightweight run state and rotation cursor
- `artifacts/`: per-run evidence bundles, LLM outputs, and reports
- `README.md`: maintainer-oriented workflow and extension guide
- `AGENTS.md`: instructions for future coding agents

## Deterministic Layer

The deterministic layer is responsible for the parts that should not require semantic interpretation:

- refreshing repos with `git pull --ff-only`
- reading manifests
- selecting changed enabled areas first, then filling the weekly area budget by rotation
- selecting source files per page
- hashing file contents and detecting changed pages
- extracting compact raw signals from docs pages and code-side surfaces
- writing page-scoped evidence bundles
- selecting changed pages plus rotating extra pages
- updating local state after a completed run

The deterministic layer intentionally keeps evidence compact so the LLM does not need to read entire files or repos.

## LLM-Assisted Layer

The semantic layer runs through the OpenAI API. For each selected page bundle, the LLM should:

1. infer normalized code claims from the evidence bundle
2. infer normalized doc claims from the docs bundle
3. identify missing documentation only
4. write concise markdown observations grouped by page or subsection

The saved artifacts should include:

- normalized code claims
- normalized doc claims
- candidate gaps
- final markdown report

The scheduled cloud workflow should use existing manifest files only. Manifest authoring and manifest
maintenance are manual maintainer activities; they may use `skills/area-manifest-authoring/SKILL.md`,
but the weekly cloud run should not edit manifests as part of normal execution.

## Running a Manual Audit

From this workspace root:

```bash
uv run python -m docs_audit.deterministic_runner select-areas --refresh --advance
```

This chooses a bounded list of enabled area manifests for the weekly run. The selector uses
`[area_selection]` in `config.toml`: changed enabled areas are considered first, then any remaining
weekly slots are filled by rotating through `enabled_areas`. Use the printed `selected_areas` list
for the per-area `prepare` commands.

```bash
uv run python -m docs_audit.deterministic_runner prepare --area indexing
```

`--area` is the manifest name, not a hardcoded value in the script. The runner loads:

`manifests/<area>.toml`

So `--area indexing` maps to `manifests/indexing.toml`. If you add `manifests/search.toml`, you would run:

```bash
uv run python -m docs_audit.deterministic_runner prepare --area search
```

This creates a pending run directory under `artifacts/pending/<run_id>/` and prints a JSON summary to stdout.

When running after `select-areas --refresh`, omit `--refresh` from `prepare`; the repos were already
refreshed once for the weekly selection.
For a standalone one-area audit where you skip `select-areas`, pass `--refresh` to `prepare`.

## Area Selection

`enabled_areas` is the full pool of manifests the weekly automation may audit. The `[area_selection]`
block controls how many of those enabled manifests are selected for a single weekly run:

```toml
[area_selection]
mode = "changed_first_rotate"
areas_per_run = 2
```

Supported modes:

- `all`: select every enabled area.
- `rotate`: ignore changed-area detection and select only by rotating through `enabled_areas`.
- `changed_first_rotate`: select changed enabled areas first, up to `areas_per_run`, then fill any remaining slots by rotation.

The area rotation cursor is stored in `state/state.json` under `area_selection.rotation_index` when
you run `select-areas --advance`. Page-level rotation still happens independently inside each
selected area through `[selection].rotation_extra_pages`.

After the LLM phase writes the expected outputs into that pending run directory, complete the run with:

```bash
uv run python -m docs_audit.deterministic_runner complete --run-id <run_id>
```

Completion publishes the directory to `artifacts/runs/<run_id>/`. Directories under `artifacts/runs/`
are completed audit artifacts and should contain `report.md`.

Run multiple area `prepare` commands sequentially. The intended workflow is one prepared area at a time.

To clean up old generated run artifacts, use:

```bash
uv run python -m docs_audit.deterministic_runner cleanup --days 30
```

The retention window is configurable with `--days`, and you can preview deletions without removing anything:

```bash
uv run python -m docs_audit.deterministic_runner cleanup --days 14 --dry-run
```

For manual testing of the fallback path, you can simulate an unrefreshable repo:

```bash
uv run python -m docs_audit.deterministic_runner prepare \
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
uv run python -m docs_audit.deterministic_runner prepare --area <new-area>
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

The weekly automation should use this workspace as its cwd and follow the deterministic selection and
prepare/complete flow described above. In the cloud, the semantic pass is performed with OpenAI API
credentials rather than a Codex Desktop agent.

The automation should:

- load the enabled area manifests as read-only workflow inputs
- run `select-areas --refresh --advance`
- run `prepare` sequentially for each selected area
- inspect the generated selected page bundles
- perform the page-scoped LLM passes through the OpenAI API
- write outputs under the run directory
- keep `report.md` limited to the missing-doc summary itself, not routine workflow or refresh-status narration
- call the completion step
- parse the completed `report.md` into finding records
- filter out findings that should not be exposed to end users, including helm chart and enterprise deployment observations
- write the completed run and public findings to LanceDB Enterprise
- return a concise markdown summary from the stored public findings

The cloud runner needs these secrets or environment variables:

- `OPENAI_API_KEY`: used for page-level semantic passes and finding embeddings
- `DOCS_AUDIT_OPENAI_MODEL`: chat/reasoning model for page-level audits; defaults to `gpt-5.5`
- `DOCS_AUDIT_OPENAI_REASONING_EFFORT`: reasoning effort for page-level audits; defaults to `high`
- `DOCS_AUDIT_EMBEDDING_MODEL`: embedding model for finding search vectors
- `LANCEDB_API_KEY`: LanceDB Enterprise API key
- `LANCEDB_HOST_OVERRIDE`: LanceDB Enterprise host URL
- `LANCEDB_REGION`: LanceDB Enterprise region, usually `us-east-1`
- `DOCS_AUDIT_DB_URI`: optional override for the Enterprise database URI; defaults to `db://docs-audit`

Use GPT-5.5 with high reasoning for the page-audit semantic pass. The audit is intentionally
judgment-heavy: the model has to compare compact evidence bundles against docs claims, avoid
implementation summaries, and emit only missing public documentation observations. Embeddings are a
separate step used only after `report.md` is complete and parsed.

The LanceDB Enterprise connection should follow the same remote-only pattern used by neighboring
internal tooling:

```python
lancedb.connect(
    uri="db://docs-audit",
    api_key=LANCEDB_API_KEY,
    host_override=LANCEDB_HOST_OVERRIDE,
    region=LANCEDB_REGION,
)
```

## EC2 Cron Deployment

An EC2 cron job is a suitable deployment target for this workflow. The instance should keep the
watched repositories checked out side by side so the relative paths in `config.toml` continue to
resolve:

```text
/opt/lancedb-docs-audit/
  lancedb/
  docs/
    workflows/docs-audit/
  sophon/
```

From the `docs` checkout, the docs-audit workspace still expects:

- `../../../lancedb`
- `../..`
- `../../../sophon`

If the EC2 checkout layout differs, update `workflows/docs-audit/config.toml` instead of adding
path translation logic to the runner.

Create a local environment file at `workflows/docs-audit/.env` on the instance. `.env` files are
ignored by this repo and must not be committed:

```bash
OPENAI_API_KEY=...
DOCS_AUDIT_OPENAI_MODEL=gpt-5.5
DOCS_AUDIT_OPENAI_REASONING_EFFORT=high
DOCS_AUDIT_EMBEDDING_MODEL=text-embedding-3-large

LANCEDB_API_KEY=...
LANCEDB_HOST_OVERRIDE=https://...
LANCEDB_REGION=us-east-1
DOCS_AUDIT_DB_URI=db://docs-audit
```

Cron should call a single cloud-runner entrypoint from the docs-audit workspace. Use a lock so a slow
run cannot overlap the next scheduled run, and write logs outside the repo:

```cron
17 13 * * 1 cd /opt/lancedb-docs-audit/docs/workflows/docs-audit && flock -n /tmp/docs-audit.lock uv run python scripts/run_weekly_audit.py >> /var/log/docs-audit/weekly.log 2>&1
```

The cloud runner should:

- load `workflows/docs-audit/.env` before reading configuration
- use GPT-5.5 with high reasoning for page-level audit calls
- use OpenAI embeddings only after `report.md` has been generated and parsed
- write completed runs and findings to `db://docs-audit`
- exit non-zero when refresh, OpenAI, parsing, or Enterprise writes fail

The EC2 instance needs outbound network access to Git remotes, the OpenAI API, and the LanceDB
Enterprise host. Prefer instance IAM or deploy keys for repository access, and keep OpenAI and
LanceDB credentials in the local `.env` or the instance's secret-management layer.

## Testing the Cloud Runner Locally

Copy `workflows/docs-audit/.env.example` to `workflows/docs-audit/.env` and fill in the secrets.

The runner has four practical modes:

| Command | Selects areas | Calls GPT-5.5 | Calls embeddings | Writes LanceDB | Use for |
| --- | --- | --- | --- | --- | --- |
| `run_weekly_audit.py --ingest-run-dir artifacts/runs/<run_id> --skip-write` | No | No | No | No | Cheapest parser/report smoke test |
| `run_weekly_audit.py --no-refresh --no-advance --skip-write` | Yes | Yes | No | No | Local report-generation test |
| `run_weekly_audit.py --ingest-run-dir artifacts/runs/<run_id>` | No | No | Yes | Yes | Backfill/test Enterprise writes for one completed run |
| `run_weekly_audit.py` | Yes | Yes | Yes | Yes | Real weekly EC2 cron run |

Argument meanings:

- `--ingest-run-dir`: bypasses weekly selection and report generation; parses an existing completed run.
- `--skip-write`: skips finding embeddings and LanceDB Enterprise writes. It does not skip GPT-5.5 if the command is generating a new report.
- `--no-refresh`: skips `git pull --ff-only` during weekly area selection.
- `--no-advance`: prevents the area rotation cursor from moving, which makes local tests repeatable.

To run the cloud workflow locally without refreshing repos or writing to LanceDB Enterprise:

```bash
cd workflows/docs-audit
uv run python scripts/run_weekly_audit.py --no-refresh --no-advance --skip-write
```

This still calls the OpenAI page-audit model and writes a completed local run artifact, but it skips
finding embeddings and Enterprise writes.

To backfill or test parsing for an existing completed run without OpenAI or Enterprise calls:

```bash
cd workflows/docs-audit
uv run python scripts/run_weekly_audit.py \
  --ingest-run-dir artifacts/runs/<run_id> \
  --skip-write
```

To test the Enterprise write path for an existing completed run, omit `--skip-write`. That path
uses OpenAI embeddings and writes to `docs_audit_runs` and `docs_audit_findings`.

## Enterprise Storage

The durable audit output is the completed `report.md`. Intermediate files under `llm_outputs/` and
`page_bundles/` are useful for inspection, but they are not the primary historical record.

Store completed report data in two LanceDB Enterprise tables under `db://docs-audit`:

### `docs_audit_runs`

One row per completed run.

| Column | Purpose |
| --- | --- |
| `run_id` | Primary run identifier |
| `completed_at` | Run completion timestamp |
| `areas` | Selected areas for the run |
| `report_text` | Raw completed `report.md` text, stored once per run |
| `report_path` | Original artifact path or cloud artifact URL |
| `repo_shas` | Docs, LanceDB, and Sophon commit SHAs |
| `selected_pages` | Pages audited in the run |
| `changed_pages` | Pages whose evidence fingerprints changed |
| `refresh` | Watched repo refresh metadata |
| `metadata` | Extra run metadata |

### `docs_audit_findings`

One row per parsed missing-doc observation from the completed report.

| Column | Purpose |
| --- | --- |
| `id` | Stable finding id, such as `{run_id}:{finding_index}` |
| `run_id` | Link back to `docs_audit_runs` |
| `completed_at` | Copied from the run for time filtering |
| `area` | Docs audit area |
| `page_id` | Manifest page id when it can be resolved |
| `page_title` | Human-readable page or report heading |
| `page_path` | Docs source path when it can be resolved |
| `report_heading` | Heading from `report.md` |
| `finding_index` | Finding order in the report |
| `finding_text` | Parsed missing-doc observation |
| `finding_hash` | Hash of normalized area, page, and finding text |
| `visibility_class` | `public-doc-gap` or `excluded` |
| `embedding_text` | Compact text sent to the OpenAI embeddings API |
| `embedding` | Vector used for later semantic search |
| `metadata` | Small finding-level extras |

For reruns or backfills of the same `run_id`, delete existing rows for that `run_id` from both tables
and replace them. Do not rewrite historical rows during routine weekly runs.

## Maintainer Notes

- Keep reports focused on what is missing in the docs, not on implementation summaries or fix proposals.
- Do not spend report tokens on routine success status such as clean repo refreshes.
- Prefer evidence from doc comments, tested snippets, request schemas, UI copy, config comments, and integration tests over deep implementation internals.
- If a new feature lands and the docs area should notice it, update the manifest manually first.
- The scheduled cloud workflow should not use `skills/area-manifest-authoring/SKILL.md`; maintainers can use that skill manually when adding or refreshing manifests.
- If the semantic pass grows too expensive, reduce weekly selection breadth before shrinking evidence quality.
