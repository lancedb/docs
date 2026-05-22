# Weekly Docs Audit Automation Prompt

You are running the weekly docs-gap audit from this workspace root.

## Objective

Produce a concise markdown report that lists only what is missing from the docs for the selected pages. Focus on conceptual and imperative gaps, not implementation summaries or fix proposals.

The scheduled workflow uses existing enabled area manifests as read-only inputs. Manifest authoring
and manifest maintenance are manual maintainer activities.

## Files to read first

- `README.md`
- `AGENTS.md`
- `config.toml`
- `prompts/page_audit_guidelines.md`

Then select the area manifests for this run using the deterministic area selector.

## Required workflow

1. Read `config.toml` and determine the enabled areas from `enabled_areas`.
2. Select the areas for this weekly run:
   - `uv run python -m docs_audit.deterministic_runner select-areas --refresh --advance`
   - Use the printed `selected_areas` list for the rest of this workflow.
   - The selector refreshes watched repos once, detects changed enabled areas, and fills the remaining weekly slots by area rotation.
   - Do not run unselected enabled areas in this weekly pass.
3. Run the deterministic prepare step for each selected area.
   - Run prepare commands sequentially, one area at a time. Do not parallelize `prepare`.
   - Repos were already refreshed by `select-areas`, so skip `--refresh` here:
     - `uv run python -m docs_audit.deterministic_runner prepare --area <area>`
4. Read the JSON summary printed by each `prepare` command and locate each pending run directory.
   - Use the printed `run_dir`; it should point under `artifacts/pending/<run_id>`.
   - Do not create or write directly under `artifacts/runs/<run_id>` before completion.
5. For each pending run directory, read `selected_pages.json` and the corresponding files in `page_bundles/`.
6. For each selected page bundle:
   - apply `prompts/page_audit_guidelines.md` as the page-level review rubric
   - infer normalized code claims from the evidence bundle
   - infer normalized doc claims from the docs bundle
   - identify only the missing documentation
7. Write semantic outputs under `llm_outputs/` in each pending run directory.
   - one file per page for code claims
   - one file per page for doc claims
   - one file per page for candidate gaps
8. Write `report.md` in each pending run directory.
   - `report.md` is the docs-gap summary only.
   - Do not include refresh status, manifest-maintenance notes, selected-pages bookkeeping, or any other workflow narration in `report.md`.
   - Include operational notes only if they materially affected audit quality, such as an unrefreshable repo, missing source files, or a manifest ambiguity that changes confidence in the findings.
   - Do not include helm chart or enterprise deployment findings.
9. Complete each run:
   - `uv run python -m docs_audit.deterministic_runner complete --run-id <run_id>`
   - Completion publishes the pending directory to `artifacts/runs/<run_id>` and updates `artifacts/latest_run.json`.
   - Only completed runs with `report.md` should appear under `artifacts/runs/`.
10. Parse the completed `report.md` into findings, generate embeddings, and write the completed run
    plus public findings to LanceDB Enterprise.
11. Return a concise markdown summary suitable for the inbox item.

## Report rules

- Describe only what is missing from the docs.
- Group by page or subsection where helpful.
- Keep the report compact and high signal.
- Do not propose code changes.
- Do not rewrite docs.
- Do not restate successful refreshes, routine maintenance steps, or other low-information operational status.
- If there were no material gaps for a selected page, omit that page instead of adding filler text.
- Prefer one short heading plus flat bullets of missing-doc observations.

## Inbox summary rules

- Keep the inbox summary short.
- Include operational details only when they require attention or explain reduced confidence.
- Do not echo routine success states such as all repos refreshing cleanly.
