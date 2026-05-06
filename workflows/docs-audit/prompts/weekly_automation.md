# Weekly Docs Audit Automation Prompt

You are running the weekly docs-gap audit from this workspace root.

## Objective

Produce a concise markdown report that lists only what is missing from the docs for the selected pages. Focus on conceptual and imperative gaps, not implementation summaries or fix proposals.

This workflow also includes manifest maintenance. Before each audit run, review the enabled area manifests to see whether the docs pages or evidence sources have drifted and update them when needed.

## Files to read first

- `README.md`
- `AGENTS.md`
- `config.toml`
- `prompts/page_audit_guidelines.md`
- `skills/area-manifest-authoring/SKILL.md`

Then select the area manifests for this run using the deterministic area selector.

## Required workflow

1. Read `config.toml` and determine the enabled areas from `enabled_areas`.
2. Select the areas for this weekly run:
   - `uv run python scripts/run_audit.py select-areas --refresh --advance`
   - Use the printed `selected_areas` list for the rest of this workflow.
   - The selector refreshes watched repos once, detects changed enabled areas, and fills the remaining weekly slots by area rotation.
   - Do not run unselected enabled areas in this weekly pass.
3. For each selected area, run a manifest maintenance pass before `prepare`.
   - Use `skills/area-manifest-authoring/SKILL.md` as the procedure.
   - Read the current `manifests/<area>.toml`.
   - Check whether the docs area boundary has changed:
     - new or renamed docs pages in the same docs section
     - stale page paths
     - page IDs that no longer match the current docs layout
   - Check whether the evidence mapping has drifted:
     - new snippets, tests, request schemas, config files, UI surfaces, or public API files related to the area
     - stale source paths that should be removed
     - source blocks whose `applies_to` mapping is now too broad or too narrow
   - Keep the manifest compact. Do not add files just because they mention the topic; add them only if they are likely to expose user-visible behavior the docs may be missing.
   - If the manifest changes, save the updated `manifests/<area>.toml` before preparing the run.
4. Run the deterministic prepare step for each selected area.
   - Repos were already refreshed by `select-areas`, so skip `--refresh` here:
     - `uv run python scripts/run_audit.py prepare --area <area>`
5. Read the JSON summary printed by each `prepare` command and locate each pending run directory.
   - Use the printed `run_dir`; it should point under `artifacts/pending/<run_id>`.
   - Do not create or write directly under `artifacts/runs/<run_id>` before completion.
6. For each pending run directory, read `selected_pages.json` and the corresponding files in `page_bundles/`.
7. For each selected page bundle:
   - apply `prompts/page_audit_guidelines.md` as the page-level review rubric
   - infer normalized code claims from the evidence bundle
   - infer normalized doc claims from the docs bundle
   - identify only the missing documentation
8. Write semantic outputs under `llm_outputs/` in each pending run directory.
   - one file per page for code claims
   - one file per page for doc claims
   - one file per page for candidate gaps
9. Write `report.md` in each pending run directory.
   - `report.md` is the docs-gap summary only.
   - Do not include refresh status, manifest-maintenance notes, selected-pages bookkeeping, or any other workflow narration in `report.md`.
   - Include operational notes only if they materially affected audit quality, such as an unrefreshable repo, missing source files, or a manifest ambiguity that changes confidence in the findings.
10. Complete each run:
   - `uv run python scripts/run_audit.py complete --run-id <run_id>`
   - Completion publishes the pending directory to `artifacts/runs/<run_id>` and updates `artifacts/latest_run.json`.
   - Only completed runs with `report.md` should appear under `artifacts/runs/`.
11. Return a concise markdown summary suitable for the Codex inbox item.

## Manifest maintenance rules

- Prefer updating the manifest when the docs area or user-facing evidence has clearly evolved.
- Prefer stability over churn. Do not rewrite a manifest just to reorganize it.
- Prefer compact source lists over exhaustive source lists.
- Prefer user-facing evidence over internal implementation detail.
- If you find a likely new source file but its relevance is ambiguous, mention it in the final summary as a follow-up risk instead of forcing it into the manifest.

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
