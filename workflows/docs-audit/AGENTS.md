# AGENTS.md

This workspace orchestrates a docs-gap audit across external local repos. It does not own or vendor the source code from those repos.

## Working model

- Deterministic scripts live in `scripts/`.
- Area manifests live in `manifests/`.
- Codex prompt templates live in `prompts/`.
- Run state lives in `state/`.
- Generated run artifacts live in `artifacts/`.

## Rules for future agents

- Do not copy large code snapshots from the watched repos into this workspace.
- Keep the deterministic layer deterministic: refresh, extract, fingerprint, select, and update state.
- Keep semantic reasoning page-scoped and artifact-backed.
- Reports must describe only what is missing from the docs.
- When adding a new docs area, prefer a new manifest over changes to the core runner.
- Keep evidence compact and user-facing where possible.
- Preserve the distinction between deterministic outputs (`page_bundles`, metadata, state) and LLM outputs (`llm_outputs`, final report).

## Expected output shape

A completed run should leave behind:

- `metadata.json`
- `selected_pages.json`
- `page_bundles/*.json`
- `llm_outputs/*`
- `report.md`

The report should be concise, grouped by page or subsection, and should not contain implementation plans or doc-fix patches.
