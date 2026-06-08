---
name: docs-audit-enterprise-summary
description: Query and summarize LanceDB docs-audit findings stored in the Enterprise `db://docs-audit` database. Use when the user asks for latest docs gaps, weekly audit gaps, remote audit findings, findings since a date, or summaries from the LanceDB docs audit DB and points to `workflows/docs-audit`, `docs_audit`, or its `.env` credentials.
---

# Docs Audit Enterprise Summary

## Workflow

Use this skill to inspect completed LanceDB docs-gap audit findings from the Enterprise audit store without exposing credentials.

1. Locate the audit workspace. Prefer a user-provided path; otherwise check the current repo for `workflows/docs-audit`, then `/Users/prrao/code/docs/workflows/docs-audit`.
2. Load credentials through the audit workspace code. Do not print `.env` values. It should provide `DOCS_AUDIT_DB_URI`, `LANCEDB_API_KEY`, `LANCEDB_HOST_OVERRIDE`, and `LANCEDB_REGION`.
3. Query `docs_audit_runs` and `docs_audit_findings` in LanceDB Enterprise.
4. Filter findings to `visibility_class == "public-doc-gap"` and the requested UTC date window.
5. Summarize by run, area, page, and recurring theme. Keep the final answer focused on actionable documentation gaps, not raw rows.

## Helper Script

Run the bundled script with the docs repo `uv` environment so the audit dependencies are available:

```bash
cd /Users/prrao/code/docs
uv run python workflows/docs-audit/skills/docs-audit-enterprise-summary/scripts/query_docs_audit.py \
  --audit-root /Users/prrao/code/docs/workflows/docs-audit \
  --start 2026-05-18 \
  --end 2026-05-25 \
  --format digest-markdown
```

Date-only `--end` is treated as inclusive, so `--end 2026-05-25` covers through `2026-05-25T23:59:59Z`. Use an explicit timestamp ending in `Z` when the user asks for an exact exclusive instant.

The script uses `RemoteTable.head(n)` because `RemoteTable.to_arrow()` is not supported on this LanceDB Cloud connection. It drops `embedding`, `embedding_text`, and large run report fields before converting rows. If plain `python3` cannot import `lancedb`, rerun through `uv run python` from `/Users/prrao/code/docs`.

Prefer `--format digest-markdown` or `--format digest-json` for normal answers. These formats keep the payload light by returning counts, affected pages, theme buckets, and a small number of representative examples. Use `--format json` only when the user asks for raw finding records.

## Summarizing Results

In the final response, include:

- the UTC date window used
- total public findings and runs included
- counts by area when useful
- the key gap themes, grouped by docs area or workflow
- any caveat if the DB query was limited by `--limit`

Do not include API keys, host overrides, raw credentials, embeddings, or long raw finding dumps unless the user explicitly asks for a queue.

## Common Paths

- Audit workspace: `/Users/prrao/code/docs/workflows/docs-audit`
- Audit package: `/Users/prrao/code/docs/workflows/docs-audit/docs_audit`
- Repo-local skill: `/Users/prrao/code/docs/workflows/docs-audit/skills/docs-audit-enterprise-summary`
- Local pulled queue, when present: `/Users/prrao/code/docs/workflows/docs-audit/artifacts/fix-queues/remote-findings-since-2026-05-18.md`
