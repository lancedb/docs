#!/usr/bin/env python3
"""Retrieve recent docs-audit gaps from the Enterprise LanceDB store.

Returns only the id, date, and description for each gap so a downstream agent
can summarize without ingesting embeddings or other heavy fields.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DROP_COLUMNS = {
    "embedding",
    "embedding_text",
    "metadata",
    "report_text",
    "refresh",
    "repo_shas",
}


def locate_audit_root(user_path: str | None) -> Path:
    candidates: list[Path] = []
    if user_path:
        candidates.append(Path(user_path).expanduser())
    cwd = Path.cwd()
    candidates.extend(
        [
            cwd,
            cwd / "workflows" / "docs-audit",
            Path("/Users/prrao/code/docs/workflows/docs-audit"),
        ]
    )
    for candidate in candidates:
        if (candidate / "docs_audit" / "enterprise_store.py").exists():
            return candidate.resolve()
    raise RuntimeError(
        "Could not locate docs-audit workspace. Pass --audit-root pointing at workflows/docs-audit."
    )


def import_audit_modules(audit_root: Path):
    sys.path.insert(0, str(audit_root))
    try:
        from docs_audit.config import load_env_file, settings_from_env
        from docs_audit.enterprise_store import DocsAuditEnterpriseStore
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise RuntimeError(
            f"Missing Python dependency '{missing}'. Run this script with the docs repo environment, "
            "for example: cd /Users/prrao/code/docs && uv run python "
            "workflows/docs-audit/skills/docs-audit-enterprise-summary/scripts/query_recent_gaps.py ..."
        ) from exc

    return load_env_file, settings_from_env, DocsAuditEnterpriseStore


def rows_from_head(table: Any, limit: int) -> list[dict[str, Any]]:
    arrow = table.head(limit)
    drop = [name for name in DROP_COLUMNS if name in arrow.column_names]
    if drop:
        arrow = arrow.drop(drop)
    return arrow.to_pylist()


def iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return str(value)


def clean_text(text: str) -> str:
    stripped = " ".join(text.strip().split())
    if stripped.startswith("- "):
        return stripped[2:]
    return stripped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retrieve recent docs-audit gaps (id, date, description) from Enterprise LanceDB."
    )
    parser.add_argument("--audit-root", help="Path to workflows/docs-audit.")
    parser.add_argument(
        "--upto",
        type=int,
        default=7,
        help="How many days back from now to retrieve gaps for (default: 7).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Maximum rows to fetch with head().",
    )
    args = parser.parse_args()

    if args.upto <= 0:
        parser.error("--upto must be a positive number of days")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.upto)

    audit_root = locate_audit_root(args.audit_root)
    load_env_file, settings_from_env, store_cls = import_audit_modules(audit_root)

    load_env_file()
    settings = settings_from_env()
    store = store_cls(
        uri=settings.docs_audit_db_uri,
        api_key=settings.lancedb_api_key,
        host_override=settings.lancedb_host_override,
        region=settings.lancedb_region,
    )

    finding_rows = rows_from_head(store.db.open_table("docs_audit_findings"), args.limit)

    gaps = []
    for row in finding_rows:
        completed_at = row.get("completed_at")
        if not isinstance(completed_at, datetime):
            continue
        completed_at = completed_at.astimezone(timezone.utc)
        if start <= completed_at < end and row.get("visibility_class") == "public-doc-gap":
            gaps.append(
                {
                    "id": row.get("id"),
                    "date": iso(completed_at),
                    "description": clean_text(str(row.get("finding_text") or "")),
                }
            )

    gaps.sort(key=lambda item: (item["date"], item["id"] or ""))

    payload = {
        "window": {
            "from": iso(start),
            "to": iso(end),
            "days": args.upto,
        },
        "total_gaps": len(gaps),
        "gaps": gaps,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
