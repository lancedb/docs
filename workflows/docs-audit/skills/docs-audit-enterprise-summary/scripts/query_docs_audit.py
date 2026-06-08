#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta, timezone
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

THEME_RULES = {
    "indexing": [
        ("Index readiness and post-write lifecycle", ("wait_for_index", "num_unindexed_rows", "continuous writes", "index_stats", "optimize()")),
        ("FTS index/search constraints", ("fts", "full-text", "bm25", "n-gram", "ngram", "tantivy")),
        ("Vector metric and training correctness", ("distance metric", "train", "training", "metric used")),
        ("Scalar and quantized index coverage", ("scalar", "btree", "bitmap", "labellist", "ivf_sq", "quantization", "quantized")),
    ],
    "search": [
        ("FTS and SQL FTS workflows", ("fts", "full-text", "nearest_to_text", "nearesttotext", "tantivy", "sql")),
        ("Hybrid filter and execution behavior", ("hybrid", "prefilter", "postfilter", "filter", "subqueries")),
        ("Vector and multivector constraints", ("vector", "multivector", "cosine", "dimension", "row id")),
        ("Query diagnostics, limits, and pagination", ("explain_plan", "analyze_plan", "limit", "offset", "profile", "plan")),
    ],
    "table-operations": [
        ("Create and ingestion semantics", ("create-table", "overwrite", "exist_ok", "ingestion", "table.add", "append")),
        ("Schema and nullability enforcement", ("schema", "nullable", "nullability", "non-nullable", "vector inference")),
        ("Mutation return/status semantics", ("update", "delete", "merge", "returned status", "row counts", "commit version")),
        ("Versioning workflows", ("version", "checkout", "restore", "tags", "managed_versioning")),
        ("Blob schema contracts", ("blob", "largebinary", "large blob", "lance-encoding")),
    ],
    "reranking": [
        ("Reranker result and score contract", ("_relevance_score", "return_score", "descending", "empty", "score")),
        ("Rank-fusion scoring knobs", ("linearcombination", "rrf", "mrr", "weight", "fill")),
        ("Provider/model reranker coverage", ("cohere", "colbert", "openai", "jina", "answerai", "voyage")),
        ("Cross-SDK support boundaries", ("typescript", "rust", "hybrid reranking", "rerank_vector", "rerank_fts")),
    ],
}


def parse_instant(value: str, *, end: bool) -> datetime:
    text = value.strip()
    if not text:
        raise ValueError("empty date")
    date_only = len(text) == 10 and text[4] == "-" and text[7] == "-"
    if date_only:
        parsed_date = date.fromisoformat(text)
        if end:
            return datetime.combine(parsed_date + timedelta(days=1), time.min, timezone.utc)
        return datetime.combine(parsed_date, time.min, timezone.utc)
    normalized = text.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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
            "workflows/docs-audit/skills/docs-audit-enterprise-summary/scripts/query_docs_audit.py ..."
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


def clean_finding_text(text: str) -> str:
    stripped = " ".join(text.strip().split())
    if stripped.startswith("- "):
        return stripped[2:]
    return stripped


def classify_theme(area: str, text: str) -> str:
    lowered = text.lower()
    for theme, tokens in THEME_RULES.get(area, []):
        if any(token in lowered for token in tokens):
            return theme
    return "Other public docs gaps"


def build_digest(
    *,
    audit_root: Path,
    start: datetime,
    end_exclusive: datetime,
    limit: int,
    findings: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    examples_per_theme: int,
) -> dict[str, Any]:
    by_area = Counter(row.get("area") or "" for row in findings)
    by_page = Counter((row.get("area") or "", row.get("page_path") or "") for row in findings)
    pages_by_area: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (area, page_path), count in sorted(by_page.items()):
        pages_by_area[area].append({"page_path": page_path, "count": count})

    themes_by_area: dict[str, list[dict[str, Any]]] = defaultdict(list)
    theme_groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in sorted(findings, key=lambda item: (item.get("area") or "", item.get("page_path") or "", item.get("id") or "")):
        area = row.get("area") or ""
        text = clean_finding_text(str(row.get("finding_text") or ""))
        theme = classify_theme(area, text)
        key = (area, theme)
        group = theme_groups.setdefault(
            key,
            {
                "theme": theme,
                "count": 0,
                "pages": Counter(),
                "examples": [],
            },
        )
        group["count"] += 1
        group["pages"][row.get("page_path") or ""] += 1
        if len(group["examples"]) < examples_per_theme:
            group["examples"].append(
                {
                    "page_path": row.get("page_path") or "",
                    "finding": text,
                }
            )

    for (area, _theme), group in sorted(
        theme_groups.items(),
        key=lambda item: (item[0][0], -item[1]["count"], item[0][1]),
    ):
        themes_by_area[area].append(
            {
                "theme": group["theme"],
                "count": group["count"],
                "pages": [
                    {"page_path": page_path, "count": count}
                    for page_path, count in group["pages"].most_common()
                ],
                "examples": group["examples"],
            }
        )

    return {
        "audit_root": str(audit_root),
        "window": {
            "start": iso(start),
            "end_exclusive": iso(end_exclusive),
        },
        "limit": limit,
        "total_public_findings": len(findings),
        "runs": [
            {
                "run_id": row.get("run_id"),
                "completed_at": row.get("completed_at"),
                "areas": row.get("areas"),
                "selected_pages": row.get("selected_pages"),
            }
            for row in runs
        ],
        "counts_by_area": dict(sorted(by_area.items())),
        "pages_by_area": dict(sorted(pages_by_area.items())),
        "themes_by_area": dict(sorted(themes_by_area.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Query LanceDB docs-audit Enterprise findings for a UTC date window."
    )
    parser.add_argument("--audit-root", help="Path to workflows/docs-audit.")
    parser.add_argument("--start", required=True, help="UTC start date or timestamp.")
    parser.add_argument("--end", required=True, help="UTC inclusive end date or exclusive timestamp.")
    parser.add_argument("--limit", type=int, default=5000, help="Maximum rows to fetch with head().")
    parser.add_argument(
        "--format",
        choices=["digest-json", "digest-markdown", "json", "markdown"],
        default="digest-json",
        help="Use digest-* for compact high-signal output. Use json for raw findings.",
    )
    parser.add_argument(
        "--examples-per-theme",
        type=int,
        default=2,
        help="Representative finding examples to include for each digest theme.",
    )
    args = parser.parse_args()

    start = parse_instant(args.start, end=False)
    end_exclusive = parse_instant(args.end, end=True)
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

    run_rows = rows_from_head(store.db.open_table("docs_audit_runs"), args.limit)
    finding_rows = rows_from_head(store.db.open_table("docs_audit_findings"), args.limit)

    findings = []
    for row in finding_rows:
        completed_at = row.get("completed_at")
        if not isinstance(completed_at, datetime):
            continue
        completed_at = completed_at.astimezone(timezone.utc)
        if start <= completed_at < end_exclusive and row.get("visibility_class") == "public-doc-gap":
            item = dict(row)
            item["completed_at"] = iso(completed_at)
            findings.append(item)

    run_ids = sorted({row["run_id"] for row in findings})
    runs = []
    for row in run_rows:
        if row.get("run_id") in run_ids:
            item = dict(row)
            item["completed_at"] = iso(item.get("completed_at"))
            runs.append(item)

    digest = build_digest(
        audit_root=audit_root,
        start=start,
        end_exclusive=end_exclusive,
        limit=args.limit,
        findings=findings,
        runs=runs,
        examples_per_theme=max(args.examples_per_theme, 0),
    )

    raw_payload = {
        "audit_root": str(audit_root),
        "window": {
            "start": iso(start),
            "end_exclusive": iso(end_exclusive),
        },
        "limit": args.limit,
        "total_public_findings": len(findings),
        "runs": runs,
        "counts_by_area": digest["counts_by_area"],
        "pages_by_area": digest["pages_by_area"],
        "themes_by_area": digest["themes_by_area"],
        "findings": sorted(findings, key=lambda row: (row.get("area") or "", row.get("page_path") or "", row.get("id") or "")),
    }

    if args.format == "json":
        print(json.dumps(raw_payload, indent=2, sort_keys=True))
        return 0
    if args.format == "digest-json":
        print(json.dumps(digest, indent=2, sort_keys=True))
        return 0

    print("# Docs Audit Findings")
    print()
    print(f"- Window: `{digest['window']['start']}` to `{digest['window']['end_exclusive']}`")
    print(f"- Public findings: {digest['total_public_findings']}")
    print(f"- Runs: {len(runs)}")
    print()
    for area, count in digest["counts_by_area"].items():
        print(f"## {area} ({count})")
        for page in digest["pages_by_area"].get(area, []):
            print(f"- `{page['page_path']}`: {page['count']}")
        if args.format == "digest-markdown":
            print()
            for theme in digest["themes_by_area"].get(area, []):
                pages = ", ".join(item["page_path"] for item in theme["pages"][:3])
                print(f"### {theme['theme']} ({theme['count']})")
                if pages:
                    print(f"Pages: {pages}")
                for example in theme["examples"]:
                    print(f"- `{example['page_path']}`: {example['finding']}")
                print()
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
