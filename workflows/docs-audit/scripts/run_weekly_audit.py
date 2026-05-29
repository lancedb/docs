#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs_audit.config import load_env_file, settings_from_env
from docs_audit.enterprise_store import (
    DocsAuditEnterpriseStore,
    json_string,
    parse_timestamp,
)
from docs_audit.openai_client import OpenAIClient
from docs_audit.report_parser import Finding, embedding_text, parse_report_findings
from docs_audit.visibility import is_public_finding


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    print(f"[{timestamp}] {message}", flush=True)


def run_json_command(args: list[str]) -> dict[str, Any]:
    log(f"running deterministic step: {' '.join(args)}")
    completed = subprocess.run(
        [sys.executable, "-m", "docs_audit.deterministic_runner", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed: python -m docs_audit.deterministic_runner {' '.join(args)}\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    payload = json.loads(completed.stdout)
    if args and args[0] == "select-areas":
        log(
            "selected areas: "
            + ", ".join(payload.get("selected_areas", []))
            + f" (changed: {', '.join(payload.get('changed_areas', [])) or 'none'})"
        )
    elif args and args[0] == "prepare":
        log(
            f"prepared area={args[args.index('--area') + 1] if '--area' in args else '?'} "
            f"run={payload.get('run_id')} pages={len(payload.get('selected_pages', []))}"
        )
    elif args and args[0] == "complete":
        log(f"completed run={payload.get('run_id')} dir={payload.get('run_dir')}")
    return payload


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            raise
        data = json.loads(stripped[start : end + 1])
    if not isinstance(data, dict):
        raise RuntimeError("Expected the page audit response to be a JSON object")
    return data


def list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [" ".join(str(item).strip().split()) for item in value if str(item).strip()]


def page_audit_prompt(
    *,
    guidelines: str,
    page_bundle: dict[str, Any],
) -> str:
    return f"""You are running one page-scoped pass of the LanceDB docs-gap audit.

Apply these page audit guidelines:

{guidelines}

Return only a JSON object with this exact shape:

{{
  "code_claims": ["stable user-visible claims from the evidence bundle"],
  "doc_claims": ["claims already present in the docs signals"],
  "candidate_gaps": ["missing-doc candidates, including lower-confidence candidates"],
  "report_observations": ["final concise missing-doc observations for report.md"]
}}

Rules:
- Include only missing documentation observations.
- Do not propose documentation patches.
- Do not summarize implementation details unless they expose user-visible behavior missing from docs.
- Do not include helm chart or enterprise deployment observations.
- If there are no material gaps, use an empty report_observations array.
- Keep report_observations self-contained and suitable as markdown bullets.

Page bundle JSON:

```json
{json.dumps(page_bundle, indent=2, sort_keys=True)}
```
"""


def audit_page(
    *,
    client: OpenAIClient,
    model: str,
    reasoning_effort: str,
    guidelines: str,
    page_bundle: dict[str, Any],
) -> dict[str, list[str]]:
    response_text = client.response_text(
        model=model,
        reasoning_effort=reasoning_effort,
        input_text=page_audit_prompt(guidelines=guidelines, page_bundle=page_bundle),
    )
    data = extract_json_object(response_text)
    return {
        "code_claims": list_of_strings(data.get("code_claims")),
        "doc_claims": list_of_strings(data.get("doc_claims")),
        "candidate_gaps": list_of_strings(data.get("candidate_gaps")),
        "report_observations": [
            text for text in list_of_strings(data.get("report_observations")) if is_public_finding(text)
        ],
    }


def write_page_outputs(
    *,
    llm_dir: Path,
    page_id: str,
    output: dict[str, list[str]],
) -> None:
    write_json(llm_dir / f"{page_id}.code_claims.json", output["code_claims"])
    write_json(llm_dir / f"{page_id}.doc_claims.json", output["doc_claims"])
    write_json(llm_dir / f"{page_id}.candidate_gaps.json", output["candidate_gaps"])
    write_json(llm_dir / f"{page_id}.report_observations.json", output["report_observations"])


def build_report(page_outputs: list[tuple[dict[str, Any], dict[str, list[str]]]]) -> str:
    lines = ["# Missing Documentation Observations", ""]
    wrote_any = False
    for bundle, output in page_outputs:
        observations = output["report_observations"]
        if not observations:
            continue
        wrote_any = True
        lines.extend([f"## {bundle['page_title']}", ""])
        for observation in observations:
            lines.append(f"- {observation}")
        lines.append("")
    if not wrote_any:
        lines.append("No material missing documentation observations for the selected pages.")
        lines.append("")
    return "\n".join(lines)


def page_lookup_from_bundles(bundles: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for bundle in bundles:
        title = str(bundle.get("page_title") or "")
        if not title:
            continue
        lookup[title.casefold()] = {
            "page_id": str(bundle.get("page_id") or ""),
            "page_title": title,
            "page_path": str(bundle.get("page_path") or ""),
        }
    return lookup


def audit_pending_run(
    *,
    run_dir: Path,
    client: OpenAIClient,
    model: str,
    reasoning_effort: str,
) -> None:
    selected = read_json(run_dir / "selected_pages.json")
    guidelines = (ROOT / "prompts" / "page_audit_guidelines.md").read_text(encoding="utf-8")
    page_outputs: list[tuple[dict[str, Any], dict[str, list[str]]]] = []
    log(f"auditing run={run_dir.name} selected_pages={len(selected['selected_pages'])}")
    for page_id in selected["selected_pages"]:
        bundle = read_json(run_dir / "page_bundles" / f"{page_id}.json")
        log(
            f"auditing page run={run_dir.name} page={page_id} "
            f"model={model} effort={reasoning_effort}"
        )
        output = audit_page(
            client=client,
            model=model,
            reasoning_effort=reasoning_effort,
            guidelines=guidelines,
            page_bundle=bundle,
        )
        write_page_outputs(llm_dir=run_dir / "llm_outputs", page_id=page_id, output=output)
        log(
            f"page audited run={run_dir.name} page={page_id} "
            f"observations={len(output['report_observations'])}"
        )
        page_outputs.append((bundle, output))
    report_text = build_report(page_outputs)
    (run_dir / "report.md").write_text(report_text, encoding="utf-8")
    log(f"wrote report run={run_dir.name} bytes={len(report_text.encode('utf-8'))}")


def prepare_selected_runs(*, refresh: bool, advance: bool) -> list[Path]:
    log(f"selecting areas refresh={refresh} advance={advance}")
    select_args = ["select-areas"]
    if refresh:
        select_args.append("--refresh")
    if advance:
        select_args.append("--advance")
    selection = run_json_command(select_args)
    run_dirs: list[Path] = []
    for area in selection["selected_areas"]:
        log(f"preparing area={area}")
        prepared = run_json_command(["prepare", "--area", area])
        run_dirs.append(Path(prepared["run_dir"]))
    return run_dirs


def complete_run(run_dir: Path) -> Path:
    run_id = run_dir.name
    log(f"completing run={run_id}")
    completed = run_json_command(["complete", "--run-id", run_id])
    return Path(completed["run_dir"])


def repo_shas(metadata: dict[str, Any]) -> dict[str, str]:
    shas: dict[str, str] = {}
    for item in metadata.get("refresh", []):
        repo = item.get("repo")
        sha = item.get("sha_after") or item.get("sha_before")
        if repo and sha:
            shas[str(repo)] = str(sha)
    return shas


def run_row_from_metadata(
    *,
    metadata: dict[str, Any],
    report_text: str,
    report_path: Path,
) -> dict[str, Any]:
    return {
        "run_id": metadata["run_id"],
        "completed_at": parse_timestamp(metadata.get("completed_at")),
        "areas": [metadata["area"]],
        "report_text": report_text,
        "report_path": str(report_path),
        "repo_shas": json_string(repo_shas(metadata)),
        "selected_pages": [str(item) for item in metadata.get("selected_pages", [])],
        "changed_pages": [str(item) for item in metadata.get("changed_pages", [])],
        "refresh": json_string(metadata.get("refresh", [])),
        "metadata": json_string(metadata),
    }


def finding_rows(
    *,
    findings: list[Finding],
    embeddings: list[list[float]],
    completed_at: Any,
) -> list[dict[str, Any]]:
    rows = []
    for finding, vector in zip(findings, embeddings, strict=True):
        rows.append(
            {
                "id": finding.id,
                "run_id": finding.run_id,
                "completed_at": completed_at,
                "area": finding.area,
                "page_id": finding.page_id or "",
                "page_title": finding.page_title or "",
                "page_path": finding.page_path or "",
                "report_heading": finding.report_heading,
                "finding_index": finding.finding_index,
                "finding_text": finding.finding_text,
                "finding_hash": finding.finding_hash,
                "visibility_class": finding.visibility_class,
                "embedding_text": embedding_text(finding),
                "embedding": [float(value) for value in vector],
                "metadata": json_string({}),
            }
        )
    return rows


def debug_finding_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    debug_rows: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        completed_at = item.get("completed_at")
        if hasattr(completed_at, "isoformat"):
            item["completed_at"] = completed_at.isoformat()
        item["embedding"] = []
        debug_rows.append(item)
    return debug_rows


def ingest_completed_run(
    *,
    run_dir: Path,
    openai_client: OpenAIClient | None,
    embedding_model: str,
    store: DocsAuditEnterpriseStore | None,
) -> dict[str, Any]:
    metadata = read_json(run_dir / "metadata.json")
    log(f"parsing completed report run={metadata['run_id']} area={metadata['area']}")
    report_path = run_dir / "report.md"
    report_text = report_path.read_text(encoding="utf-8")
    bundles = [
        read_json(path)
        for path in sorted((run_dir / "page_bundles").glob("*.json"))
    ]
    findings = [
        finding
        for finding in parse_report_findings(
            report_text,
            run_id=metadata["run_id"],
            area=metadata["area"],
            page_lookup=page_lookup_from_bundles(bundles),
        )
        if finding.visibility_class == "public-doc-gap"
    ]
    texts = [embedding_text(finding) for finding in findings]
    log(f"parsed findings run={metadata['run_id']} public_findings={len(findings)}")
    vectors: list[list[float]]
    if store is None:
        log(f"skipping embeddings and Enterprise write run={metadata['run_id']}")
        vectors = [[] for _finding in findings]
    else:
        if openai_client is None:
            raise RuntimeError("OpenAI client is required when writing embedded findings")
        log(
            f"generating embeddings run={metadata['run_id']} "
            f"count={len(texts)} model={embedding_model}"
        )
        vectors = openai_client.embeddings(model=embedding_model, inputs=texts)
    run_row = run_row_from_metadata(
        metadata=metadata,
        report_text=report_text,
        report_path=report_path,
    )
    rows = finding_rows(
        findings=findings,
        embeddings=vectors,
        completed_at=run_row["completed_at"],
    )
    write_json(run_dir / "llm_outputs" / "parsed_findings.json", debug_finding_rows(rows))
    stored = {"runs": 0, "findings": 0}
    if store is not None:
        log(f"writing Enterprise rows run={metadata['run_id']} findings={len(rows)}")
        stored = store.replace_run(run_row=run_row, finding_rows=rows)
        log(
            f"wrote Enterprise rows run={metadata['run_id']} "
            f"runs={stored['runs']} findings={stored['findings']}"
        )
    return {
        "run_id": metadata["run_id"],
        "area": metadata["area"],
        "findings": len(findings),
        "stored": stored,
        "report_path": str(report_path),
    }


def build_summary(results: list[dict[str, Any]]) -> str:
    total = sum(int(result["findings"]) for result in results)
    lines = [f"Docs audit completed: {len(results)} run(s), {total} public finding(s)."]
    for result in results:
        lines.append(
            f"- {result['area']} `{result['run_id']}`: {result['findings']} finding(s), report {result['report_path']}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenAI API-driven docs-gap audit runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Common modes:

              No-API parser smoke test:
                uv run python scripts/run_weekly_audit.py --ingest-run-dir artifacts/runs/<run_id> --skip-write

              Local audit generation without Enterprise writes:
                uv run python scripts/run_weekly_audit.py --no-refresh --no-advance --skip-write

              Backfill one completed run into Enterprise:
                uv run python scripts/run_weekly_audit.py --ingest-run-dir artifacts/runs/<run_id>

              Weekly EC2 cron mode:
                uv run python scripts/run_weekly_audit.py

            Cost/network behavior:
              --ingest-run-dir with --skip-write does not call OpenAI and does not write to LanceDB.
              --skip-write skips embeddings and LanceDB writes, but audit generation still calls GPT-5.5.
              Omitting --no-refresh allows select-areas to run git pull --ff-only on watched repos.
              Omitting --no-advance updates the weekly area rotation cursor.
            """
        ),
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Skip git pull --ff-only during area selection. Useful for local tests.",
    )
    parser.add_argument(
        "--no-advance",
        action="store_true",
        help="Do not advance the area rotation cursor. Useful for repeatable local tests.",
    )
    parser.add_argument(
        "--skip-write",
        action="store_true",
        help="Do not generate embeddings or write LanceDB Enterprise rows. Audit generation may still call OpenAI.",
    )
    parser.add_argument(
        "--ingest-run-dir",
        type=Path,
        help="Parse an existing completed run directory instead of selecting areas or generating a new report.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    log("starting weekly docs audit runner")
    load_env_file()
    settings = settings_from_env()
    log(
        f"mode ingest_run_dir={args.ingest_run_dir or 'none'} "
        f"skip_write={args.skip_write} no_refresh={args.no_refresh} no_advance={args.no_advance}"
    )
    needs_openai = args.ingest_run_dir is None or not args.skip_write
    openai_client = None
    if needs_openai:
        log("initializing OpenAI client")
        openai_client = OpenAIClient(
            api_key=settings.openai_api_key,
            timeout_seconds=settings.openai_timeout_seconds,
        )
    store = None
    if not args.skip_write:
        log(f"connecting to LanceDB Enterprise uri={settings.docs_audit_db_uri}")
        store = DocsAuditEnterpriseStore(
            uri=settings.docs_audit_db_uri,
            api_key=settings.lancedb_api_key,
            host_override=settings.lancedb_host_override,
            region=settings.lancedb_region,
        )

    completed_dirs: list[Path] = []
    if args.ingest_run_dir is not None:
        completed_dirs = [args.ingest_run_dir]
    else:
        pending_dirs = prepare_selected_runs(refresh=not args.no_refresh, advance=not args.no_advance)
        if openai_client is None:
            raise RuntimeError("OpenAI client is required for audit generation")
        for pending_dir in pending_dirs:
            audit_pending_run(
                run_dir=pending_dir,
                client=openai_client,
                model=settings.audit_model,
                reasoning_effort=settings.audit_reasoning_effort,
            )
            completed_dirs.append(complete_run(pending_dir))

    results = [
        ingest_completed_run(
            run_dir=run_dir,
            openai_client=openai_client,
            embedding_model=settings.embedding_model,
            store=store,
        )
        for run_dir in completed_dirs
    ]
    print(build_summary(results))
    log("weekly docs audit runner finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
