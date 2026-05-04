#!/usr/bin/env python3
"""Deterministic runner for the local docs-gap audit workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.toml"
DEFAULT_PENDING_ARTIFACTS_DIR = "artifacts/pending"
EXCERPT_LIMIT = 12
SYMBOL_PATTERNS = [
    re.compile(r"\b(pub\s+enum|pub\s+struct|pub\s+fn|async\s+fn|fn\s+test_|def\s+test_|test\(|describe\()"),
]
ADMONITION_RE = re.compile(r"<(Note|Tip|Warning|Info|Badge)\b")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
IMPORT_RE = re.compile(r"from\s+['\"](/snippets/[^'\"]+)['\"]")
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


@dataclass
class RepoInfo:
    name: str
    path: Path


@dataclass
class Page:
    id: str
    title: str
    path: str
    keywords: list[str]


@dataclass
class Source:
    id: str
    repo: str
    kind: str
    applies_to: list[str]
    paths: list[str]
    extract_keywords: list[str]


@dataclass
class AreaManifest:
    name: str
    description: str
    docs_repo: str
    rotation_unit: str
    keywords: list[str]
    pages: list[Page]
    sources: list[Source]


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_config(path: Path) -> dict[str, Any]:
    config = load_toml(path)
    config_dir = path.resolve().parent
    for repo in config.get("repos", {}).values():
        repo_path = Path(repo["path"])
        if not repo_path.is_absolute():
            repo["path"] = str((config_dir / repo_path).resolve())
    return config


def load_manifest(path: Path) -> AreaManifest:
    data = load_toml(path)
    pages = [Page(**page) for page in data.get("pages", [])]
    sources = [Source(**source) for source in data.get("sources", [])]
    return AreaManifest(
        name=data["name"],
        description=data.get("description", ""),
        docs_repo=data["docs_repo"],
        rotation_unit=data.get("rotation_unit", "page"),
        keywords=data.get("keywords", []),
        pages=pages,
        sources=sources,
    )


def run_git(repo: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def line_matches(text: str, keywords: list[str], limit: int = EXCERPT_LIMIT) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    lowered = [kw.lower() for kw in keywords if kw.strip()]
    if not lowered:
        return results
    lines = text.splitlines()
    for idx, line in enumerate(lines, start=1):
        line_l = line.lower()
        matched = [kw for kw in lowered if kw in line_l]
        if not matched:
            continue
        start = max(0, idx - 2)
        end = min(len(lines), idx + 1)
        excerpt = "\n".join(lines[start:end]).strip()
        results.append({
            "line": idx,
            "matched_keywords": matched,
            "excerpt": excerpt,
        })
        if len(results) >= limit:
            break
    return results


def symbol_lines(text: str, limit: int = EXCERPT_LIMIT) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        if any(pattern.search(line) for pattern in SYMBOL_PATTERNS):
            results.append({"line": idx, "text": line.strip()})
            if len(results) >= limit:
                break
    return results


def doc_comments(text: str, keywords: list[str], limit: int = EXCERPT_LIMIT) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    comment_prefixes = ("///", "//", "#", "<!--")
    lowered = [kw.lower() for kw in keywords if kw.strip()]
    for idx, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped.startswith(comment_prefixes):
            continue
        line_l = stripped.lower()
        matched = [kw for kw in lowered if kw in line_l]
        if not matched:
            continue
        results.append({"line": idx, "matched_keywords": matched, "text": stripped})
        if len(results) >= limit:
            break
    return results


def frontmatter(text: str) -> dict[str, Any]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}
    block = match.group(1)
    info: dict[str, Any] = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        info[key.strip()] = value.strip().strip('"')
    return info


def headings(text: str) -> list[str]:
    found: list[str] = []
    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            found.append(match.group(2).strip())
    return found


def snippet_imports(text: str) -> list[str]:
    return sorted(set(IMPORT_RE.findall(text)))


def admonitions(text: str) -> list[str]:
    return ADMONITION_RE.findall(text)


def extract_docs_signals(text: str, keywords: list[str]) -> dict[str, Any]:
    return {
        "frontmatter": frontmatter(text),
        "headings": headings(text),
        "imports": snippet_imports(text),
        "admonitions": admonitions(text),
        "keyword_matches": line_matches(text, keywords),
    }


def extract_code_signals(text: str, keywords: list[str]) -> dict[str, Any]:
    return {
        "keyword_matches": line_matches(text, keywords),
        "doc_comments": doc_comments(text, keywords),
        "symbol_lines": symbol_lines(text),
    }


def repo_snapshot(repo: RepoInfo, refresh: bool, simulate_failure: bool) -> dict[str, Any]:
    branch = run_git(repo.path, ["rev-parse", "--abbrev-ref", "HEAD"]) 
    sha_before = run_git(repo.path, ["rev-parse", "HEAD"])
    dirty = run_git(repo.path, ["status", "--porcelain"])
    status = {
        "repo": repo.name,
        "path": str(repo.path),
        "branch": branch.stdout.strip(),
        "sha_before": sha_before.stdout.strip(),
        "dirty": bool(dirty.stdout.strip()),
        "refresh_requested": refresh,
        "refresh_status": "not_requested",
        "message": "",
    }
    if not refresh:
        return status
    if simulate_failure:
        status["refresh_status"] = "skipped"
        status["message"] = "Simulated refresh failure"
        return status
    pull = run_git(repo.path, ["pull", "--ff-only"])
    if pull.returncode == 0:
        sha_after = run_git(repo.path, ["rev-parse", "HEAD"])
        status["refresh_status"] = "refreshed"
        status["sha_after"] = sha_after.stdout.strip()
        status["message"] = pull.stdout.strip() or "fast-forward pull succeeded"
    else:
        sha_after = run_git(repo.path, ["rev-parse", "HEAD"])
        status["refresh_status"] = "skipped"
        status["sha_after"] = sha_after.stdout.strip()
        status["message"] = (pull.stderr.strip() or pull.stdout.strip() or "git pull --ff-only failed")
    return status


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def completed_runs_dir(config: dict[str, Any]) -> Path:
    return ROOT / config["paths"]["artifacts_dir"]


def pending_runs_dir(config: dict[str, Any]) -> Path:
    return ROOT / config["paths"].get("pending_artifacts_dir", DEFAULT_PENDING_ARTIFACTS_DIR)


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "areas": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_run_id_timestamp(run_id: str) -> datetime | None:
    try:
        return datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def run_dir_timestamp(run_dir: Path) -> datetime:
    parsed = parse_run_id_timestamp(run_dir.name)
    if parsed is not None:
        return parsed
    return datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc)


def clear_deleted_run_references(state_path: Path, deleted_run_ids: set[str]) -> None:
    state = load_state(state_path)
    changed = False
    for area_state in state.get("areas", {}).values():
        if area_state.get("last_completed_run") not in deleted_run_ids:
            continue
        for key in (
            "last_completed_run",
            "last_selected_pages",
            "last_changed_pages",
            "last_report_path",
            "completed_at",
        ):
            if key in area_state:
                del area_state[key]
                changed = True
    if changed:
        write_json(state_path, state)


def refresh_latest_run_file(latest_run_file: Path, runs_dir: Path, deleted_run_ids: set[str]) -> None:
    if not latest_run_file.exists():
        return
    latest = json.loads(latest_run_file.read_text(encoding="utf-8"))
    if latest.get("run_id") not in deleted_run_ids:
        return
    remaining_runs = sorted(
        path for path in runs_dir.iterdir() if path.is_dir() and (path / "report.md").exists()
    )
    if not remaining_runs:
        latest_run_file.unlink()
        return
    newest = remaining_runs[-1]
    report_path = newest / "report.md"
    replacement = {
        "run_id": newest.name,
        "run_dir": str(newest),
    }
    metadata_path = newest / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        replacement["area"] = metadata.get("area", "")
        replacement["completed_at"] = metadata.get("completed_at", metadata.get("created_at", ""))
    if report_path.exists():
        replacement["report_path"] = str(report_path)
    write_json(latest_run_file, replacement)


def sources_for_page(manifest: AreaManifest, page_id: str) -> list[Source]:
    results = []
    for source in manifest.sources:
        if page_id in source.applies_to or "*" in source.applies_to:
            results.append(source)
    return results


def collect_page_bundle(
    manifest: AreaManifest,
    config: dict[str, Any],
    page: Page,
    repos: dict[str, RepoInfo],
) -> dict[str, Any]:
    docs_repo = repos[manifest.docs_repo]
    page_path = docs_repo.path / page.path
    page_text = read_text(page_path)
    docs_keywords = list(dict.fromkeys(manifest.keywords + page.keywords))
    docs_signals = extract_docs_signals(page_text, docs_keywords)
    evidence: list[dict[str, Any]] = []
    fingerprint_parts = [sha256_file(page_path)]
    for source in sources_for_page(manifest, page.id):
        repo = repos[source.repo]
        source_keywords = list(dict.fromkeys(docs_keywords + source.extract_keywords))
        for rel_path in source.paths:
            abs_path = repo.path / rel_path
            if not abs_path.exists():
                evidence.append({
                    "source_id": source.id,
                    "repo": source.repo,
                    "kind": source.kind,
                    "path": rel_path,
                    "missing": True,
                })
                fingerprint_parts.append(sha256_text(f"missing:{source.repo}:{rel_path}"))
                continue
            text = read_text(abs_path)
            fingerprint_parts.append(sha256_file(abs_path))
            evidence.append({
                "source_id": source.id,
                "repo": source.repo,
                "kind": source.kind,
                "path": rel_path,
                "missing": False,
                "signals": extract_code_signals(text, source_keywords),
            })
    fingerprint = sha256_text("\n".join(fingerprint_parts))
    return {
        "area": manifest.name,
        "page_id": page.id,
        "page_title": page.title,
        "page_path": page.path,
        "page_keywords": docs_keywords,
        "page_fingerprint": fingerprint,
        "docs_signals": docs_signals,
        "evidence": evidence,
    }


def select_pages(
    pages: list[Page],
    fingerprints: dict[str, str],
    previous_fingerprints: dict[str, str],
    rotation_index: int,
    rotation_extra_pages: int,
) -> tuple[list[str], list[str], int]:
    ordered_ids = [page.id for page in pages]
    changed = [page_id for page_id in ordered_ids if previous_fingerprints.get(page_id) != fingerprints[page_id]]
    selected = list(changed)
    next_index = rotation_index
    extras_added = 0
    while ordered_ids and extras_added < rotation_extra_pages:
        candidate = ordered_ids[next_index % len(ordered_ids)]
        next_index = (next_index + 1) % len(ordered_ids)
        if candidate in selected:
            if len(selected) == len(ordered_ids):
                break
            continue
        selected.append(candidate)
        extras_added += 1
    return changed, selected, next_index


def prepare(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    repos = {
        name: RepoInfo(name=name, path=Path(info["path"]))
        for name, info in config["repos"].items()
    }
    manifest = load_manifest(ROOT / config["paths"]["manifests_dir"] / f"{args.area}.toml")
    state_path = ROOT / config["paths"]["state_file"]
    state = load_state(state_path)
    area_state = state.get("areas", {}).get(args.area, {})
    previous_fingerprints = area_state.get("page_fingerprints", {})
    rotation_index = int(area_state.get("rotation_index", 0))
    run_id = utc_run_id()
    run_dir = pending_runs_dir(config) / run_id
    page_dir = run_dir / "page_bundles"
    llm_dir = run_dir / "llm_outputs"
    ensure_dir(page_dir)
    ensure_dir(llm_dir)

    refresh_results = []
    simulated = set(args.simulate_refresh_failure or [])
    for repo in repos.values():
        refresh_results.append(repo_snapshot(repo, args.refresh, repo.name in simulated))

    page_fingerprints: dict[str, str] = {}
    page_bundles: dict[str, dict[str, Any]] = {}
    for page in manifest.pages:
        bundle = collect_page_bundle(manifest, config, page, repos)
        page_bundles[page.id] = bundle
        page_fingerprints[page.id] = bundle["page_fingerprint"]
        write_json(page_dir / f"{page.id}.json", bundle)

    changed_pages, selected_pages, next_rotation_index = select_pages(
        manifest.pages,
        page_fingerprints,
        previous_fingerprints,
        rotation_index,
        int(config["selection"]["rotation_extra_pages"]),
    )

    selected_payload = {
        "area": args.area,
        "changed_pages": changed_pages,
        "selected_pages": selected_pages,
    }
    write_json(run_dir / "selected_pages.json", selected_payload)

    created_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "run_id": run_id,
        "area": args.area,
        "status": "prepared",
        "manifest": {
            "name": manifest.name,
            "description": manifest.description,
            "pages": [page.id for page in manifest.pages],
        },
        "refresh": refresh_results,
        "page_fingerprints": page_fingerprints,
        "rotation": {
            "previous_index": rotation_index,
            "next_index": next_rotation_index,
            "rotation_extra_pages": int(config["selection"]["rotation_extra_pages"]),
        },
        "selected_pages": selected_pages,
        "changed_pages": changed_pages,
        "created_at": created_at,
        "prepared_at": created_at,
    }
    write_json(run_dir / "metadata.json", metadata)

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "final_run_dir": str(completed_runs_dir(config) / run_id),
        "pending": True,
        "selected_pages": selected_pages,
        "changed_pages": changed_pages,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def complete(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    state_path = ROOT / config["paths"]["state_file"]
    latest_run_file = ROOT / config["paths"]["latest_run_file"]
    final_run_dir = completed_runs_dir(config) / args.run_id
    pending_run_dir = pending_runs_dir(config) / args.run_id
    run_dir = pending_run_dir if pending_run_dir.exists() else final_run_dir
    metadata_path = run_dir / "metadata.json"
    selected_path = run_dir / "selected_pages.json"
    report_path = run_dir / "report.md"
    if not run_dir.exists():
        raise SystemExit(
            f"Missing prepared run {args.run_id}. Looked in {pending_run_dir} and {final_run_dir}"
        )
    if not metadata_path.exists():
        raise SystemExit(f"Missing metadata.json for run {args.run_id}")
    if not selected_path.exists():
        raise SystemExit(f"Missing selected_pages.json for run {args.run_id}")
    if not report_path.exists():
        raise SystemExit(f"Missing report.md for run {args.run_id}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    completed_at = datetime.now(timezone.utc).isoformat()
    if run_dir == pending_run_dir:
        if final_run_dir.exists():
            raise SystemExit(f"Cannot publish run {args.run_id}: {final_run_dir} already exists")
        ensure_dir(final_run_dir.parent)
        shutil.move(str(pending_run_dir), str(final_run_dir))
        run_dir = final_run_dir
        metadata_path = run_dir / "metadata.json"
        report_path = run_dir / "report.md"

    metadata["status"] = "completed"
    metadata["completed_at"] = completed_at
    write_json(metadata_path, metadata)

    state = load_state(state_path)
    areas = state.setdefault("areas", {})
    area_state = areas.setdefault(metadata["area"], {})
    area_state["page_fingerprints"] = metadata["page_fingerprints"]
    area_state["rotation_index"] = metadata["rotation"]["next_index"]
    area_state["last_completed_run"] = args.run_id
    area_state["last_selected_pages"] = selected["selected_pages"]
    area_state["last_changed_pages"] = selected["changed_pages"]
    area_state["last_report_path"] = str(report_path)
    area_state["completed_at"] = completed_at
    write_json(state_path, state)
    write_json(
        latest_run_file,
        {
            "run_id": args.run_id,
            "area": metadata["area"],
            "run_dir": str(run_dir),
            "report_path": str(report_path),
            "completed_at": completed_at,
        },
    )
    print(json.dumps({"completed": True, "run_id": args.run_id, "run_dir": str(run_dir)}, indent=2))
    return 0


def cleanup(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    runs_dir = completed_runs_dir(config)
    latest_run_file = ROOT / config["paths"]["latest_run_file"]
    state_path = ROOT / config["paths"]["state_file"]
    if not runs_dir.exists():
        print(json.dumps({"deleted_run_ids": [], "dry_run": args.dry_run, "kept_run_ids": []}, indent=2))
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
    run_dirs = sorted(path for path in runs_dir.iterdir() if path.is_dir())
    to_delete = [run_dir for run_dir in run_dirs if run_dir_timestamp(run_dir) < cutoff]
    kept = [run_dir.name for run_dir in run_dirs if run_dir not in to_delete]
    deleted = [run_dir.name for run_dir in to_delete]

    if not args.dry_run:
        for run_dir in to_delete:
            shutil.rmtree(run_dir)
        deleted_ids = set(deleted)
        clear_deleted_run_references(state_path, deleted_ids)
        refresh_latest_run_file(latest_run_file, runs_dir, deleted_ids)

    print(
        json.dumps(
            {
                "cutoff": cutoff.isoformat(),
                "days": args.days,
                "dry_run": args.dry_run,
                "deleted_run_ids": deleted,
                "kept_run_ids": kept,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local docs-gap audit runner")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.toml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Refresh repos, build evidence bundles, and select pages")
    prepare_parser.add_argument("--area", required=True, help="Area manifest name, e.g. indexing")
    prepare_parser.add_argument("--refresh", action="store_true", help="Attempt git pull --ff-only on watched repos")
    prepare_parser.add_argument(
        "--simulate-refresh-failure",
        action="append",
        choices=["lancedb", "docs", "sophon"],
        help="Simulate an unrefreshable repo for manual validation",
    )
    prepare_parser.set_defaults(func=prepare)

    complete_parser = subparsers.add_parser("complete", help="Mark a run complete and update state")
    complete_parser.add_argument("--run-id", required=True, help="Run ID returned by prepare")
    complete_parser.set_defaults(func=complete)

    cleanup_parser = subparsers.add_parser(
        "cleanup",
        help="Delete old generated run artifacts under artifacts/runs",
    )
    cleanup_parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Delete run directories older than this many days (default: 30)",
    )
    cleanup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which runs would be deleted without removing anything",
    )
    cleanup_parser.set_defaults(func=cleanup)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
