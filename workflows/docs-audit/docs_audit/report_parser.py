from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .visibility import visibility_class


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
BULLET_RE = re.compile(r"^\s*[-*]\s+(.+?)\s*$")


@dataclass(frozen=True)
class Finding:
    id: str
    run_id: str
    area: str
    report_heading: str
    finding_index: int
    finding_text: str
    finding_hash: str
    visibility_class: str
    page_id: str | None = None
    page_title: str | None = None
    page_path: str | None = None


def normalize_finding_text(text: str) -> str:
    return " ".join(text.strip().split())


def finding_hash(area: str, page_path: str | None, heading: str, text: str) -> str:
    normalized = "\n".join(
        [
            area.strip().casefold(),
            (page_path or "").strip().casefold(),
            heading.strip().casefold(),
            normalize_finding_text(text).casefold(),
        ]
    )
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def parse_report_findings(
    report_text: str,
    *,
    run_id: str,
    area: str,
    page_lookup: dict[str, dict[str, str]] | None = None,
) -> list[Finding]:
    page_lookup = page_lookup or {}
    current_heading = ""
    findings: list[Finding] = []
    pending_index = 0

    for line in report_text.splitlines():
        heading_match = HEADING_RE.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            if level >= 2:
                current_heading = title
            continue

        bullet_match = BULLET_RE.match(line)
        if not bullet_match:
            continue

        text = normalize_finding_text(bullet_match.group(1))
        if not text:
            continue

        pending_index += 1
        page_info = page_lookup.get(current_heading.casefold(), {})
        page_path = page_info.get("page_path")
        digest = finding_hash(area, page_path, current_heading, text)
        findings.append(
            Finding(
                id=f"{run_id}:{pending_index:03d}",
                run_id=run_id,
                area=area,
                report_heading=current_heading,
                finding_index=pending_index,
                finding_text=text,
                finding_hash=digest,
                visibility_class=visibility_class(text),
                page_id=page_info.get("page_id"),
                page_title=page_info.get("page_title") or current_heading or None,
                page_path=page_path,
            )
        )

    return findings


def embedding_text(finding: Finding) -> str:
    parts = [
        f"Area: {finding.area}",
        f"Page: {finding.page_title or finding.report_heading or 'Unknown'}",
    ]
    if finding.page_path:
        parts.append(f"Docs path: {finding.page_path}")
    parts.append(f"Finding: {finding.finding_text}")
    return "\n".join(parts)

