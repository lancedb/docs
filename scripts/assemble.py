"""
Assemble the published documentation tree from one or more content roots.

The site is not published straight from this repository. It is assembled here
and pushed to the `assembled` branch, which Mintlify serves. Today there is one
root and the output is byte-identical to `docs/`; the value is the seam. Later
phases add roots — the open-source pages from `lancedb/lancedb`, then the
Enterprise overlays from `sophon` — by editing `assemble.yaml` rather than this
file.

Six stages:

    resolve   walk each root in order and map output path -> source file
    validate  anchors unique per page, nav references resolve, no path escapes
    merge     apply overlay fragments onto reference anchors      (no overlays yet)
    nav       assemble docs.json                                  (passthrough)
    emit      write the output tree
    check     `mint broken-links`, run separately in CI

Hard-fails on any inconsistency. A partially assembled site that renders is far
worse than a build that stops and says why.

Usage:

    python scripts/assemble.py                 assemble into the configured output
    python scripts/assemble.py --check-spec    verify the tracked OpenAPI spec matches its release
    python scripts/assemble.py --sync-spec     rewrite the tracked spec from its release
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "assemble.yaml"

# `## Heading {#anchor}` — the stable identity a section keeps across rewording
# and reordering, and the key Enterprise overlays will join on from A5.
ANCHOR_RE = re.compile(r"^#{1,6}\s+.*\{#([A-Za-z0-9][A-Za-z0-9._-]*)\}\s*$", re.M)
PAGE_SUFFIXES = (".mdx", ".md")


class AssembleError(Exception):
    """A condition that must stop the build rather than degrade the output."""


@dataclass(frozen=True)
class Root:
    name: str
    path: Path
    role: str  # "reference" owns structure; "overlay" contributes to it


@dataclass
class Config:
    output: Path
    roots: list[Root]
    openapi: dict[str, str] | None = None


@dataclass
class Resolved:
    """Output-relative path -> the root and file it came from."""

    files: dict[str, tuple[Root, Path]] = field(default_factory=dict)
    overlays: dict[str, tuple[Root, Path]] = field(default_factory=dict)


def load_config(path: Path = CONFIG_PATH) -> Config:
    raw = yaml.safe_load(path.read_text())
    roots = []
    for entry in raw["roots"]:
        role = entry.get("role", "reference")
        if role not in ("reference", "overlay"):
            raise AssembleError(f"root {entry['name']}: unknown role {role!r}")
        root_path = (REPO_ROOT / entry["path"]).resolve()
        if not root_path.is_dir():
            raise AssembleError(f"root {entry['name']}: {root_path} is not a directory")
        roots.append(Root(name=entry["name"], path=root_path, role=role))
    if not any(r.role == "reference" for r in roots):
        raise AssembleError("at least one reference root is required")
    return Config(
        output=(REPO_ROOT / raw["output"]).resolve(),
        roots=roots,
        openapi=raw.get("openapi"),
    )


# --------------------------------------------------------------------------- #
# stage 1: resolve
# --------------------------------------------------------------------------- #


def resolve(config: Config) -> Resolved:
    """Map every output path to the root that provides it.

    Two reference roots claiming the same path is an error rather than
    last-writer-wins: silently preferring one repository's copy of a page is
    exactly the drift this project exists to remove.
    """
    resolved = Resolved()
    for root in config.roots:
        for src in sorted(root.path.rglob("*")):
            if not src.is_file():
                continue
            rel = src.relative_to(root.path).as_posix()
            target = resolved.overlays if root.role == "overlay" else resolved.files
            if rel in target:
                other = target[rel][0]
                raise AssembleError(
                    f"{rel} is provided by both {other.name} and {root.name}; "
                    "reference roots must not overlap"
                )
            target[rel] = (root, src)
    return resolved


# --------------------------------------------------------------------------- #
# stage 2: validate
# --------------------------------------------------------------------------- #


def iter_pages(resolved: Resolved) -> Iterable[tuple[str, Path]]:
    for rel, (_root, src) in sorted(resolved.files.items()):
        if rel.endswith(PAGE_SUFFIXES):
            yield rel, src


def nav_page_paths(docs_json: dict) -> set[str]:
    """Page paths referenced anywhere in the navigation tree.

    Navigation also holds labels, icons and external links, so this collects a
    superset. Callers must only ever use it to test membership of a known page
    path, never to enumerate pages.

    `openapi` blocks are skipped: their `source` and `directory` name a spec file
    and the directory its endpoint pages are generated into, neither of which is
    an authored page.
    """
    found: set[str] = set()
    skip_keys = {"openapi", "href", "icon", "logo"}

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key not in skip_keys:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            found.add(node.lstrip("/"))

    walk(docs_json.get("navigation", {}))
    return found


def validate(config: Config, resolved: Resolved, docs_json: dict) -> list[str]:
    """Return warnings; raise on anything that must stop the build."""
    warnings: list[str] = []

    # Anchors must be unique within a page: an overlay keyed on a duplicated
    # anchor would have no single place to attach.
    for rel, src in iter_pages(resolved):
        anchors = ANCHOR_RE.findall(src.read_text(encoding="utf-8", errors="replace"))
        duplicates = {a for a in anchors if anchors.count(a) > 1}
        if duplicates:
            raise AssembleError(
                f"{rel}: duplicate anchors {sorted(duplicates)}"
            )

    # Every page the navigation names must exist, or the site ships dead entries.
    page_stems = {
        rel.rsplit(".", 1)[0] for rel, _ in iter_pages(resolved)
    }
    for referenced in nav_page_paths(docs_json):
        if referenced.startswith(("http://", "https://", "#")):
            continue
        if referenced in page_stems:
            continue
        # Labels and group names share the string space with page paths, so only
        # a path-shaped miss is worth reporting.
        if "/" in referenced and not referenced.endswith("/"):
            warnings.append(f"navigation references missing page: {referenced}")

    # An overlay for a page that no reference root provides would never render.
    for rel in resolved.overlays:
        if rel not in resolved.files:
            raise AssembleError(
                f"overlay {rel} has no reference page to attach to"
            )

    return warnings


# --------------------------------------------------------------------------- #
# stage 3: merge  (overlays land in A5)
# --------------------------------------------------------------------------- #


def merge(resolved: Resolved) -> None:
    """Apply overlay fragments onto reference anchors.

    A5 fills this in: each overlay contributes content keyed on an anchor in the
    reference page, rendered after that section inside an Enterprise banner. With
    no overlay roots configured there is nothing to merge, and `resolve` has
    already proven the set is empty.
    """
    if resolved.overlays:
        raise AssembleError(
            "overlay roots are configured but the merge stage is not implemented yet"
        )


# --------------------------------------------------------------------------- #
# stage 4: nav
# --------------------------------------------------------------------------- #


def assemble_nav(resolved: Resolved) -> tuple[dict, bytes | None]:
    """Produce the published docs.json.

    Returns the parsed navigation for validation, and the original bytes when
    nothing transformed it. Emitting those bytes verbatim keeps the assembled
    tree *literally* byte-identical to its source rather than merely equivalent,
    which is worth more than tidy formatting while the assembler is meant to be
    a no-op. Re-serializing loses that for no gain: it rewrote a literal em-dash
    as `\u2014` and nothing else.

    A5 merges overlay nav fragments by page path; A6 wraps the result in a
    `navigation.versions` array, mounting the newest bundle twice — unprefixed so
    existing URLs survive, and under its version path so it stays addressable.
    Both genuinely change the navigation, and both will serialize.
    """
    entry = resolved.files.get("docs.json")
    if entry is None:
        raise AssembleError("no docs.json found in any reference root")
    raw = entry[1].read_bytes()
    return json.loads(raw.decode("utf-8")), raw


# --------------------------------------------------------------------------- #
# stage 5: emit
# --------------------------------------------------------------------------- #


def emit(
    config: Config, resolved: Resolved, docs_json: dict, nav_raw: bytes | None
) -> int:
    output = config.output
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    written = 0
    for rel, (_root, src) in sorted(resolved.files.items()):
        if rel == "docs.json":
            continue
        dest = (output / rel).resolve()
        # A root containing `../` in a name would otherwise write outside the
        # output tree.
        if not dest.is_relative_to(output):
            raise AssembleError(f"{rel} resolves outside the output directory")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        written += 1

    if nav_raw is not None:
        (output / "docs.json").write_bytes(nav_raw)
    else:
        (output / "docs.json").write_text(
            json.dumps(docs_json, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return written + 1


# --------------------------------------------------------------------------- #
# OpenAPI spec, tracked at a release
# --------------------------------------------------------------------------- #


def released_spec(openapi: dict[str, str]) -> bytes:
    url = (
        f"https://raw.githubusercontent.com/{openapi['repo']}/"
        f"{openapi['release']}/{openapi['source']}"
    )
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()
    except urllib.error.URLError as exc:
        raise AssembleError(f"could not fetch {url}: {exc}") from exc


def spec_paths(config: Config) -> Path:
    if not config.openapi:
        raise AssembleError("no openapi section in assemble.yaml")
    return REPO_ROOT / config.openapi["dest"]


def sync_spec(config: Config) -> bool:
    """Write the released spec into the tree. Returns True if it changed."""
    dest = spec_paths(config)
    released = released_spec(config.openapi)
    current = dest.read_bytes() if dest.exists() else b""
    if current == released:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(released)
    return True


def check_spec(config: Config) -> None:
    """Fail if the tracked spec has drifted from its release.

    Keeping this a check rather than a build-time fetch means assembly is
    deterministic and needs no network, while drift is still impossible to miss.
    """
    dest = spec_paths(config)
    if not dest.exists():
        raise AssembleError(f"{config.openapi['dest']} is missing; run --sync-spec")
    if dest.read_bytes() != released_spec(config.openapi):
        raise AssembleError(
            f"{config.openapi['dest']} differs from "
            f"{config.openapi['repo']}@{config.openapi['release']}; run --sync-spec"
        )


# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--config", type=Path, default=CONFIG_PATH, help="path to assemble.yaml"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check-spec",
        action="store_true",
        help="verify the tracked OpenAPI spec matches its release, then exit",
    )
    mode.add_argument(
        "--sync-spec",
        action="store_true",
        help="rewrite the tracked OpenAPI spec from its release, then exit",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)

        if args.check_spec:
            check_spec(config)
            print(
                f"openapi spec matches {config.openapi['repo']}@{config.openapi['release']}"
            )
            return 0

        if args.sync_spec:
            changed = sync_spec(config)
            tag = f"{config.openapi['repo']}@{config.openapi['release']}"
            print(f"openapi spec {'updated from' if changed else 'already matches'} {tag}")
            return 0

        resolved = resolve(config)
        docs_json, nav_raw = assemble_nav(resolved)
        warnings = validate(config, resolved, docs_json)
        merge(resolved)
        count = emit(config, resolved, docs_json, nav_raw)
    except AssembleError as exc:
        print(f"assemble: {exc}", file=sys.stderr)
        return 1

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    roots = ", ".join(f"{r.name}({r.role})" for r in config.roots)
    print(f"assembled {count} files from {roots} into {config.output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
