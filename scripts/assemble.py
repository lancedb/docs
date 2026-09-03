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
import os
import shutil
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
# The reference root ships a complete `docs.json` so it can be served on its own.
# Every other root contributes a `docs.nav.json` fragment instead: tabs merged by
# name into the base. Neither file is content, so neither is copied to the output.
NAV_BASE = "docs.json"
NAV_FRAGMENT = "docs.nav.json"


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
    fragments: dict[str, Path] = field(default_factory=dict)


ENV_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def expand(value: str) -> str:
    """Resolve ${VAR} and ${VAR:-default} in a configured path.

    Root paths point at sibling checkouts, which sit somewhere different in CI
    than on a laptop. Everything else in the config stays literal.
    """
    return ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)


def load_config(path: Path = CONFIG_PATH) -> Config:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or "roots" not in raw or "output" not in raw:
        raise AssembleError(f"{path} must define `output` and `roots`")
    roots = []
    for entry in raw["roots"]:
        role = entry.get("role", "reference")
        if role not in ("reference", "overlay"):
            raise AssembleError(f"root {entry['name']}: unknown role {role!r}")
        root_path = (REPO_ROOT / expand(entry["path"])).resolve()
        if not root_path.is_dir():
            raise AssembleError(f"root {entry['name']}: {root_path} is not a directory")
        roots.append(Root(name=entry["name"], path=root_path, role=role))
    if not any(r.role == "reference" for r in roots):
        raise AssembleError("at least one reference root is required")
    return Config(
        output=(REPO_ROOT / expand(raw["output"])).resolve(),
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
            if rel == NAV_FRAGMENT:
                resolved.fragments[root.name] = src
                continue
            target = resolved.overlays if root.role == "overlay" else resolved.files
            if rel in target:
                other = target[rel][0]
                raise AssembleError(
                    f"{rel} is provided by both {other.name} and {root.name}; "
                    f"{root.role} roots must not overlap"
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

    Strings are classified by the key that holds them rather than by shape. An
    earlier version only trusted strings containing a slash, which silently
    exempted every top-level page — `quickstart`, `index` — from validation.
    Naming the label-bearing and non-page keys instead makes the check complete:
    against the current navigation it finds 179 page paths and 46 labels with no
    misclassification either way.
    """
    found: set[str] = set()
    # Keys whose values name something other than a page.
    non_page_keys = {"openapi", "href", "icon", "logo"}
    # Keys whose values are human-readable labels.
    label_keys = {
        "group", "tab", "dropdown", "anchor", "language", "version",
        "tag", "name", "title",
    }

    def walk(node: object, key: str | None = None) -> None:
        if isinstance(node, dict):
            for child_key, value in node.items():
                if child_key not in non_page_keys:
                    walk(value, child_key)
        elif isinstance(node, list):
            for value in node:
                walk(value, key)
        elif isinstance(node, str) and key not in label_keys:
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

    # Every page the navigation names must exist. This fails the build rather
    # than warning: a navigation entry pointing at nothing is a dead link in the
    # published sidebar, and the whole point of assembling is to catch that
    # before it ships.
    page_stems = {rel.rsplit(".", 1)[0] for rel, _ in iter_pages(resolved)}
    missing = sorted(
        ref
        for ref in nav_page_paths(docs_json)
        if not ref.startswith(("http://", "https://", "#"))
        and ref not in page_stems
    )
    if missing:
        raise AssembleError(
            "navigation references pages that do not exist: " + ", ".join(missing)
        )

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


def merge_nav(base: dict, fragment: dict) -> dict:
    """Fold a root's navigation fragment into the base navigation.

    Tabs are matched by name: a fragment tab that already exists contributes its
    groups to it, and a new tab is inserted. Both carry an optional `after`
    naming the sibling they follow, because appending is not good enough --
    sidebar order is what a reader navigates by, and dropping Geneva below
    Support or Datasets past Use Cases silently reorders the whole site.

    This is the same shape sophon's Enterprise fragment will use in A5.
    """

    def descend(node: dict | list, path: list[str]) -> list:
        """Follow a list of group names to the container that should hold an entry."""
        items = node["navigation"]["tabs"] if isinstance(node, dict) else node
        for name in path:
            match = next(
                (
                    i
                    for i in items
                    # Page paths sit alongside groups in the same list.
                    if isinstance(i, dict)
                    and (i.get("tab") == name or i.get("group") == name)
                ),
                None,
            )
            if match is None:
                raise AssembleError(
                    f"navigation fragment targets {name!r}, which does not exist"
                )
            items = match.setdefault("groups" if "groups" in match else "pages", [])
        return items

    def insert_value(items: list, value: str, after: str | None) -> None:
        """Insert a bare page path after a named sibling."""
        if after is None:
            items.insert(0, value)
            return
        for index, item in enumerate(items):
            name = item.get("tab") or item.get("group") if isinstance(item, dict) else item
            if name == after:
                items.insert(index + 1, value)
                return
        items.append(value)

    def insert(items: list, entry: dict, key: str) -> None:
        after = entry.pop("after", None)
        if after is None:
            items.append(entry)
            return
        for index, item in enumerate(items):
            # A container list holds tabs, groups and bare page paths, so match
            # on whichever names the entry rather than on one fixed key.
            name = (
                (item.get("tab") or item.get("group"))
                if isinstance(item, dict)
                else item
            )
            if name == after:
                items.insert(index + 1, entry)
                return
        raise AssembleError(
            f"navigation fragment wants to follow {after!r}, which does not exist"
        )

    # Entries that name a nested container, e.g. the Enterprise group inside
    # "Get started". A5's Enterprise fragment uses the same mechanism.
    for spec in fragment.get("insert", []):
        target = descend(base, spec["into"])
        entry = spec["entry"]
        if isinstance(entry, str):
            insert_value(target, entry, spec.get("after"))
        else:
            insert(target, dict(entry, after=spec.get("after")), "group")

    # Keys lifted out of the base because the file they name lives here.
    for spec in fragment.get("set", []):
        container = descend(base, spec["into"][:-1] or [])
        name = spec["into"][-1]
        target = next(
            (
                i
                for i in container
                if isinstance(i, dict)
                and (i.get("tab") == name or i.get("group") == name)
            ),
            None,
        )
        if target is None:
            raise AssembleError(f"navigation fragment targets {name!r}, which does not exist")
        target[spec["key"]] = spec["value"]

    tabs = base.setdefault("navigation", {}).setdefault("tabs", [])
    by_name = {t.get("tab"): t for t in tabs}
    for incoming in fragment.get("tabs", []):
        existing = by_name.get(incoming.get("tab"))
        if existing is None:
            insert(tabs, incoming, "tab")
            by_name[incoming.get("tab")] = incoming
            continue
        for key, value in incoming.items():
            if key in ("tab", "after"):
                continue
            if key == "groups":
                for group in value:
                    insert(existing.setdefault("groups", []), group, "group")
            else:
                # `openapi` and friends: the root that owns the generated pages
                # owns how they are generated.
                existing[key] = value
    if fragment.get("redirects"):
        base["redirects"] = fragment["redirects"]
    return base


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
    entry = resolved.files.get(NAV_BASE)
    if entry is None:
        raise AssembleError(f"no {NAV_BASE} found in any reference root")
    raw = entry[1].read_bytes()
    base = json.loads(raw.decode("utf-8"))

    fragments = sorted(resolved.fragments.items())
    if not fragments:
        # Nothing to merge: emit the source bytes so the tree stays literally
        # byte-identical rather than merely equivalent.
        return base, raw
    for _name, path in fragments:
        base = merge_nav(base, json.loads(path.read_text(encoding="utf-8")))
    return base, None


# --------------------------------------------------------------------------- #
# stage 5: emit
# --------------------------------------------------------------------------- #


def check_output_safe(config: Config) -> None:
    """Refuse to emit into, or over, a source root.

    `emit` clears the output directory before writing. If the output overlapped a
    root, that would delete the source and then fail copying the files it had
    just removed — verified: it destroys the tree. The paths do not overlap in
    the current configuration, so this guards the edit that adds a root rather
    than today's setup.
    """
    output = config.output
    for root in config.roots:
        if output == root.path:
            raise AssembleError(
                f"output {output} is the {root.name} root; assembling would delete it"
            )
        if output.is_relative_to(root.path):
            raise AssembleError(
                f"output {output} is inside the {root.name} root; "
                "assembling would delete part of the source"
            )
        if root.path.is_relative_to(output):
            raise AssembleError(
                f"the {root.name} root {root.path} is inside output {output}; "
                "assembling would delete it"
            )


def emit(
    config: Config, resolved: Resolved, docs_json: dict, nav_raw: bytes | None
) -> int:
    check_output_safe(config)
    output = config.output
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    written = 0
    for rel, (_root, src) in sorted(resolved.files.items()):
        if rel == NAV_BASE:
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
        (output / NAV_BASE).write_bytes(nav_raw)
    else:
        # `ensure_ascii=True` matches how the navigation was written before the
        # assembler existed. It is not cosmetic: the banner text carries an
        # em-dash, and escaping it differently changes the bytes Mintlify hashes
        # its CSS and JS bundles from, which renames those assets on every page.
        (output / NAV_BASE).write_text(
            json.dumps(docs_json, indent=2) + "\n", encoding="utf-8"
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
    dest = (REPO_ROOT / config.openapi["dest"]).resolve()
    # Same containment rule the output path gets: a `dest` of `../..` would
    # otherwise write outside the repository.
    if not dest.is_relative_to(REPO_ROOT):
        raise AssembleError(f"openapi dest {dest} is outside the repository")
    return dest


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
    try:
        where = config.output.relative_to(REPO_ROOT)
    except ValueError:
        where = config.output
    print(f"assembled {count} files from {roots} into {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
