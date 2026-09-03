"""
Split one navigation into a base owned by the reference root and a fragment
contributed by this repository.

The published navigation is authored once, in the root that owns the pages. When
a page lives somewhere else, its entry has to be contributed by that root instead
— and put back in the same place, because sidebar order is what a reader
navigates by.

This walks the original navigation and, for every entry naming a page this
repository still holds, records where it sat: the chain of group names above it
and the sibling it followed. The assembler replays those, so the merged
navigation is identical to the one it replaced.

Run once when a set of pages moves between roots:

    python scripts/split_nav.py --nav <original docs.json> --moved <dir of moved pages>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def split(nav: dict, owned: set[str], owned_files: set[str]) -> tuple[dict, list[dict], list[dict]]:
    """Return (base navigation, insertions, settings) — `owned` is what the base keeps."""
    inserts: list[dict] = []
    settings: list[dict] = []

    def walk(items: list, path: list[str]) -> list:
        kept: list = []
        for item in items:
            if isinstance(item, str):
                if item.lstrip("/") in owned:
                    kept.append(item)
                else:
                    inserts.append(
                        {"into": list(path), "after": kept[-1] if kept else None,
                         "entry": item}
                    )
                continue
            name = item.get("tab") or item.get("group")
            # An `openapi` block names a spec file. If the base root does not
            # hold that file, Mintlify refuses to build at all — so the key is
            # lifted out and restored by whichever root does own the spec.
            spec = (item.get("openapi") or {}).get("source") if isinstance(item.get("openapi"), dict) else None
            if spec and spec.lstrip("/").removesuffix(".yml") not in owned_files:
                settings.append(
                    {"into": path + [name], "key": "openapi", "value": item["openapi"]}
                )
                item = {k: v for k, v in item.items() if k != "openapi"}
            child_key = "groups" if "groups" in item else "pages"
            # Remember where this container's own insertions start: if it turns
            # out to move wholesale, they are redundant and must be dropped, or
            # they would try to fill a container that does not exist yet.
            mark = len(inserts)
            children = walk(item.get(child_key, []), path + [name])
            if children or child_key not in item:
                kept.append({**item, child_key: children} if child_key in item else item)
            else:
                del inserts[mark:]
                # Nothing left in this container, so the whole thing belongs to
                # the other root — carried across intact rather than rebuilt.
                previous = kept[-1] if kept else None
                after = (
                    (previous.get("tab") or previous.get("group"))
                    if isinstance(previous, dict)
                    else previous
                )
                inserts.append({"into": list(path), "after": after, "entry": item})
        return kept

    base = {**nav, "tabs": walk(nav["tabs"], [])}
    return base, inserts, settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--nav", type=Path, required=True)
    parser.add_argument("--owned", type=Path, required=True,
                        help="directory whose .mdx files the base root owns")
    parser.add_argument("--base-out", type=Path, required=True)
    parser.add_argument("--fragment-out", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.nav.read_text())
    owned = {
        str(p.relative_to(args.owned)).removesuffix(".mdx")
        for p in args.owned.rglob("*.mdx")
    }
    owned_files = {
        str(p.relative_to(args.owned)).removesuffix(".yml")
        for p in args.owned.rglob("*.yml")
    }
    base_nav, inserts, settings = split(source["navigation"], owned, owned_files)

    # Key order is preserved, not rebuilt: the assembled file is compared byte
    # for byte against the one it replaces, and Mintlify hashes its bundles from
    # these bytes, so a reordered key renames every CSS and JS asset.
    base = {}
    for key, value in source.items():
        if key == "navigation":
            base[key] = base_nav
        elif key == "redirects":
            continue  # contributed by the fragment, which owns the full list
        else:
            base[key] = value
    args.base_out.write_text(json.dumps(base, indent=2, ensure_ascii=False) + "\n")

    fragment = {
        "insert": inserts,
        "set": settings,
        "redirects": source.get("redirects", []),
    }
    args.fragment_out.write_text(json.dumps(fragment, indent=2, ensure_ascii=False) + "\n")

    print(
        f"base keeps {len(owned)} pages; fragment contributes {len(inserts)} entries "
        f"and {len(settings)} settings"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
