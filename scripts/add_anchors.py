"""
Give every section heading a stable anchor.

An anchor is the identity a section keeps when it is reworded or moved, and the
key Enterprise overlays attach to from A5: an overlay says "put this after
`{#branch-create}`" and must still land correctly after someone rewrites the
heading above it. Heading-derived slugs cannot do that — they change with the
text — so the anchor is written down once and then never regenerated.

That "never regenerated" is the whole point, and it shapes this script:

  * headings that already carry an anchor are left exactly as they are, so
    re-running is safe and an anchor edited by hand survives;
  * names come from the ids the site *already renders*, not from a slug rule of
    our own. That is what keeps every existing deep link working: a reader's
    bookmark to `#what's-next` must still resolve afterwards. Deriving the name
    independently looked equivalent and was not — Mintlify keeps a curly
    apostrophe in the id where a naive slug turns it into a hyphen, so
    `tables/index` would have silently changed its anchor.

    Names are still readable, because Mintlify derives them from the heading
    text too. They are simply frozen at today's value rather than recomputed:
    a later reworded heading keeps the original anchor, and that divergence is
    the point, not drift.

Fenced code blocks are skipped: a `## comment` inside a shell example is not a
heading.

The rendered ids come from a `mint export` bundle, so run one first:

    cd docs && mint export --output /tmp/site.zip

Usage:

    python scripts/add_anchors.py --export /tmp/site.zip docs/tables
    python scripts/add_anchors.py --export /tmp/site.zip --check docs/tables
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import zipfile
from pathlib import Path

# `## Heading`, capturing any existing `{#anchor}` so it can be preserved.
HEADING_RE = re.compile(
    r"^(?P<hashes>#{2,6})\s+(?P<text>.+?)(?:\s+\{#(?P<anchor>[^}]+)\})?\s*$"
)
FENCE_RE = re.compile(r"^\s*```")
# `### 1. Setup` renders with its number until an explicit `{#anchor}` is added,
# at which point Mintlify re-parses the text and treats the number as an ordered
# list marker, silently dropping it from the heading and the table of contents.
# 117 headings across 17 pages start this way. Escaping the period keeps the
# rendered text and the id identical; the backslash does not reach the output.
LEADING_NUMBER_RE = re.compile(r"^(\d+)\. (?=\S)")
FRONTMATTER_DELIM = "---"

# Inline markup to strip before deriving a name, so `## Use \`add_columns()\``
# becomes `use-add-columns` rather than carrying backticks into the anchor.
INLINE_CODE_RE = re.compile(r"`([^`]*)`")
LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
JSX_RE = re.compile(r"<[^>]+>")
NON_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    text = LINK_RE.sub(r"\1", text)
    text = INLINE_CODE_RE.sub(r"\1", text)
    text = JSX_RE.sub(" ", text)
    text = NON_SLUG_RE.sub("-", text.lower()).strip("-")
    return text or "section"


HEADING_ID_RE = re.compile(r'<h([2-6])[^>]*\bid="([^"]+)"')


def rendered_ids(export: Path) -> dict[str, list[str]]:
    """Map page path -> heading ids, in document order, from a mint export.

    Mintlify emits an id on every heading. Reusing those ids as anchors is what
    makes the pass invisible to readers and safe for existing links.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        if export.is_dir():
            root = export
        else:
            with zipfile.ZipFile(export) as zf:
                zf.extractall(root)
        ids: dict[str, list[str]] = {}
        for page in root.rglob("index.html"):
            rel = page.parent.relative_to(root).as_posix()
            # The site root renders to index.html at the top level; everything
            # else keeps its full path, including a page literally named index.
            if rel == ".":
                rel = "index"
            found = [
                anchor
                for _level, anchor in HEADING_ID_RE.findall(
                    page.read_text(encoding="utf-8", errors="replace")
                )
                # React-generated ids are per-build noise, not heading slugs.
                if not anchor.startswith("_R_")
            ]
            ids[rel] = found
        return ids


def anchor_file(
    path: Path, docs_root: Path, ids: dict[str, list[str]], apply: bool
) -> tuple[int, int, list[str]]:
    """Return (added, existing, sample) for one page."""
    page = str(path.relative_to(docs_root)).removesuffix(".mdx")
    available = list(ids.get(page, []))

    lines = path.read_text(encoding="utf-8").split("\n")
    out: list[str] = []
    used: set[str] = set()
    added = existing = 0
    sample: list[str] = []
    consumed = 0

    in_fence = False
    in_frontmatter = False
    for index, line in enumerate(lines):
        # Frontmatter opens on the very first line and is not content.
        if index == 0 and line.strip() == FRONTMATTER_DELIM:
            in_frontmatter = True
            out.append(line)
            continue
        if in_frontmatter:
            if line.strip() == FRONTMATTER_DELIM:
                in_frontmatter = False
            out.append(line)
            continue
        if FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue

        match = HEADING_RE.match(line)
        if not match:
            out.append(line)
            continue

        if match.group("anchor"):
            used.add(match.group("anchor"))
            existing += 1
            out.append(line)
            continue

        # Take the id the site already renders for this heading. Falling back to
        # a derived slug would reintroduce exactly the divergence this avoids, so
        # a missing id is an error rather than a guess.
        if consumed >= len(available):
            raise SystemExit(
                f"{path}: more headings in source than rendered ids "
                f"({consumed + 1} > {len(available)}); re-run mint export"
            )
        anchor = available[consumed]
        consumed += 1
        if anchor in used:
            raise SystemExit(f"{path}: rendered id {anchor!r} appears twice")
        used.add(anchor)
        added += 1
        if len(sample) < 3:
            sample.append(f"{match.group('text')[:44]}  ->  {{#{anchor}}}")
        text = LEADING_NUMBER_RE.sub(r"\1\\. ", match.group("text"))
        out.append(f"{match.group('hashes')} {text} {{#{anchor}}}")

    if apply and added:
        path.write_text("\n".join(out), encoding="utf-8")
    return added, existing, sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("paths", nargs="+", type=Path, help="files or directories")
    parser.add_argument(
        "--export",
        type=Path,
        required=True,
        help="mint export zip or directory supplying the rendered heading ids",
    )
    parser.add_argument(
        "--docs-root", type=Path, default=Path("docs"), help="docs root for page paths"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report headings without an anchor and exit non-zero if any remain",
    )
    args = parser.parse_args()

    targets: list[Path] = []
    for path in args.paths:
        targets.extend(sorted(path.rglob("*.mdx")) if path.is_dir() else [path])

    ids = rendered_ids(args.export)
    total_added = total_existing = 0
    for target in targets:
        added, existing, sample = anchor_file(
            target, args.docs_root, ids, apply=not args.check
        )
        total_added += added
        total_existing += existing
        if added:
            verb = "missing" if args.check else "anchored"
            print(f"{target}: {verb} {added}, already anchored {existing}")
            for line in sample:
                print(f"    {line}")

    if args.check:
        print(f"\n{total_added} headings without an anchor, {total_existing} with one")
        return 1 if total_added else 0
    print(f"\nanchored {total_added} headings, left {total_existing} unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
