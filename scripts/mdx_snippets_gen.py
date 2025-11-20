"""
Generate MDX code snippets from annotated source directories.

Markers use the same syntax across languages, with the line comment
prefix varying by language:

  Python:   # --8<-- [start:snippet_name] ... # --8<-- [end:snippet_name]
  TS/RS:    // --8<-- [start:snippet_name] ... // --8<-- [end:snippet_name]

This script scans all source files in a given directory and outputs
MDX files under `docs/snippets/<lang>/<snippet_name>.mdx`.

Examples:
  python scripts/mdx_snippets_gen.py --source-dir examples/py
  python scripts/mdx_snippets_gen.py --source-dir examples/ts
  python scripts/mdx_snippets_gen.py --source-dir examples/rs
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def build_markers(comment_markers: Iterable[str]) -> Tuple[re.Pattern[str], re.Pattern[str]]:
    """Build start/end regex patterns for the given comment markers.

    comment_markers: e.g., ["#"] for Python or ["//"] for TS/RS.
    """
    escaped = "|".join(re.escape(c) for c in comment_markers)
    start_re = re.compile(rf"^\s*(?:{escaped})\s*--8<--\s*\[start:([^\]]+)\]")
    end_re = re.compile(rf"^\s*(?:{escaped})\s*--8<--\s*\[end:([^\]]+)\]")
    return start_re, end_re


def parse_snippets(
    lines: List[str], start_re: re.Pattern[str], end_re: re.Pattern[str]
) -> Dict[str, List[str]]:
    """Extract snippets from lines using start/end markers.

    Returns a mapping of snippet_name -> list of lines (content only).
    """
    snippets: Dict[str, List[str]] = {}
    stack: List[Tuple[str, int]] = []

    for idx, line in enumerate(lines):
        m_start = start_re.match(line)
        if m_start:
            name = m_start.group(1).strip()
            stack.append((name, idx))
            continue

        m_end = end_re.match(line)
        if m_end:
            name = m_end.group(1).strip()
            if not stack:
                raise ValueError(
                    f"End marker for '{name}' found at line {idx + 1} without matching start"
                )
            start_name, start_idx = stack.pop()
            if start_name != name:
                raise ValueError(
                    f"Mismatched markers: start '{start_name}' at line {start_idx + 1}, "
                    f"end '{name}' at line {idx + 1}"
                )
            # Content is between start and end, exclusive of the marker lines
            content = lines[start_idx + 1 : idx]
            snippets[name] = content
            continue

    if stack:
        open_names = ", ".join(f"{n}@{i + 1}" for n, i in stack)
        raise ValueError(f"Unclosed snippet markers for: {open_names}")

    return snippets


def dedent(lines: List[str]) -> List[str]:
    """Dedent a block of code while preserving relative indentation and blank lines."""
    # Compute minimal indentation across non-empty lines
    min_indent = None
    for ln in lines:
        if ln.strip() == "":
            continue
        # Count leading spaces after expanding tabs
        expanded = ln.expandtabs(4)
        indent = len(expanded) - len(expanded.lstrip(" "))
        if min_indent is None or indent < min_indent:
            min_indent = indent

    if not lines:
        return []
    if min_indent in (None, 0):
        return [ln.rstrip("\n") for ln in lines]

    dedented: List[str] = []
    for ln in lines:
        expanded = ln.expandtabs(4)
        if expanded.strip() == "":
            dedented.append(expanded.rstrip("\n"))
        else:
            dedented.append(expanded[min_indent:].rstrip("\n"))
    return dedented


def write_mdx_snippet(
    out_dir: Path, fence_lang: str, name: str, content_lines: List[str], source_rel: str
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.mdx"

    header = f"{{/* Auto-generated from {source_rel}::{name}. Do not edit manually. */}}\n"
    code = "\n".join(content_lines).rstrip() + "\n"
    mdx_body = f"```{fence_lang} {fence_lang}\n{code}```\n"

    new_content = header + mdx_body

    # Only write if changed
    if out_path.exists():
        try:
            existing = out_path.read_text(encoding="utf-8")
            if existing == new_content:
                return
        except Exception:
            pass

    out_path.write_text(new_content, encoding="utf-8")


def write_mdx_snippet_with_labels(
    out_dir: Path, fence_lang: str, name: str, content_lines: List[str], source_rel: str
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.mdx"

    header = f"{{/* Auto-generated from {source_rel}::{name}. Do not edit manually. */}}\n"
    code = "\n".join(content_lines).rstrip() + "\n"
    # Use nicer tab labels in Mintlify CodeGroup
    label_map = {
        "py": "Python",
        "python": "Python",
        "ts": "TypeScript",
        "tsx": "TypeScript",
        "js": "JavaScript",
        "jsx": "JavaScript",
        "rs": "Rust",
    }
    display_label = label_map.get(fence_lang, fence_lang)

    # First token: syntax highlight; second: tab label
    mdx_body = f"```{fence_lang} {display_label} icon={display_label}\n{code}```\n"

    new_content = header + mdx_body

    # Only write if changed
    if out_path.exists():
        try:
            existing = out_path.read_text(encoding="utf-8")
            if existing == new_content:
                return
        except Exception:
            pass

    out_path.write_text(new_content, encoding="utf-8")


def discover_language_from_dir(source_dir: Path) -> str:
    """Infer language key (py|ts|rs) from directory name or file extensions."""
    name = source_dir.name.lower()
    if name in {"py", "python"}:
        return "py"
    if name in {"ts", "tsx", "typescript"}:
        return "ts"
    if name in {"rs", "rust"}:
        return "rs"

    # Fallback: detect by file extensions present
    has_py = any(source_dir.rglob("*.py"))
    has_ts = any(source_dir.rglob("*.ts")) or any(source_dir.rglob("*.tsx"))
    has_rs = any(source_dir.rglob("*.rs"))
    candidates = [(has_py, "py"), (has_ts, "ts"), (has_rs, "rs")]
    detected = [lang for flag, lang in candidates if flag]
    if len(detected) == 1:
        return detected[0]
    raise ValueError(
        f"Unable to determine language for directory: {source_dir}. "
        "Please name the folder py/ts/rs or ensure only one language is present."
    )


def file_globs_for_lang(lang: str) -> List[str]:
    if lang == "py":
        return ["*.py"]
    if lang == "ts":
        return ["*.ts", "*.tsx"]
    if lang == "rs":
        return ["*.rs"]
    raise ValueError(f"Unsupported language: {lang}")


def markers_for_lang(lang: str) -> Tuple[re.Pattern[str], re.Pattern[str]]:
    if lang == "py":
        return build_markers(["#"])  # Python line comment
    if lang in {"ts", "rs"}:
        return build_markers(["//"])  # TS/Rust line comment
    raise ValueError(f"Unsupported language for markers: {lang}")


def fence_lang_for_lang(lang: str) -> str:
    # Normalize fence language id for syntax highlighting + labels
    if lang == "py":
        return "py"
    if lang == "ts":
        return "ts"
    if lang == "rs":
        return "rs"
    return lang


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate MDX snippets from annotated source directory"
    )
    parser.add_argument(
        "--source_dir", "-s", required=True, help="Path to the source directory (e.g., examples/py)"
    )
    parser.add_argument(
        "--output_path",
        "-o",
        default=str(Path("docs") / "snippets"),
        help="Output directory for MDX code snippets (default: docs/snippets)",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.exists() or not source_dir.is_dir():
        parser.error(f"Source directory not found or not a directory: {source_dir}")

    lang = discover_language_from_dir(source_dir)
    fence_lang = fence_lang_for_lang(lang)
    out_dir = Path(args.output_path) / lang

    start_re, end_re = markers_for_lang(lang)

    # Collect files
    file_patterns = file_globs_for_lang(lang)
    files: List[Path] = []
    for pat in file_patterns:
        files.extend(source_dir.rglob(pat))
    # Deduplicate and sort
    files = sorted(set(files))

    if not files:
        print(f"No source files found under {source_dir} for language '{lang}'")
        return 0

    # Aggregate snippets, detect duplicates across files
    all_snippets: Dict[str, Tuple[Path, List[str]]] = {}

    for file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception:
            # Skip unreadable files
            continue
        lines = text.splitlines(keepends=True)
        try:
            raw = parse_snippets(lines, start_re, end_re)
        except ValueError as e:
            raise ValueError(f"Error parsing {file_path}: {e}") from e

        for name, content in raw.items():
            if name in all_snippets:
                prev_file, prev_content = all_snippets[name]
                if "".join(prev_content) != "".join(content):
                    raise ValueError(
                        f"Duplicate snippet name '{name}' with different content: {prev_file} and {file_path}"
                    )
                # If identical, keep the first occurrence
                continue
            all_snippets[name] = (file_path, content)

    # Write snippets
    written = 0
    for name, (src_path, content) in all_snippets.items():
        dedented = dedent(content)
        src_rel = os.path.relpath(src_path, Path.cwd())
        write_mdx_snippet_with_labels(out_dir, fence_lang, name, dedented, src_rel)
        written += 1

    print(f"Generated {written} MDX snippets in {out_dir} from {len(files)} files")
    return 0


if __name__ == "__main__":
    main()
