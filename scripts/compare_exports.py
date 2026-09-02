"""Compare two `mint export` bundles — the byte-comparability oracle for A1.

`mint export` is deterministic except for the OpenAPI-generated pages under
`api-reference/rest/`, which embed a freshly generated UUID per response code
(React keys for the playground tabs). Two exports of an unchanged tree differ in
exactly those tokens and nowhere else.

This normalizes UUIDs before comparing, but still compares the *count* of UUIDs
per file, so a page that gains or loses one is reported rather than masked.

    python3 compare_exports.py BASELINE NEW

BASELINE and NEW may each be a .zip produced by `mint export` or an already
extracted directory. Exit status is 0 when the bundles are equivalent.
"""

from __future__ import annotations

import re
import sys
import tempfile
import zipfile
from pathlib import Path

UUID_RE = re.compile(
    rb"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    rb"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

# Pages that fail to generate request examples are not reproducible: the failure
# renders "A valid request URL is required to generate request examples" in place
# of the code samples, and *which* pages fail varies per run. That was an upstream
# spec defect -- every `servers` entry was templated -- fixed in
# lance-format/lance-namespace@3b167e4c.
#
# Quarantine is therefore keyed on the placeholder itself, not on the path. It
# retires automatically: once the tree carries the fixed spec no page contains the
# placeholder, nothing is quarantined, and the whole reference is compared like any
# other page. Verified against exports built from the fixed spec -- all 54 REST
# pages that differ between runs differ *only* by per-build UUIDs, with zero
# content differences.
UNSTABLE_PREFIX = "api-reference/rest/"
PLACEHOLDER = b"A valid request URL is required"
# Observed jitter on identical input: 43-48 placeholders across runs. A hard
# gate on this number would fail at random, so it is reported always and warned
# on only outside the jitter band -- never allowed to decide the verdict.
PLACEHOLDER_JITTER = 6


def materialize(arg: str, tmp: Path, label: str) -> Path:
    path = Path(arg).expanduser()
    if path.is_dir():
        return path
    if not path.is_file():
        sys.exit(f"not found: {path}")
    dest = tmp / label
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(path) as zf:
            zf.extractall(dest)
    except zipfile.BadZipFile:
        sys.exit(f"not a mint export archive: {path}")
    return dest


def normalize(raw: bytes) -> tuple[bytes, int]:
    """Return (uuid-normalized bytes, number of UUIDs replaced)."""
    count = len(UUID_RE.findall(raw))
    return UUID_RE.sub(b"<UUID>", raw), count


def relmap(root: Path) -> dict[str, Path]:
    return {
        p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()
    }


def main() -> int:
    if len(sys.argv) != 3:
        sys.exit(__doc__)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        base = materialize(sys.argv[1], tmp, "baseline")
        new = materialize(sys.argv[2], tmp, "new")

        a, b = relmap(base), relmap(new)
        only_base = sorted(set(a) - set(b))
        only_new = sorted(set(b) - set(a))
        shared = sorted(set(a) & set(b))

        real_diffs: list[str] = []
        uuid_only: list[str] = []
        quarantined: list[str] = []

        for rel in shared:
            ra, rb = a[rel].read_bytes(), b[rel].read_bytes()
            if ra == rb:
                continue
            # UUID normalization is scoped to the generated reference, whose React
            # keys are regenerated per build. Applying it to authored pages would
            # silently accept a genuine UUID-only edit.
            if not rel.startswith(UNSTABLE_PREFIX):
                real_diffs.append(f"{rel}  (content)")
                continue
            na, ca = normalize(ra)
            nb, cb = normalize(rb)
            if na == nb and ca == cb:
                uuid_only.append(rel)
            elif PLACEHOLDER in ra or PLACEHOLDER in rb:
                # Example generation failed on at least one side; not reproducible.
                quarantined.append(rel)
            else:
                why = "content" if na != nb else f"uuid count {ca} vs {cb}"
                real_diffs.append(f"{rel}  ({why})")

        def placeholders(m: dict[str, Path]) -> int:
            return sum(
                1
                for rel, p in m.items()
                if rel.startswith(UNSTABLE_PREFIX) and PLACEHOLDER in p.read_bytes()
            )

        ph_a, ph_b = placeholders(a), placeholders(b)

        print(f"baseline : {sys.argv[1]}  ({len(a)} files)")
        print(f"new      : {sys.argv[2]}  ({len(b)} files)")
        identical = len(shared) - len(uuid_only) - len(real_diffs) - len(quarantined)
        print(f"identical bytes      : {identical}")
        print(f"equivalent (uuid-only): {len(uuid_only)}")
        label = "quarantined (broken examples)"
        print(f"{label:<22}: {len(quarantined)}" + ("  — retires once the fixed spec lands" if quarantined else "  — none: reference is fully compared"))
        print(f"REAL DIFFERENCES      : {len(real_diffs)}")
        print(f"only in baseline      : {len(only_base)}")
        print(f"only in new           : {len(only_new)}")

        for rel in only_base[:40]:
            print(f"  - removed: {rel}")
        for rel in only_new[:40]:
            print(f"  + added:   {rel}")
        for line in real_diffs[:40]:
            print(f"  ~ changed: {line}")
        hidden = max(0, len(only_base) - 40) + max(0, len(only_new) - 40)
        hidden += max(0, len(real_diffs) - 40)
        if hidden:
            print(f"  … {hidden} more not shown")

        print(f"openapi error placeholders: {ph_a} baseline vs {ph_b} new", end="")
        delta = ph_b - ph_a
        if delta > PLACEHOLDER_JITTER:
            print(f"  ** WARN: +{delta}, beyond the +/-{PLACEHOLDER_JITTER} jitter band **")
        elif -delta > PLACEHOLDER_JITTER:
            # A large drop is the expected shape once the upstream spec fix
            # (lance-format/lance-namespace@3b167e4c) reaches the tree. Reporting
            # it says the baseline is stale rather than that something broke.
            print(f"  ** {delta}: reference repaired, re-baseline **")
        else:
            print("  (within jitter, not a signal)")

        equivalent = not (real_diffs or only_base or only_new)
        print("\nVERDICT:", "EQUIVALENT" if equivalent else "DIFFERENT")
        if quarantined and equivalent:
            print(
                f"note: {len(quarantined)} reference pages could not generate request "
                "examples and were not compared. Expected until the tree carries "
                "lance-format/lance-namespace@3b167e4c; zero after."
            )
        return 0 if equivalent else 1


if __name__ == "__main__":
    raise SystemExit(main())
