"""
Prepare a saved documentation edition from one product release, and check the
saved editions.

An edition is the site the assembler builds from one release's own sources,
saved under `editions/<product>/<version>/`. Its release record,
`releases/<product>-<version>.yaml`, names the release and the commit every
root was read from. Later builds compose the saved editions; nothing fetches
fresh content into one. See VERSIONING.md.

    prepare oss VERSION         the lancedb/lancedb tag vVERSION, with the
                                Enterprise pages from a lancedb/sophon tag or
                                commit named explicitly with --enterprise
    prepare enterprise VERSION  the lancedb/sophon tag vVERSION, which must be
                                qualified, with the shared pages from the
                                LanceDB commit that release bundles
    check                       pair every record with its edition, and require
                                an edition changed since it was prepared to
                                list its corrections

Each root is read from Git objects at its commit, never from a working tree,
and only from its documentation folder: `docs/web` of lancedb/lancedb and
lancedb/sophon, and `docs` of this repository at HEAD, which also supplies
assemble.yaml and the assembler.

A saved edition is never replaced. Preparing a release that already has one
changes nothing when the release inputs match its record, even after the
edition has been corrected, and stops when they differ. A preparation that
fails leaves neither an edition nor a record behind.

Usage:

    python scripts/edition.py prepare oss VERSION --enterprise REF [--lancedb REPO] [--sophon REPO]
    python scripts/edition.py prepare enterprise VERSION [--bundled-lancedb COMMIT]
        [--lancedb REPO] [--sophon REPO]
    python scripts/edition.py check
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import assemble  # noqa: E402
import candidate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
FORMAT = 1
PRODUCTS = ("oss", "enterprise")
# Editions are of stable releases. Prereleases need a policy first.
VERSION_RE = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
COMMIT_RE = candidate.COMMIT_RE
# Where each root in assemble.yaml is read from: repository and folder. A root
# added there without a source here stops preparation, so nothing reaches an
# edition without its provenance in the record.
SOURCES = {
    "lancedb": ("lancedb/lancedb", "docs/web"),
    "enterprise": ("lancedb/sophon", "docs/web"),
    "build": ("lancedb/docs", "docs"),
}
# An Enterprise release bundles LanceDB as this gitlink, and a release that
# passed qualification has an annotated tag of this name on its commit.
BUNDLED_LANCEDB = "src/lancedb"
QUALIFIED_TAG = "release-qualified/v{version}"
# Files that would change how Git stores the edition's own files.
GIT_CONTROL = frozenset({".gitignore", ".gitattributes"})
HEADER = """\
# Release record, written by scripts/edition.py prepare. `release` and `sources`
# say what the edition was built from, and `prepared` describes the snapshot as
# first saved. Neither changes afterwards. A reviewed correction to the saved
# edition adds an entry to `corrections`; see VERSIONING.md.
"""


class EditionError(Exception):
    """Inputs or state that must stop preparation before anything is saved."""


# --------------------------------------------------------------------------- #
# git
# --------------------------------------------------------------------------- #


def git(repo: Path, *args: str, data: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], input=data, capture_output=True
    )
    if result.returncode:
        message = result.stderr.decode(errors="replace").strip()
        raise EditionError(f"git {args[0]} in {repo} failed: {message}")
    return result.stdout


def text(repo: Path, *args: str) -> str:
    return git(repo, *args).decode().strip()


def commit_of(repo: Path, ref: str, what: str) -> str:
    try:
        return text(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    except EditionError:
        raise EditionError(f"{what}: {ref} is not a commit in {repo}; fetch it first") from None


def tag(repo: Path, name: str, what: str) -> tuple[str, str]:
    """The object a tag points at and the commit it names."""
    ref = f"refs/tags/{name}"
    try:
        obj = text(repo, "rev-parse", "--verify", "--quiet", ref)
    except EditionError:
        raise EditionError(f"{what}: {repo} has no tag {name}; fetch the release tags first") from None
    return obj, commit_of(repo, ref, what)


def pinned(repo: Path, ref: str, what: str) -> str:
    """A tag or a full commit hash; a branch can move, so it is not accepted."""
    if COMMIT_RE.fullmatch(ref):
        return commit_of(repo, ref, what)
    if not text(repo, "tag", "--list", ref):
        raise EditionError(
            f"{what}: {ref} is neither a tag in {repo} nor a full commit hash; "
            "name a pinned revision, not a branch"
        )
    return tag(repo, ref, what)[1]


def bundled_lancedb(sophon: Path, commit: str, what: str) -> str:
    """The LanceDB commit an Enterprise release bundles, from its gitlink."""
    listing = git(sophon, "ls-tree", "-z", commit, "--", BUNDLED_LANCEDB).decode()
    meta = listing.split("\t", 1)[0].split()
    if len(meta) != 3 or meta[0] != "160000":
        raise EditionError(
            f"{what} records no bundled LanceDB: {BUNDLED_LANCEDB} is not a gitlink "
            f"at {commit}"
        )
    return meta[2]


def export(repo: Path, commit: str, folder: str, dest: Path, what: str) -> None:
    """Write the files under `folder` at `commit` into `dest`, byte for byte."""
    listing = git(repo, "ls-tree", "-r", "-z", "--full-tree", commit, "--", folder)
    paths, blobs = [], []
    for entry in listing.decode().split("\0"):
        if not entry:
            continue
        meta, path = entry.split("\t", 1)
        mode, kind, blob = meta.split()
        if not path.startswith(folder + "/"):
            continue
        if any(part in ("", ".", "..") for part in path.split("/")):
            raise EditionError(f"{what}: {path!r} is not a plain path inside {folder}")
        if mode == "120000" or kind != "blob":
            raise EditionError(
                f"{what}: {path} is a {'symlink' if mode == '120000' else kind}; "
                "an edition reads only regular files"
            )
        paths.append(path[len(folder) + 1 :])
        blobs.append(blob)
    if not paths:
        raise EditionError(
            f"{what} has no {folder}: an edition is built only from a release's own "
            "documentation sources"
        )
    out = git(repo, "cat-file", "--batch", data="".join(f"{b}\n" for b in blobs).encode())
    offset = 0
    for path, blob in zip(paths, blobs):
        header_end = out.index(b"\n", offset)
        header = out[offset:header_end].split()
        if len(header) != 3:
            raise EditionError(f"{what}: {repo} lacks {folder}/{path} ({blob}); fetch it first")
        size = int(header[2])
        target = dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(out[header_end + 1 : header_end + 1 + size])
        offset = header_end + 1 + size + 1


# --------------------------------------------------------------------------- #
# release inputs
# --------------------------------------------------------------------------- #


def source(name: str, commit: str) -> dict:
    repository, folder = SOURCES[name]
    return {"repository": repository, "path": folder, "commit": commit}


def docs_commit(repo: Path) -> str:
    """This repository's HEAD, which must hold everything the build reads."""
    changed = text(repo, "status", "--porcelain", "--", "assemble.yaml", "scripts", SOURCES["build"][1])
    if changed:
        raise EditionError(
            "commit or stash the changes to assemble.yaml, scripts/ and docs/ first; "
            "the record names this checkout's HEAD:\n" + changed
        )
    return commit_of(repo, "HEAD", "lancedb/docs")


def resolve_release(args: argparse.Namespace, repo: Path) -> dict:
    """The record a preparation would write, before anything is built."""
    product, version = args.product, args.version
    if not VERSION_RE.fullmatch(version):
        raise EditionError(f"{version!r} is not a stable release version such as 1.2.3")
    lancedb, sophon = args.lancedb.resolve(), args.sophon.resolve()
    if product == "oss":
        if args.bundled_lancedb:
            raise EditionError("--bundled-lancedb applies to Enterprise releases only")
        if not args.enterprise:
            raise EditionError(
                "an OSS edition includes the Enterprise pages, so name the lancedb/sophon "
                "tag or commit they come from with --enterprise"
            )
        release_tag = f"v{version}"
        commit = tag(lancedb, release_tag, f"lancedb/lancedb {release_tag}")[1]
        enterprise = pinned(sophon, args.enterprise, "lancedb/sophon --enterprise")
        release = {
            "repository": SOURCES["lancedb"][0],
            "tag": release_tag,
            "commit": commit,
            "enterprise_pages": {"ref": args.enterprise, "commit": enterprise},
        }
        shared = commit
    else:
        if args.enterprise:
            raise EditionError(
                "--enterprise applies to OSS releases; an Enterprise release reads its own pages"
            )
        release_tag = f"v{version}"
        what = f"lancedb/sophon {release_tag}"
        enterprise = tag(sophon, release_tag, what)[1]
        qualified = QUALIFIED_TAG.format(version=version)
        try:
            qualification, qualified_commit = tag(sophon, qualified, what)
        except EditionError:
            raise EditionError(
                f"{what} is not qualified: {sophon} has no tag {qualified}"
            ) from None
        if text(sophon, "cat-file", "-t", qualification) != "tag":
            raise EditionError(f"{what}: {qualified} is not an annotated tag")
        if qualified_commit != enterprise:
            raise EditionError(
                f"{what} is {enterprise}, but {qualified} qualifies {qualified_commit}"
            )
        shared = bundled_lancedb(sophon, enterprise, what)
        if args.bundled_lancedb and args.bundled_lancedb != shared:
            raise EditionError(
                f"--bundled-lancedb {args.bundled_lancedb} disagrees with {what}, which "
                f"bundles LanceDB {shared} at {BUNDLED_LANCEDB}"
            )
        commit_of(lancedb, shared, f"the LanceDB that {what} bundles")
        release = {
            "repository": SOURCES["enterprise"][0],
            "tag": release_tag,
            "commit": enterprise,
            "qualification": {"tag": qualified, "object": qualification},
            "bundled_lancedb": shared,
        }
    return {
        "format": FORMAT,
        "product": product,
        "version": version,
        "edition": f"editions/{product}/{version}",
        "release": release,
        "sources": {
            "lancedb": source("lancedb", shared),
            "enterprise": source("enterprise", enterprise),
            "build": source("build", docs_commit(repo)),
        },
    }


def identity(record: dict) -> dict:
    """What makes two preparations the same release: commits, not ref names.

    This repository's own commit is left out: re-running a release from a later
    HEAD is still the same release, and its saved edition is kept as it is.
    """
    release = record["release"]
    return {
        "release": release.get("commit"),
        "qualification": (release.get("qualification") or {}).get("object"),
        "bundled_lancedb": release.get("bundled_lancedb"),
        "enterprise_pages": (release.get("enterprise_pages") or {}).get("commit"),
        "lancedb": record["sources"]["lancedb"]["commit"],
        "enterprise": record["sources"]["enterprise"]["commit"],
    }


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #


def record_path(repo: Path, product: str, version: str) -> Path:
    return repo / "releases" / f"{product}-{version}.yaml"


def load_record(path: Path) -> dict:
    """A release record, checked enough to compare and to verify its edition."""
    try:
        record = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EditionError(f"{path.name}: unreadable: {exc}") from None
    if not isinstance(record, dict) or record.get("format") != FORMAT:
        raise EditionError(f"{path.name}: not a format {FORMAT} release record")
    product, version = record.get("product"), record.get("version")
    if product not in PRODUCTS or not isinstance(version, str) or not VERSION_RE.fullmatch(version):
        raise EditionError(f"{path.name}: product and version are not a release")
    if path.name != f"{product}-{version}.yaml":
        raise EditionError(f"{path.name}: names {product} {version}")
    if record.get("edition") != f"editions/{product}/{version}":
        raise EditionError(f"{path.name}: edition must be editions/{product}/{version}")
    prepared = record.get("prepared")
    if not isinstance(prepared, dict) or not candidate.CHECKSUM_RE.fullmatch(
        str(prepared.get("sha256"))
    ):
        raise EditionError(f"{path.name}: `prepared` has no snapshot checksum")
    try:
        identity(record)
    except (AttributeError, KeyError, TypeError):
        raise EditionError(f"{path.name}: `release` or `sources` is incomplete") from None
    corrections = record.get("corrections")
    if not isinstance(corrections, list) or not all(
        isinstance(c, dict) and isinstance(c.get("summary"), str) and c["summary"].strip()
        for c in corrections
    ):
        raise EditionError(f"{path.name}: every correction needs a one-line `summary`")
    return record


def saved_hashes(repo: Path, edition: str) -> dict[str, str]:
    """The edition's files as Git would save them: ignored local files, such as
    `.DS_Store`, are not part of it. Preparation refuses an edition that has
    any ignored path of its own."""
    hashes = candidate.file_hashes(repo / edition)
    listing = "".join(f"{edition}/{p}\0" for p in hashes)
    ignored = ignored_paths(repo, listing, index=True)
    return {p: h for p, h in hashes.items() if f"{edition}/{p}" not in ignored}


def ignored_paths(repo: Path, listing: str, index: bool = False) -> set[str]:
    """The NUL-separated paths in `listing` that .gitignore leaves out.

    With `index`, files Git already tracks are never ignored, as for `git add`.
    """
    result = subprocess.run(
        ["git", "-C", str(repo), "check-ignore", "-z", "--stdin", *([] if index else ["--no-index"])],
        input=listing.encode(), capture_output=True,
    )
    if result.returncode not in (0, 1):
        raise EditionError(f"git check-ignore failed: {result.stderr.decode().strip()}")
    return {name for name in result.stdout.decode().split("\0") if name}


def snapshot_state(repo: Path, record: dict) -> str:
    """Whether the saved edition is still the prepared snapshot."""
    current = candidate.site_checksum(saved_hashes(repo, record["edition"]))
    if current == record["prepared"]["sha256"]:
        return "as prepared"
    count = len(record["corrections"])
    if not count:
        raise EditionError(
            f"{record['edition']} differs from its prepared snapshot, and "
            f"releases/{record['product']}-{record['version']}.yaml lists no correction"
        )
    return f"corrected ({count} correction{'s' if count > 1 else ''})"


# --------------------------------------------------------------------------- #
# prepare
# --------------------------------------------------------------------------- #


def existing(repo: Path, requested: dict) -> str:
    """What to say about a release that already has a record or an edition."""
    product, version = requested["product"], requested["version"]
    path = record_path(repo, product, version)
    folder = repo / requested["edition"]
    if not path.exists():
        raise EditionError(
            f"{requested['edition']} exists without releases/{path.name}; an edition "
            "is never replaced, so remove an incomplete one in a reviewed change first"
        )
    record = load_record(path)
    if not folder.is_dir():
        raise EditionError(
            f"releases/{path.name} exists, but {requested['edition']} does not; restore "
            "it from Git history rather than preparing it again"
        )
    recorded, wanted = identity(record), identity(requested)
    if recorded != wanted:
        changes = "; ".join(
            f"{key} recorded {recorded[key]}, requested {wanted[key]}"
            for key in recorded
            if recorded[key] != wanted[key]
        )
        raise EditionError(
            f"{product} {version} is already saved from different inputs ({changes}); "
            "an edition is never replaced, so review the difference instead"
        )
    try:
        state = snapshot_state(repo, record)
    except EditionError as exc:
        raise EditionError(f"{product} {version} is already saved; nothing changed, but {exc}") from None
    return (
        f"{product} {version} is already saved from these release inputs: "
        f"{requested['edition']} is {state}; nothing changed"
    )


def build_config(raw: dict, roots: dict[str, Path], output: Path, path: Path) -> Path:
    """assemble.yaml as committed, reading the exported roots into `output`."""
    raw = dict(raw, roots=[dict(entry) for entry in raw.get("roots", [])])
    names = [entry.get("name") for entry in raw.get("roots", [])]
    if sorted(names) != sorted(SOURCES):
        raise EditionError(
            f"assemble.yaml has roots {names}, but edition.py records the sources of "
            f"{sorted(SOURCES)}; add every root to SOURCES"
        )
    for entry in raw["roots"]:
        entry["path"] = str(roots[entry["name"]])
    raw["output"] = str(output)
    if raw.get("openapi"):
        # The spec is a file of the docs root; check the exported copy.
        dest = PurePosixPath(raw["openapi"]["dest"]).relative_to(SOURCES["build"][1])
        raw["openapi"] = dict(raw["openapi"], dest=str(roots["build"] / dest))
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def openapi_source(raw: dict, hashes: dict[str, str]) -> dict | None:
    """The released spec the docs root ships, and the hash of the saved copy."""
    spec = raw.get("openapi")
    if not spec:
        return None
    rel = PurePosixPath(spec["dest"]).relative_to(SOURCES["build"][1]).as_posix()
    return {
        "repository": spec["repo"],
        "release": spec["release"],
        "path": spec["source"],
        "sha256": hashes.get(rel),
    }


def place(staging: Path, site: Path, record: str, repo: Path, edition: Path, path: Path) -> None:
    """Move the edition into place, then create its record; undo both on failure."""
    created = [d for d in (edition.parent.parent, edition.parent, path.parent) if not d.exists()]
    moved = claimed = False
    try:
        for directory in (edition.parent, path.parent):
            directory.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(edition) or os.path.lexists(path):
            raise EditionError(f"{edition.relative_to(repo)} appeared while preparing; nothing saved")
        os.rename(site, edition)
        moved = True
        # Claim the record's name, then put the whole record there at once.
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL))
        claimed = True
        draft = staging / "record.yaml"
        draft.write_text(record, encoding="utf-8")
        os.replace(draft, path)
    except BaseException:
        if claimed:
            path.unlink()
        if moved:
            os.rename(edition, site)
        for directory in reversed(created):
            if directory.exists() and not any(directory.iterdir()):
                directory.rmdir()
        raise


def prepare(args: argparse.Namespace, repo: Path = REPO_ROOT) -> str:
    requested = resolve_release(args, repo)
    edition = repo / requested["edition"]
    path = record_path(repo, requested["product"], requested["version"])
    if os.path.lexists(edition) or os.path.lexists(path):
        return existing(repo, requested)

    (repo / "build").mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="edition-", dir=repo / "build")).resolve()
    try:
        roots = {}
        for name, (repository, folder) in SOURCES.items():
            roots[name] = staging / "roots" / name
            commit = requested["sources"][name]["commit"]
            source_repo = {"lancedb": args.lancedb, "enterprise": args.sophon}.get(name, repo)
            export(source_repo.resolve(), commit, folder, roots[name], f"{repository} {commit}")
        site = staging / "site"
        raw = yaml.safe_load(git(repo, "show", "HEAD:assemble.yaml").decode())
        config = assemble.load_config(build_config(raw, roots, site, staging / "assemble.yaml"))
        _count, warnings = assemble.assemble_site(config)
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)

        hashes = candidate.file_hashes(site)
        control = sorted(p for p in hashes if PurePosixPath(p).name in GIT_CONTROL)
        if control:
            raise EditionError(f"the edition would hold {control}, which change how Git stores it")
        listing = "".join(f"{requested['edition']}/{p}\0" for p in hashes) + f"releases/{path.name}\0"
        ignored = sorted(ignored_paths(repo, listing))
        if ignored:
            raise EditionError(f".gitignore would leave out {ignored[:5]}; the edition must be saved whole")

        openapi = openapi_source(raw, hashes)
        record = {
            **requested,
            "sources": {**requested["sources"], **({"openapi": openapi} if openapi else {})},
            "prepared": {"files": len(hashes), "sha256": candidate.site_checksum(hashes)},
            "corrections": [],
        }
        place(staging, site, HEADER + yaml.safe_dump(record, sort_keys=False), repo, edition, path)
    except (assemble.AssembleError, candidate.CandidateError) as exc:
        raise EditionError(str(exc)) from None
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        try:
            (repo / "build").rmdir()
        except OSError:
            pass
    sources = " ".join(f"{k}={v['commit']}" for k, v in requested["sources"].items())
    return (
        f"prepared {requested['product']} {requested['version']}: {len(hashes)} files, "
        f"sha256 {record['prepared']['sha256']}, in {requested['edition']} with "
        f"releases/{path.name}\nsources: {sources}"
    )


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #


def check(repo: Path = REPO_ROOT) -> tuple[list[str], list[str]]:
    """Each edition's state, and every problem with records and editions."""
    lines, problems, paired = [], [], set()
    for path in sorted((repo / "releases").glob("*.yaml")):
        try:
            record = load_record(path)
            if not (repo / record["edition"]).is_dir():
                raise EditionError(f"{path.name}: {record['edition']} is missing")
            paired.add(record["edition"])
            lines.append(f"{record['product']} {record['version']}: {snapshot_state(repo, record)}")
        except (EditionError, candidate.CandidateError) as exc:
            problems.append(str(exc))
    editions = repo / "editions"
    entries = sorted([*editions.glob("*"), *editions.glob("*/*")]) if editions.is_dir() else []
    names = [entry.relative_to(repo).as_posix() for entry in entries]
    ignored = ignored_paths(repo, "".join(f"{name}\0" for name in names), index=True)
    for entry, rel in zip(entries, names):
        if rel in ignored:
            continue
        if entry.parent == editions:
            if entry.name not in PRODUCTS or not entry.is_dir():
                problems.append(f"{rel} is not a product folder")
        elif rel not in paired:
            problems.append(f"{rel} has no release record")
    return lines, problems


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="save one release's edition")
    prep.add_argument("product", choices=PRODUCTS)
    prep.add_argument("version", help="the release version, such as 1.2.3")
    prep.add_argument("--enterprise", metavar="REF",
                      help="OSS only: the lancedb/sophon tag or commit for the Enterprise pages")
    prep.add_argument("--bundled-lancedb", metavar="COMMIT",
                      help="Enterprise only: the LanceDB commit the release is expected to bundle")
    prep.add_argument("--lancedb", type=Path, default=REPO_ROOT.parent / "lancedb",
                      help="a lancedb/lancedb repository (default: ../lancedb)")
    prep.add_argument("--sophon", type=Path, default=REPO_ROOT.parent / "sophon",
                      help="a lancedb/sophon repository (default: ../sophon)")
    commands.add_parser("check", help="check the saved editions against their records")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    try:
        if args.command == "prepare":
            print(prepare(args))
            return 0
        lines, problems = check()
    except (EditionError, candidate.CandidateError) as exc:
        print(f"edition: {exc}", file=sys.stderr)
        return 1
    print("\n".join(lines) or "no saved editions")
    for problem in problems:
        print(f"edition: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
