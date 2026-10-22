"""
Prepare a saved documentation edition from one product release, and check the
saved editions.

Product sources come from their release commits. Docs-owned content and
assemble.yaml come from canonical lancedb/docs main, with matching scripts/.
Preparation never replaces an existing edition. Preservation across merges
requires `check --base` to pass before merging. See VERSIONING.md.

Usage:

    python scripts/edition.py prepare oss VERSION --enterprise REF [--lancedb REPO] [--enterprise-repo REPO]
        [--docs REF]
    python scripts/edition.py prepare enterprise VERSION [--bundled-lancedb COMMIT]
        [--lancedb REPO] [--enterprise-repo REPO] [--docs REF]
    python scripts/edition.py check [--base COMMIT]
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
# Every assemble.yaml root needs provenance here.
SOURCES = {
    "lancedb": ("lancedb/lancedb", "docs/web"),
    "enterprise": ("lancedb/sophon", "docs/web"),
    "build": ("lancedb/docs", "docs"),
}
# Pin docs inputs to canonical main; squash merges can discard branch commits.
DOCS_REMOTE = "origin"
# Fully qualified: a local branch or tag named origin/main would win the shorthand.
DOCS_MAIN = f"refs/remotes/{DOCS_REMOTE}/main"
DOCS_URLS = re.compile(
    r"(?:https://github\.com/|ssh://git@github\.com/|git@github\.com:)lancedb/docs(?:\.git)?", re.IGNORECASE
)
# Qualification requires an annotated tag on the release commit.
BUNDLED_LANCEDB = "src/lancedb"
QUALIFIED_TAG = "release-qualified/v{version}"
# Files that would change how Git stores the edition's own files.
GIT_CONTROL = frozenset({".gitignore", ".gitattributes"})
HEADER = """\
# Release record, written by scripts/edition.py prepare. `release` and `sources`
# say what the edition was built from, and `prepared` describes the snapshot as
# first saved. Once merged, neither changes; `edition.py check --base` refuses
# a change to them. A reviewed correction to the saved edition adds an entry at
# the end of `corrections` with the checksum of the corrected edition; see
# "Saved editions" in README.md.
"""


class EditionError(Exception):
    """Inputs or state that must stop preparation before anything is saved."""


# git


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


# release inputs


def source(name: str, commit: str) -> dict:
    repository, folder = SOURCES[name]
    return {"repository": repository, "path": folder, "commit": commit}


def is_ancestor(repo: Path, commit: str, of: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", commit, of], capture_output=True
    )
    if result.returncode not in (0, 1):
        raise EditionError(f"git merge-base in {repo} failed: {result.stderr.decode(errors='replace').strip()}")
    return result.returncode == 0


def docs_main(repo: Path) -> str:
    """Fetch canonical main; an existing tracking ref may come from a fork."""
    result = subprocess.run(
        ["git", "-C", str(repo), "remote", "get-url", "--all", DOCS_REMOTE], capture_output=True
    )
    # As fetching would use them, with any `insteadOf` rewriting applied.
    urls = result.stdout.decode().splitlines() if result.returncode == 0 else []
    others = [url for url in urls if not DOCS_URLS.fullmatch(url)]
    if not urls or others:
        raise EditionError(
            f"this checkout's {DOCS_REMOTE} is {'not supported' if urls else 'not set'}, not "
            f"{SOURCES['build'][0]}; a fork's main can hold commits {SOURCES['build'][0]} never has, "
            "so prepare an edition in a clone of https://github.com/lancedb/docs.git, over HTTPS "
            "or SSH"
        )
    try:
        git(repo, "fetch", "--quiet", "--no-tags", DOCS_REMOTE, f"+refs/heads/main:{DOCS_MAIN}")
    except EditionError:
        raise EditionError(
            f"could not fetch {SOURCES['build'][0]} main; check network access and Git authentication"
        ) from None
    return commit_of(repo, DOCS_MAIN, SOURCES["build"][0])


def docs_commit(repo: Path, ref: str | None) -> str:
    """Select docs inputs from main and require this checkout's scripts to match."""
    what = f"{SOURCES['build'][0]} --docs"
    main = docs_main(repo)
    commit = main if ref is None else commit_of(repo, ref, what)
    if not is_ancestor(repo, commit, main):
        raise EditionError(
            f"{what}: {ref} is {commit}, which {SOURCES['build'][0]} main ({main}) does not "
            "contain; name a commit of main, because merging gives a branch's commits new hashes"
        )
    changed = text(repo, "status", "--porcelain", "--", "scripts")
    if changed:
        raise EditionError(
            "commit or stash the changes to scripts/ first; they are the assembler that "
            "builds the edition:\n" + changed
        )
    if text(repo, "ls-tree", "-d", "HEAD", "--", "scripts") != text(repo, "ls-tree", "-d", commit, "--", "scripts"):
        raise EditionError(
            f"this checkout's scripts/ are not those of {commit}, which the record would name; "
            "check out a branch whose scripts/ match it, or name the commit of main they come "
            "from with --docs"
        )
    return commit


def release_inputs(args: argparse.Namespace) -> dict:
    product, version = args.product, args.version
    if not VERSION_RE.fullmatch(version):
        raise EditionError(f"{version!r} is not a stable release version such as 1.2.3")
    lancedb, sophon = args.lancedb.resolve(), args.enterprise_repo.resolve()
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
        if args.bundled_lancedb and not COMMIT_RE.fullmatch(args.bundled_lancedb):
            raise EditionError(
                f"--bundled-lancedb takes the full 40-character commit hash the release "
                f"bundles, not {args.bundled_lancedb!r}"
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
        },
    }


def resolve_release(args: argparse.Namespace, repo: Path) -> dict:
    requested = release_inputs(args)
    requested["sources"]["build"] = source("build", docs_commit(repo, args.docs))
    return requested


def identity(record: dict) -> dict:
    """Exclude docs main so preparing again after it advances keeps the saved edition."""
    release = record["release"]
    return {
        "release": release.get("commit"),
        "qualification": (release.get("qualification") or {}).get("object"),
        "bundled_lancedb": release.get("bundled_lancedb"),
        "enterprise_pages": (release.get("enterprise_pages") or {}).get("commit"),
        "lancedb": record["sources"]["lancedb"]["commit"],
        "enterprise": record["sources"]["enterprise"]["commit"],
    }


# records


def record_path(repo: Path, product: str, version: str) -> Path:
    return repo / "releases" / f"{product}-{version}.yaml"


def load_record(path: Path) -> dict:
    """A release record that holds everything `prepare` writes."""
    name = f"releases/{path.name}"
    try:
        record = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EditionError(f"{name}: unreadable: {exc}") from None
    problems = record_problems(record, path.name)
    if problems:
        raise EditionError(f"{name}: " + "; ".join(problems))
    return record


MISSING = object()


def record_problems(record: object, filename: str) -> list[str]:
    """Validate fields offline; commit existence is not checked.

    Compare fields only after validating their formats to avoid duplicate errors.
    """
    problems: list[str] = []

    def lookup(path: str) -> object:
        value = record
        for key in path.split("."):
            if not isinstance(value, dict) or key not in value:
                return MISSING
            value = value[key]
        return value

    def fields(path: str, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> bool:
        """Whether `path` is a mapping; reports any field it should not have."""
        value = lookup(path) if path else record
        if value is MISSING:
            problems.append(f"{path} is missing")
            return False
        if not isinstance(value, dict):
            problems.append(f"{path or 'the record'} is not a mapping")
            return False
        for key in sorted(set(value) - set(required) - set(optional), key=str):
            problems.append(f"{path + '.' if path else ''}{key} is not a field here")
        return True

    def check(path: str, valid, expected: str) -> object:
        """The value at `path` if `valid` accepts it, else None and a problem."""
        value = lookup(path)
        if value is MISSING:
            problems.append(f"{path} is missing")
        elif not valid(value):
            problems.append(f"{path} is {value!r}, not {expected}")
        else:
            return value
        return None

    def commit(value: object) -> bool:
        return isinstance(value, str) and COMMIT_RE.fullmatch(value) is not None

    def checksum(value: object) -> bool:
        return isinstance(value, str) and candidate.CHECKSUM_RE.fullmatch(value) is not None

    def line(value: object) -> bool:
        return isinstance(value, str) and bool(value.strip()) and "\n" not in value.strip()

    top = ("format", "product", "version", "edition", "release", "sources", "prepared", "corrections")
    if not fields("", top):
        return problems
    if check("format", lambda v: v == FORMAT and not isinstance(v, bool), str(FORMAT)) is None:
        return problems
    product = check("product", lambda v: v in PRODUCTS, " or ".join(PRODUCTS))
    version = check(
        "version", lambda v: isinstance(v, str) and VERSION_RE.fullmatch(v), "a version such as 1.2.3"
    )
    if product and version:
        check("edition", lambda v: v == f"editions/{product}/{version}", f"editions/{product}/{version}")
        if filename != f"{product}-{version}.yaml":
            problems.append(f"product and version name {product}-{version}.yaml, not {filename}")

    own = "lancedb" if product == "oss" else "enterprise"
    extra = ("enterprise_pages",) if product == "oss" else ("qualification", "bundled_lancedb")
    if product and fields("release", ("repository", "tag", "commit", *extra)):
        check("release.repository", lambda v: v == SOURCES[own][0], SOURCES[own][0])
        if version:
            check("release.tag", lambda v: v == f"v{version}", f"v{version}")
        check("release.commit", commit, "a commit hash")
        if product == "oss":
            if fields("release.enterprise_pages", ("ref", "commit")):
                check("release.enterprise_pages.ref", line, "the tag or commit named with --enterprise")
                check("release.enterprise_pages.commit", commit, "a commit hash")
        else:
            if fields("release.qualification", ("tag", "object")):
                if version:
                    tag_name = QUALIFIED_TAG.format(version=version)
                    check("release.qualification.tag", lambda v: v == tag_name, tag_name)
                check("release.qualification.object", commit, "a tag object hash")
            check("release.bundled_lancedb", commit, "a commit hash")

    if fields("sources", tuple(SOURCES), ("openapi",)):
        for name, (repository, folder) in SOURCES.items():
            if fields(f"sources.{name}", ("repository", "path", "commit")):
                check(f"sources.{name}.repository", lambda v, r=repository: v == r, repository)
                check(f"sources.{name}.path", lambda v, f=folder: v == f, folder)
                check(f"sources.{name}.commit", commit, "a commit hash")
        if lookup("sources.openapi") is not MISSING and fields(
            "sources.openapi", ("repository", "release", "path", "sha256")
        ):
            for key in ("repository", "release", "path"):
                check(f"sources.openapi.{key}", line, "a name")
            check("sources.openapi.sha256", checksum, "a SHA-256 checksum")

    if product:
        pairs = (
            [("sources.lancedb.commit", "release.commit"),
             ("sources.enterprise.commit", "release.enterprise_pages.commit")]
            if product == "oss"
            else [("sources.enterprise.commit", "release.commit"),
                  ("sources.lancedb.commit", "release.bundled_lancedb")]
        )
        for source_path, release_path in pairs:
            a, b = lookup(source_path), lookup(release_path)
            if commit(a) and commit(b) and a != b:
                problems.append(f"{source_path} is {a}, but {release_path} is {b}")

    if fields("prepared", ("files", "sha256")):
        check(
            "prepared.files",
            lambda v: isinstance(v, int) and not isinstance(v, bool) and v > 0,
            "a number of files",
        )
        check("prepared.sha256", checksum, "a SHA-256 checksum")

    corrections = check("corrections", lambda v: isinstance(v, list), "a list")
    accepted = lookup("prepared.sha256")
    for index, entry in enumerate(corrections or []):
        where = f"corrections[{index}]"
        if not isinstance(entry, dict):
            problems.append(f"{where} is not a mapping")
            accepted = None
            continue
        for key in sorted(set(entry) - {"summary", "files", "sha256"}, key=str):
            problems.append(f"{where}.{key} is not a field here")
        if not line(entry.get("summary")):
            problems.append(f"{where}.summary must be one line of text")
        files = entry.get("files", [])
        if not isinstance(files, list) or not all(isinstance(f, str) and f for f in files):
            problems.append(f"{where}.files must be a list of the edition's paths")
        sha256 = entry.get("sha256")
        if not checksum(sha256):
            problems.append(
                f"{where}.sha256 is {'missing' if sha256 is None else repr(sha256)}, not the "
                "SHA-256 checksum of the corrected edition that `check` reports"
            )
        elif sha256 == accepted:
            problems.append(f"{where}.sha256 is the checksum before it; a correction changes the edition")
        accepted = sha256
    return problems


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
    """Require the latest recorded checksum, including corrections."""
    current = candidate.site_checksum(saved_hashes(repo, record["edition"]))
    corrections = record["corrections"]
    accepted = corrections[-1]["sha256"] if corrections else record["prepared"]["sha256"]
    if current != accepted:
        since = "its latest recorded correction" if corrections else "it was prepared"
        raise EditionError(
            f"{record['edition']} has changed since {since}; its files now have checksum "
            f"{current}. For a reviewed correction, add an entry under `corrections` in "
            f"releases/{record['product']}-{record['version']}.yaml with `sha256: {current}`"
        )
    if not corrections:
        return "as prepared"
    count = len(corrections)
    return f"corrected ({count} correction{'s' if count > 1 else ''})"


# prepare


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
    requested = release_inputs(args)
    edition = repo / requested["edition"]
    path = record_path(repo, requested["product"], requested["version"])
    if os.path.lexists(edition) or os.path.lexists(path):
        return existing(repo, requested)

    requested["sources"]["build"] = source("build", docs_commit(repo, args.docs))
    (repo / "build").mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="edition-", dir=repo / "build")).resolve()
    try:
        roots = {}
        for name, (repository, folder) in SOURCES.items():
            roots[name] = staging / "roots" / name
            commit = requested["sources"][name]["commit"]
            source_repo = {"lancedb": args.lancedb, "enterprise": args.enterprise_repo}.get(name, repo)
            export(source_repo.resolve(), commit, folder, roots[name], f"{repository} {commit}")
        site = staging / "site"
        build = requested["sources"]["build"]["commit"]
        raw = yaml.safe_load(git(repo, "show", f"{build}:assemble.yaml").decode())
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


# check


def differences(was: object, now: object, path: str) -> list[str]:
    if isinstance(was, dict) and isinstance(now, dict):
        return [
            change
            for key in sorted(set(was) | set(now), key=str)
            for change in differences(
                was.get(key, MISSING), now.get(key, MISSING), f"{path}.{key}" if path else str(key)
            )
        ]
    if type(was) is type(now) and was == now:
        return []
    return [f"{path} {'was removed' if now is MISSING else 'was added' if was is MISSING else 'changed'}"]


def preserved(repo: Path, base: str) -> list[str]:
    """Require base editions and records to remain; only append corrections.

    Compare parsed records so formatting changes are harmless.
    """
    problems = []
    listing = git(repo, "ls-tree", "-z", "--name-only", base, "--", "releases/").decode()
    for rel in sorted(name for name in listing.split("\0") if name.endswith(".yaml")):
        path = repo / rel
        if not path.is_file():
            problems.append(f"{rel} is on the base {base} but has been removed; a saved release keeps its record")
            continue
        try:
            was = yaml.safe_load(git(repo, "show", f"{base}:{rel}"))
            now = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            was = now = None
        if not isinstance(was, dict) or not isinstance(now, dict):
            # `check` reports a malformed record by itself, and one on the base
            # can only be repaired by changing it.
            continue
        changes = differences(
            {k: v for k, v in was.items() if k != "corrections"},
            {k: v for k, v in now.items() if k != "corrections"},
            "",
        )
        before, after = was.get("corrections"), now.get("corrections")
        after = after if isinstance(after, list) else []
        for index, entry in enumerate(before if isinstance(before, list) else []):
            changes += differences(entry, after[index] if index < len(after) else MISSING, f"corrections[{index}]")
        if changes:
            problems.append(
                f"{rel} differs from the base {base}: {', '.join(changes)}; a saved release's "
                "record only gains corrections, at the end"
            )
    listing = git(repo, "ls-tree", "-z", "-d", "--name-only", base, "--",
                  *(f"editions/{product}/" for product in PRODUCTS)).decode()
    for rel in sorted(name for name in listing.split("\0") if name):
        _editions, product, version = rel.split("/")
        # A missing edition whose record remains is reported with the record.
        if not (repo / rel).is_dir() and not record_path(repo, product, version).is_file():
            problems.append(f"{rel} is on the base {base} but has been removed; a saved edition is never removed")
    return problems


def check(repo: Path = REPO_ROOT, base: str | None = None) -> tuple[list[str], list[str]]:
    """Return edition statuses and errors, optionally enforcing preservation from base."""
    lines, problems, paired = [], [], set()
    for path in sorted((repo / "releases").glob("*.yaml")):
        # The file name says which edition is this record's, even when the
        # record itself is malformed.
        paired.add("editions/" + path.stem.replace("-", "/", 1))
        try:
            record = load_record(path)
            if not (repo / record["edition"]).is_dir():
                raise EditionError(f"releases/{path.name}: {record['edition']} is missing")
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
    if base:
        problems += preserved(repo, commit_of(repo, base, "--base"))
    return lines, problems


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="save one release's edition")
    prep.add_argument("product", choices=PRODUCTS)
    prep.add_argument("version", help="the release version, such as 1.2.3")
    prep.add_argument("--enterprise", metavar="REF",
                      help="OSS only: the Enterprise tag or commit for the Enterprise pages")
    prep.add_argument("--bundled-lancedb", metavar="COMMIT",
                      help="Enterprise only: the LanceDB commit the release is expected to bundle")
    prep.add_argument("--lancedb", type=Path, default=REPO_ROOT.parent / "lancedb",
                      help="a lancedb/lancedb repository (default: ../lancedb)")
    prep.add_argument("--enterprise-repo", type=Path, default=REPO_ROOT.parent / "enterprise",
                      help="an Enterprise source repository (default: ../enterprise)")
    prep.add_argument("--docs", metavar="REF",
                      help="the commit of lancedb/docs main to read docs/ and assemble.yaml from "
                           "(default: main as fetched from origin when preparing)")
    checking = commands.add_parser("check", help="check the saved editions against their records")
    checking.add_argument("--base", metavar="COMMIT",
                          help="also require the editions and records on COMMIT, such as a pull "
                               "request's base, to be kept")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    try:
        if args.command == "prepare":
            print(prepare(args))
            return 0
        lines, problems = check(base=args.base)
    except (EditionError, candidate.CandidateError) as exc:
        print(f"edition: {exc}", file=sys.stderr)
        return 1
    if lines or not problems:
        print("\n".join(lines) or "no saved editions")
    for problem in problems:
        print(f"edition: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
