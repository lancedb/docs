"""
Record an assembled candidate, and publish one exactly as it was checked.

The Assemble workflow builds and checks the site on every pull request and
push, then records the checked tree as a candidate: the commit each root was
read from, the refs it asked for, and a SHA-256 for every file. A checksum over
those file hashes names the candidate. That workflow cannot write to the
repository, so merging publishes nothing.

Publishing is the Publish workflow, run by hand with the ID of a successful
Assemble run and its candidate's checksum. It takes that run's candidate -- it
never rebuilds -- checks every file against the record and the record against
the run, and pushes exactly those files as a new commit on top of the target
branch:

    staging      the `staging` branch, for a hosted preview of the combined site
    production   the `assembled` branch

Production takes only candidates built on `main` from both producers' `main`.
If the run's artifact has expired, an earlier publication of the same candidate
on the target branch is published again once its files match the checksum, so
a rollback can reach further back than artifacts are kept.

Hard-fails before pushing whenever the evidence does not hold.

Usage:

    python scripts/candidate.py record SITE --sources LINE [--ref ROOT=REF ...] --output FILE
    python scripts/candidate.py publish --target staging|production --run-id ID
        --site-sha256 HEX --run RUN_JSON --repository OWNER/NAME --content-dir DIR
        [--candidate DIR] [--remote REMOTE]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Each target writes one branch and nothing else.
TARGETS = {"staging": "staging", "production": "assembled"}
WORKFLOW = ".github/workflows/assemble.yml"
# A pull request runs its own copy of the workflow, so its output is no
# evidence of what this repository's workflow checked.
EVENTS = ("push", "workflow_dispatch")
# The root read from this repository: its commit is the one the run checked out.
DOCS_ROOT = "build"
# The roots in assemble.yaml, in its order. A record names each exactly once.
ROOTS = ("lancedb", "enterprise", DOCS_ROOT)
# The roots read from the other repositories, each from a ref the run asked for.
PRODUCERS = ("lancedb", "enterprise")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
CHECKSUM_RE = re.compile(r"[0-9a-f]{64}")
REF_RE = re.compile(r"[A-Za-z0-9._/-]+")
BOT = ("github-actions[bot]", "41898282+github-actions[bot]@users.noreply.github.com")


class CandidateError(Exception):
    """Evidence that does not establish the candidate: nothing is published."""


# --------------------------------------------------------------------------- #
# the record
# --------------------------------------------------------------------------- #


def file_hashes(site: Path) -> dict[str, str]:
    """SHA-256 of every file under `site`, hidden ones included."""
    if not site.is_dir():
        raise CandidateError(f"{site} is not a directory")
    hashes = {}
    for directory, dirs, files in os.walk(site):
        for name in dirs + files:
            path = Path(directory, name)
            rel = path.relative_to(site).as_posix()
            # A symlink cannot survive the artifact, and a path the checksum
            # listing cannot spell would make the checksum ambiguous.
            if path.is_symlink():
                raise CandidateError(f"{rel} is a symlink")
            if any(c in rel for c in "\\\n\r"):
                raise CandidateError(f"{rel!r} contains a backslash or line break")
        for name in files:
            path = Path(directory, name)
            if not path.is_file():
                raise CandidateError(f"{path.relative_to(site)} is not a regular file")
            hashes[path.relative_to(site).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    if not hashes:
        raise CandidateError(f"{site} has no files")
    return dict(sorted(hashes.items()))


def site_checksum(hashes: dict[str, str]) -> str:
    """SHA-256 of the `sha256sum` listing of the site, in byte order of path.

    Recomputable without this script:
    (cd site && find . -type f -printf '%P\\0' | LC_ALL=C sort -z | xargs -0 sha256sum) | sha256sum
    """
    listing = "".join(f"{hashes[path]}  {path}\n" for path in sorted(hashes))
    return hashlib.sha256(listing.encode()).hexdigest()


def parse_pairs(text: str, what: str) -> dict[str, str]:
    pairs = {}
    for item in text.split():
        name, sep, value = item.partition("=")
        if not sep or not name or name in pairs:
            raise CandidateError(f"cannot read {what} {text!r}")
        pairs[name] = value
    return pairs


def exactly(found: dict, names: tuple[str, ...], what: str) -> None:
    """`found` must name each of `names` and nothing else."""
    if not isinstance(found, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in found.items()
    ):
        raise CandidateError(f"{what} {found!r} are not a map of names to strings")
    missing = [name for name in names if name not in found]
    unexpected = sorted(set(found) - set(names))
    if missing or unexpected:
        raise CandidateError(
            f"{what} must name exactly {', '.join(names)}: "
            f"missing {missing}, unexpected {unexpected}"
        )


def check_sources(sources: dict) -> dict[str, str]:
    """Every root of assemble.yaml, each a clean commit, in that order."""
    exactly(sources, ROOTS, "sources")
    for name in ROOTS:
        # The assembler marks a dirty root `<sha>+uncommitted` and a root
        # outside git `unversioned`: neither names what was built.
        if not COMMIT_RE.fullmatch(sources[name]):
            raise CandidateError(f"source {name}={sources[name]} is not a clean commit")
    return {name: sources[name] for name in ROOTS}


def check_refs(refs: dict) -> dict[str, str]:
    """The ref each producer was read from."""
    exactly(refs, PRODUCERS, "refs")
    for name in PRODUCERS:
        if not REF_RE.fullmatch(refs[name]):
            raise CandidateError(f"ref {name}={refs[name]!r} is not a plain ref name")
    return {name: refs[name] for name in PRODUCERS}


def parse_sources(text: str) -> dict[str, str]:
    """`lancedb=<sha> enterprise=<sha> build=<sha>`, as the assembler prints it."""
    return check_sources(parse_pairs(text, "sources"))


def parse_refs(text: str) -> dict[str, str]:
    return check_refs(parse_pairs(text, "refs"))


def run_context() -> dict[str, str] | None:
    """The workflow run recording the candidate; None outside GitHub Actions."""
    env = os.environ
    if "GITHUB_RUN_ID" not in env:
        return None
    return {
        "repository": env["GITHUB_REPOSITORY"],
        "run_id": env["GITHUB_RUN_ID"],
        "run_attempt": env.get("GITHUB_RUN_ATTEMPT", ""),
        "event": env.get("GITHUB_EVENT_NAME", ""),
        "ref": env.get("GITHUB_REF", ""),
        "sha": env.get("GITHUB_SHA", ""),
        "workflow_ref": env.get("GITHUB_WORKFLOW_REF", ""),
    }


def record(site: Path, sources_line: str, refs: list[str], output: Path) -> dict:
    sources = parse_sources(sources_line)
    refs = parse_refs(" ".join(refs))
    hashes = file_hashes(site)
    manifest = {
        "format": 1,
        "site_sha256": site_checksum(hashes),
        "files": len(hashes),
        "sources": sources,
        "refs": refs,
        "run": run_context(),
        "file_sha256": hashes,
    }
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def record_summary(manifest: dict) -> str:
    run = manifest["run"] or {}
    lines = [
        "### Candidate",
        "",
        f"Site checksum `{manifest['site_sha256']}`: {manifest['files']} files.",
        "",
        "| Root | Commit | Ref |",
        "|---|---|---|",
    ]
    for name, commit in manifest["sources"].items():
        ref = run.get("ref", "") if name == DOCS_ROOT else manifest["refs"].get(name, "")
        lines.append(f"| {name} | `{commit}` | {ref} |")
    lines += [
        "",
        f"To publish it, run the Publish workflow with run ID `{run.get('run_id', '?')}`"
        " and this checksum. Production accepts it only from a successful run on"
        " `main` that read both producers' `main`.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# git
# --------------------------------------------------------------------------- #


def git(*args: str, data: bytes | None = None, env: dict | None = None) -> bytes:
    result = subprocess.run(
        ["git", *args],
        input=data,
        capture_output=True,
        env={**os.environ, **(env or {})},
    )
    if result.returncode:
        message = result.stderr.decode(errors="replace").strip()
        raise CandidateError(f"git {args[0]} failed: {message}")
    return result.stdout


def text(*args: str, **kwargs) -> str:
    return git(*args, **kwargs).decode().strip()


def write_tree(site: Path, paths: list[str], prefix: str) -> str:
    """A tree holding the files at `paths`, byte for byte, under `prefix`.

    Built from the bytes on disk, so neither ignore rules nor line-ending
    conversion can change what is published.
    """
    listing = "".join(f"{site / path}\n" for path in paths).encode()
    blobs = text("hash-object", "-w", "--no-filters", "--stdin-paths", data=listing).split()
    entries = b"".join(
        f"100644 blob {blob}\t{prefix}{path}\0".encode() for blob, path in zip(blobs, paths)
    )
    with tempfile.TemporaryDirectory() as tmp:
        index = {"GIT_INDEX_FILE": str(Path(tmp, "index"))}
        git("update-index", "-z", "--add", "--index-info", data=entries, env=index)
        return text("write-tree", env=index)


def tree_hashes(tree: str, prefix: str) -> dict[str, str]:
    """SHA-256 of every file in `tree`, which must hold only files under `prefix`."""
    paths, blobs = [], []
    for entry in git("ls-tree", "-r", "-z", "--full-tree", tree).decode().split("\0"):
        if not entry:
            continue
        meta, path = entry.split("\t", 1)
        mode, kind, blob = meta.split()
        if kind != "blob" or mode != "100644":
            raise CandidateError(f"{path} is not a regular file in {tree}")
        if not path.startswith(prefix):
            raise CandidateError(f"{path} is outside {prefix or 'the root'} in {tree}")
        paths.append(path[len(prefix) :])
        blobs.append(blob)
    out = git("cat-file", "--batch", data="".join(f"{b}\n" for b in blobs).encode())
    hashes, offset = {}, 0
    for path in paths:
        header_end = out.index(b"\n", offset)
        size = int(out[offset:header_end].split()[2])
        content = out[header_end + 1 : header_end + 1 + size]
        hashes[path] = hashlib.sha256(content).hexdigest()
        offset = header_end + 1 + size + 1
    return dict(sorted(hashes.items()))


def remote_head(remote: str, branch: str) -> str | None:
    """The branch's current commit on the remote, fetched; None if it is absent."""
    if not text("ls-remote", "--heads", remote, f"refs/heads/{branch}"):
        return None
    local = f"refs/candidate-publish/{branch}"
    git("fetch", "--quiet", "--no-tags", remote, f"+refs/heads/{branch}:{local}")
    return text("rev-parse", "--verify", f"{local}^{{commit}}")


# --------------------------------------------------------------------------- #
# publishing
# --------------------------------------------------------------------------- #


def content_prefix(value: str) -> str:
    """The directory Mintlify reads `docs.json` from, as a tree path prefix."""
    if not value.strip():
        raise CandidateError(
            "no content directory: set the repository variable MINTLIFY_CONTENT_DIR "
            "to the path in Mintlify's Git settings ('/' for the repository root)"
        )
    path = value.strip().strip("/")
    if path in ("", "."):
        return ""
    if any(part in ("", ".", "..") for part in path.split("/")):
        raise CandidateError(f"content directory {value!r} is not a plain relative path")
    return path + "/"


def check_run(run: dict, run_id: str, repository: str, target: str) -> None:
    """The run must be a finished, successful Assemble run of this repository."""
    if str(run.get("id")) != run_id:
        raise CandidateError(f"the run evidence describes run {run.get('id')}, not {run_id}")
    if run["repository"].get("full_name") != repository:
        raise CandidateError(f"run {run_id} is not a run of {repository}")
    if run.get("path") != WORKFLOW:
        raise CandidateError(f"run {run_id} ran {run.get('path')}, not {WORKFLOW}")
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        raise CandidateError(
            f"run {run_id} has not succeeded: {run.get('status')}, {run.get('conclusion')}"
        )
    if run.get("event") not in EVENTS:
        raise CandidateError(
            f"run {run_id} was a {run.get('event')} run; only {' and '.join(EVENTS)} "
            "runs check this repository's own workflow"
        )
    if target == "production" and run.get("head_branch") != "main":
        raise CandidateError(
            f"production takes candidates built on main; run {run_id} built "
            f"{run.get('head_branch')}"
        )


def check_artifact(candidate: Path, run: dict, expected: str) -> tuple[dict, dict]:
    """The artifact's files must match its record, and the record this run."""
    try:
        manifest = json.loads((candidate / "candidate.json").read_text())
    except FileNotFoundError:
        raise CandidateError(f"{candidate} holds no candidate.json") from None
    except json.JSONDecodeError as exc:
        raise CandidateError(f"candidate.json is not JSON: {exc}") from None
    if not isinstance(manifest, dict):
        raise CandidateError("candidate.json is not a record")
    recorded_by = manifest.get("run") if isinstance(manifest.get("run"), dict) else {}
    if str(recorded_by.get("run_id")) != str(run["id"]) or recorded_by.get(
        "repository"
    ) != run["repository"]["full_name"]:
        raise CandidateError(
            f"the candidate was recorded by {recorded_by.get('repository')} run "
            f"{recorded_by.get('run_id')}, not run {run['id']}"
        )
    recorded = manifest.get("file_sha256")
    if not isinstance(recorded, dict) or not all(isinstance(h, str) for h in recorded.values()):
        raise CandidateError("the record has no file hashes")
    actual = file_hashes(candidate / "site")
    if actual != recorded:
        missing = sorted(set(recorded) - set(actual))
        extra = sorted(set(actual) - set(recorded))
        changed = sorted(p for p in set(actual) & set(recorded) if actual[p] != recorded[p])
        raise CandidateError(
            "the artifact's files differ from its record: "
            f"missing {missing[:5]}, extra {extra[:5]}, changed {changed[:5]}"
        )
    if site_checksum(recorded) != manifest.get("site_sha256"):
        raise CandidateError("the record's checksum does not match its own file list")
    if manifest["site_sha256"] != expected:
        raise CandidateError(
            f"run {run['id']} recorded candidate {manifest['site_sha256']}, not {expected}"
        )
    return check_sources(manifest.get("sources")), check_refs(manifest.get("refs"))


TRAILER = "^{}: (.+)$"


def trailer(message: str, key: str) -> str | None:
    found = re.findall(TRAILER.format(re.escape(key)), message, re.M)
    return found[-1].strip() if found else None


def earlier_publication(head: str | None, run_id: str, expected: str):
    """The latest commit on the branch that published this run's candidate."""
    if head is None:
        return None
    for entry in git("log", "-z", "--first-parent", "--format=%H%n%B", head).decode().split("\0"):
        commit, _, message = entry.strip().partition("\n")
        if trailer(message, "Candidate-Run") == run_id and (
            trailer(message, "Candidate-Site-SHA256") == expected
        ):
            sources = parse_sources(trailer(message, "Sources") or "")
            refs = parse_refs(trailer(message, "Refs") or "")
            return commit, sources, refs
    return None


def publish(args: argparse.Namespace) -> dict:
    branch = TARGETS[args.target]
    if not re.fullmatch(r"[0-9]+", args.run_id):
        raise CandidateError(f"run ID {args.run_id!r} is not a number")
    expected = args.site_sha256.strip()
    if not CHECKSUM_RE.fullmatch(expected):
        raise CandidateError(f"{args.site_sha256!r} is not a lowercase SHA-256")
    prefix = content_prefix(args.content_dir)
    try:
        run = json.loads(args.run.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateError(f"no evidence of run {args.run_id}: {exc}") from None
    if not isinstance(run, dict) or not isinstance(run.get("repository"), dict):
        raise CandidateError(f"the evidence of run {args.run_id} is not a workflow run")
    check_run(run, args.run_id, args.repository, args.target)

    head = remote_head(args.remote, branch)
    artifact = args.candidate
    if artifact is not None and artifact.is_dir() and any(artifact.iterdir()):
        sources, refs = check_artifact(artifact, run, expected)
        paths = list(file_hashes(artifact / "site"))
        tree = write_tree(artifact / "site", paths, prefix)
        origin = "the run's artifact"
    else:
        earlier = earlier_publication(head, args.run_id, expected)
        if earlier is None:
            raise CandidateError(
                f"run {args.run_id} has no candidate artifact, and {branch} has no "
                f"earlier publication of candidate {expected} from it"
            )
        commit, sources, refs = earlier
        tree = text("rev-parse", f"{commit}^{{tree}}")
        origin = f"earlier publication {commit}"

    if sources[DOCS_ROOT] != run.get("head_sha"):
        raise CandidateError(
            f"the candidate was built from {DOCS_ROOT} {sources[DOCS_ROOT]}, "
            f"but run {args.run_id} checked out {run.get('head_sha')}"
        )
    # Both producers are always named: a record missing one was refused above.
    if args.target == "production":
        off_main = {name: refs[name] for name in PRODUCERS if refs[name] != "main"}
        if off_main:
            raise CandidateError(f"production takes producers' main; this read {off_main}")

    # Whatever the source, what is about to be pushed is checked once more.
    published = tree_hashes(tree, prefix)
    if site_checksum(published) != expected:
        raise CandidateError(f"the tree from {origin} is not candidate {expected}")

    lines = [
        f"Publish candidate {expected[:12]} to {args.target}",
        "",
        f"Candidate-Site-SHA256: {expected}",
        f"Candidate-Run: {args.run_id}",
        f"Candidate-Files: {len(published)}",
        "Sources: " + " ".join(f"{k}={v}" for k, v in sources.items()),
        "Refs: " + " ".join(f"{k}={v}" for k, v in refs.items()),
    ]
    if os.environ.get("GITHUB_RUN_ID"):
        lines.append(
            f"Published-By: {os.environ.get('GITHUB_ACTOR')} in "
            f"{os.environ.get('GITHUB_REPOSITORY')} run {os.environ['GITHUB_RUN_ID']}"
        )
    identity = {
        "GIT_AUTHOR_NAME": BOT[0],
        "GIT_AUTHOR_EMAIL": BOT[1],
        "GIT_COMMITTER_NAME": BOT[0],
        "GIT_COMMITTER_EMAIL": BOT[1],
    }
    parents = ["-p", head] if head else []
    commit = text(
        "commit-tree", "--no-gpg-sign", tree, *parents, "-F", "-",
        data=("\n".join(lines) + "\n").encode(),
        env=identity,
    )
    # Never forced: if the branch moved since it was read, the push fails and
    # the branch keeps whatever moved it.
    git("push", "--quiet", args.remote, f"{commit}:refs/heads/{branch}")
    if remote_head(args.remote, branch) != commit:
        raise CandidateError(f"{branch} does not point at {commit} after the push")
    return {
        "target": args.target,
        "branch": branch,
        "commit": commit,
        "parent": head,
        "site_sha256": expected,
        "files": len(published),
        "run_id": args.run_id,
        "sources": sources,
        "refs": refs,
        "from": origin,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    commands = parser.add_subparsers(dest="command", required=True)

    rec = commands.add_parser("record", help="record a checked site as a candidate")
    rec.add_argument("site", type=Path)
    rec.add_argument("--sources", required=True, help="the assembler's sources line")
    rec.add_argument("--ref", action="append", default=[], metavar="ROOT=REF")
    rec.add_argument("--output", type=Path, required=True)
    rec.add_argument("--summary", type=Path, help="append a Markdown summary to this file")

    pub = commands.add_parser("publish", help="publish one recorded candidate")
    pub.add_argument("--target", choices=TARGETS, required=True)
    pub.add_argument("--run-id", required=True)
    pub.add_argument("--site-sha256", required=True)
    pub.add_argument("--run", type=Path, required=True, help="the run as the API returns it")
    pub.add_argument("--repository", required=True, help="OWNER/NAME of this repository")
    pub.add_argument("--content-dir", required=True)
    pub.add_argument("--candidate", type=Path, help="the downloaded candidate artifact")
    pub.add_argument("--remote", default="origin")
    pub.add_argument("--summary", type=Path, help="append a Markdown summary to this file")
    args = parser.parse_args()

    try:
        if args.command == "record":
            manifest = record(args.site, args.sources, args.ref, args.output)
            print(
                f"candidate {manifest['site_sha256']}: {manifest['files']} files from "
                + " ".join(f"{k}={v}" for k, v in manifest["sources"].items())
            )
            if args.summary:
                with args.summary.open("a") as summary:
                    summary.write(record_summary(manifest))
            return 0
        result = publish(args)
    except CandidateError as exc:
        print(f"candidate: refused: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    if args.summary:
        with args.summary.open("a") as summary:
            summary.write(
                f"Published candidate `{result['site_sha256']}` from run "
                f"{result['run_id']} ({result['from']}) to `{result['branch']}` "
                f"as `{result['commit']}`.\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
