"""
Record a checked site as a candidate, and publish one exactly as it was checked.

Two workflows build a site and check it on every pull request and push, then
record the checked tree as a candidate: the commit each root was read from,
the refs it asked for, and a SHA-256 for every file. A checksum over those file
hashes names the candidate. Neither workflow can write to the repository, so
merging publishes nothing.

    Assemble   the assembled site, from the producers' docs and this repository
    Compose    the versioned site, from the saved editions in this repository

Publishing is the Publish workflow. It takes a run's candidate -- it never
rebuilds -- checks every file against the record and the record against the
run, and pushes exactly those files as a new commit on top of the target
branch:

    staging      the `staging` branch, a hosted preview of the site. By hand,
                 the candidate of a successful run of either workflow; by
                 itself, once staging shows the versioned site, main's newest
                 versioned candidate.
    production   the `assembled` branch. Only the versioned site: the
                 candidate staging shows, once Mintlify reports it live at the
                 staging address, or one production has published before, to
                 roll back; and only if production is as it was when the run
                 was requested.

If the run's artifact has expired, an earlier publication of the same candidate
on the target branch is published again once its files match the checksum, so
a rollback can reach further back than artifacts are kept.

Hard-fails before pushing whenever the evidence does not hold.

Usage:

    python scripts/candidate.py record SITE --sources LINE [--ref ROOT=REF ...] --output FILE
        [--site assembled|versioned]
    python scripts/candidate.py publish --target staging|production --run-id ID
        --site-sha256 HEX --run RUN_JSON --repository OWNER/NAME --content-dir DIR
        [--candidate DIR] [--remote REMOTE]
        [--staging-deployments JSON --staging-url URL --production-was COMMIT]   (production)
    python scripts/candidate.py publish --target staging --automatic [--run-id ID]
        [--run RUN_JSON] --repository OWNER/NAME --content-dir DIR [--candidate DIR]
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
from dataclasses import dataclass
from pathlib import Path

# Each target writes one branch and nothing else.
TARGETS = {"staging": "staging", "production": "assembled"}


@dataclass(frozen=True)
class Site:
    """A site one workflow builds, and the sources its candidates name."""

    name: str
    workflow: str
    # Its roots, each recorded with the commit it was read from.
    roots: tuple[str, ...]
    # The roots read from other repositories, each from a ref the run asked for.
    producers: tuple[str, ...]


# The roots in assemble.yaml, in its order. A record names each exactly once.
ASSEMBLED = Site(
    "assembled", ".github/workflows/assemble.yml", ("lancedb", "enterprise", "build"),
    ("lancedb", "enterprise"),
)
# Composed from the saved editions at one commit of this repository.
VERSIONED = Site("versioned", ".github/workflows/compose.yml", ("build",), ())
SITES = {site.name: site for site in (ASSEMBLED, VERSIONED)}
# A pull request runs its own copy of the workflow, so its output is no
# evidence of what this repository's workflow checked.
EVENTS = ("push", "workflow_dispatch")
# The root read from this repository: its commit is the one the run checked out.
DOCS_ROOT = "build"
# Mintlify reports each deployment of a commit as this check run, for people,
# and records it as a GitHub deployment whose status names the address it went
# live at. Only its GitHub App's account can create either.
MINTLIFY_CHECK = "Mintlify Deployment"
MINTLIFY_BOT = "mintlify[bot]"
ADDRESS_RE = re.compile(r"https://[a-z0-9.-]+(?:/[A-Za-z0-9._~-]+)*")
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
            f"{what} must name exactly {', '.join(names) or 'nothing'}: "
            f"missing {missing}, unexpected {unexpected}"
        )


def check_sources(sources: dict, site: Site = ASSEMBLED) -> dict[str, str]:
    """Every root of the site, each a clean commit, in order."""
    exactly(sources, site.roots, "sources")
    for name in site.roots:
        # The assembler marks a dirty root `<sha>+uncommitted` and a root
        # outside git `unversioned`: neither names what was built.
        if not COMMIT_RE.fullmatch(sources[name]):
            raise CandidateError(f"source {name}={sources[name]} is not a clean commit")
    return {name: sources[name] for name in site.roots}


def check_refs(refs: dict, site: Site = ASSEMBLED) -> dict[str, str]:
    """The ref each producer was read from."""
    exactly(refs, site.producers, "refs")
    for name in site.producers:
        if not REF_RE.fullmatch(refs[name]):
            raise CandidateError(f"ref {name}={refs[name]!r} is not a plain ref name")
    return {name: refs[name] for name in site.producers}


def parse_sources(text: str, site: Site = ASSEMBLED) -> dict[str, str]:
    """`lancedb=<sha> enterprise=<sha> build=<sha>`, as the assembler prints it."""
    return check_sources(parse_pairs(text, "sources"), site)


def parse_refs(text: str, site: Site = ASSEMBLED) -> dict[str, str]:
    return check_refs(parse_pairs(text, "refs"), site)


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


def record(
    site_dir: Path, sources_line: str, refs: list[str], output: Path, site: Site = ASSEMBLED
) -> dict:
    sources = parse_sources(sources_line, site)
    refs = parse_refs(" ".join(refs), site)
    hashes = file_hashes(site_dir)
    manifest = {
        "format": 1,
        "site": site.name,
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
    site = SITES[manifest["site"]]
    lines = [
        f"### Candidate of the {site.name} site",
        "",
        f"Site checksum `{manifest['site_sha256']}`: {manifest['files']} files.",
        "",
        "| Root | Commit | Ref |",
        "|---|---|---|",
    ]
    for name, commit in manifest["sources"].items():
        ref = run.get("ref", "") if name == DOCS_ROOT else manifest["refs"].get(name, "")
        lines.append(f"| {name} | `{commit}` | {ref} |")
    publish = (
        f"run the Publish workflow with target `staging`, run ID `{run.get('run_id', '?')}`"
        " and this checksum"
    )
    if site is VERSIONED:
        lines += [
            "",
            "Once staging shows the versioned site, Publish puts main's newest"
            f" versioned candidate there by itself. To publish this one by hand, {publish}."
            " Production takes it, with the same run ID and checksum, after approval and"
            " while staging shows it.",
        ]
    else:
        lines += ["", f"To put it on staging, {publish}. Production takes only the versioned site."]
    return "\n".join(lines + [""])


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


def tree_of(commit: str) -> str:
    return text("rev-parse", f"{commit}^{{tree}}")


def on_main(remote: str, commit: object, run_id: str) -> None:
    """The remote's `refs/heads/main` must contain `commit`: a run's head_branch
    is only a name, which a run of a tag named main has too."""
    if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
        raise CandidateError(f"run {run_id} names no commit it built")
    main = remote_head(remote, "main")
    if main is None:
        raise CandidateError(f"{remote} has no main branch")
    # A shallow checkout lacks the earlier commits of main.
    if text("rev-parse", "--is-shallow-repository") == "true":
        git("fetch", "--quiet", "--no-tags", "--unshallow", remote, "+refs/heads/main:refs/candidate-publish/main")
    if subprocess.run(["git", "merge-base", "--is-ancestor", commit, main], capture_output=True).returncode:
        raise CandidateError(
            f"the versioned site is published only from main; run {run_id} built {commit}, which main "
            "does not contain"
        )


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


def check_workflow(run: dict, run_id: str) -> Site:
    """The run's workflow must be this repository's Assemble or Compose file.

    The API gives its path bare or, as GitHub's REST reference shows, with the
    ref the file was read from: `.github/workflows/assemble.yml@main`. That ref
    must then be the run's own branch or commit. The file says which site the
    run built.
    """
    path = run.get("path")
    file, at, ref = path.partition("@") if isinstance(path, str) else ("", "", "")
    site = next((s for s in SITES.values() if s.workflow == file), None)
    if site is None or (at and not REF_RE.fullmatch(ref)):
        workflows = " or ".join(s.workflow for s in SITES.values())
        raise CandidateError(f"run {run_id} ran {path}, not {workflows}")
    branch = run.get("head_branch")
    own = {branch, f"refs/heads/{branch}", run.get("head_sha")} if branch else {run.get("head_sha")}
    if at and ref not in own - {None}:
        raise CandidateError(
            f"run {run_id} ran {file} from {ref}, not from its own branch "
            f"{branch} or commit {run.get('head_sha')}"
        )
    return site


def load_run(path: Path | None, run_id: str) -> dict:
    """The run as the API returns it."""
    try:
        run = json.loads(path.read_text()) if path else None
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateError(f"no evidence of run {run_id}: {exc}") from None
    if run is None:
        raise CandidateError(f"no evidence of run {run_id}")
    if not isinstance(run, dict) or not isinstance(run.get("repository"), dict):
        raise CandidateError(f"the evidence of run {run_id} is not a workflow run")
    return run


def check_run(run: dict, run_id: str, repository: str, target: str) -> Site:
    """The run must be a finished, successful run of this repository's Assemble
    or Compose workflow; returns the site it built."""
    if str(run.get("id")) != run_id:
        raise CandidateError(f"the run evidence describes run {run.get('id')}, not {run_id}")
    if run["repository"].get("full_name") != repository:
        raise CandidateError(f"run {run_id} is not a run of {repository}")
    site = check_workflow(run, run_id)
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        raise CandidateError(
            f"run {run_id} has not succeeded: {run.get('status')}, {run.get('conclusion')}"
        )
    if run.get("event") not in EVENTS:
        raise CandidateError(
            f"run {run_id} was a {run.get('event')} run; only {' and '.join(EVENTS)} "
            "runs check this repository's own workflow"
        )
    # Staging shows what production would get, so the versioned site comes from main.
    if site is VERSIONED and run.get("head_branch") != "main":
        raise CandidateError(
            f"the versioned site is published only from main; run {run_id} built "
            f"{run.get('head_branch')}"
        )
    if target == "production" and site is not VERSIONED:
        raise CandidateError(
            f"production takes only the versioned site; run {run_id} built the {site.name} site"
        )
    return site


def read_record(candidate: Path) -> dict:
    try:
        manifest = json.loads((candidate / "candidate.json").read_text())
    except FileNotFoundError:
        raise CandidateError(f"{candidate} holds no candidate.json") from None
    except json.JSONDecodeError as exc:
        raise CandidateError(f"candidate.json is not JSON: {exc}") from None
    if not isinstance(manifest, dict):
        raise CandidateError("candidate.json is not a record")
    return manifest


def check_products(site: Path) -> None:
    """The versioned site's navigation lists products, each with versions."""
    try:
        products = json.loads((site / "docs.json").read_text())["navigation"]["products"]
        listed = bool(products) and all(p["versions"] for p in products)
    except (OSError, ValueError, KeyError, TypeError):
        listed = False
    if not listed:
        raise CandidateError("the versioned candidate's docs.json lists no products and versions")


def check_artifact(candidate: Path, run: dict, expected: str, site: Site) -> tuple[dict, dict]:
    """The artifact's files must match its record, and the record this run."""
    manifest = read_record(candidate)
    recorded_by = manifest.get("run") if isinstance(manifest.get("run"), dict) else {}
    if str(recorded_by.get("run_id")) != str(run["id"]) or recorded_by.get(
        "repository"
    ) != run["repository"]["full_name"]:
        raise CandidateError(
            f"the candidate was recorded by {recorded_by.get('repository')} run "
            f"{recorded_by.get('run_id')}, not run {run['id']}"
        )
    # Records made before the versioned site existed do not name their site.
    recorded_site = manifest.get("site", ASSEMBLED.name)
    if recorded_site != site.name:
        raise CandidateError(
            f"the candidate is a record of the {recorded_site} site, but run {run['id']} "
            f"built the {site.name} site"
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
    if site is VERSIONED:
        check_products(candidate / "site")
    return check_sources(manifest.get("sources"), site), check_refs(manifest.get("refs"), site)


def has_artifact(candidate: Path | None) -> bool:
    return candidate is not None and candidate.is_dir() and any(candidate.iterdir())


def from_artifact(candidate: Path, run: dict, expected: str, site: Site, prefix: str):
    sources, refs = check_artifact(candidate, run, expected, site)
    paths = list(file_hashes(candidate / "site"))
    return write_tree(candidate / "site", paths, prefix), sources, refs


TRAILER = "^{}: (.+)$"


def trailer(message: str, key: str) -> str | None:
    found = re.findall(TRAILER.format(re.escape(key)), message, re.M)
    return found[-1].strip() if found else None


def parse_publication(commit: str, message: str) -> dict | None:
    """What a publication commit says it published; None for any other commit."""
    run, checksum = trailer(message, "Candidate-Run"), trailer(message, "Candidate-Site-SHA256")
    if run is None or checksum is None:
        return None
    return {
        "commit": commit,
        "run": run,
        "site_sha256": checksum,
        # Publications made before the versioned site existed do not name their site.
        "site": trailer(message, "Candidate-Site") or ASSEMBLED.name,
        "sources": trailer(message, "Sources") or "",
        "refs": trailer(message, "Refs") or "",
    }


def publication(commit: str | None) -> dict | None:
    if commit is None:
        return None
    return parse_publication(commit, text("log", "-1", "--format=%B", commit))


def earlier_publication(head: str | None, run_id: str, expected: str, site: Site):
    """The latest commit on the branch that published this run's candidate."""
    if head is None:
        return None
    for entry in git("log", "-z", "--first-parent", "--format=%H%n%B", head).decode().split("\0"):
        commit, _, message = entry.strip().partition("\n")
        found = parse_publication(commit, message)
        if found and found["run"] == run_id and found["site_sha256"] == expected:
            if found["site"] != site.name:
                raise CandidateError(
                    f"{commit} published candidate {expected} as the {found['site']} site, "
                    f"but run {run_id} built the {site.name} site"
                )
            sources = parse_sources(found["sources"], site)
            refs = parse_refs(found["refs"], site)
            return commit, sources, refs
    return None


def staging_address(value: str | None) -> str:
    """Where Mintlify serves staging, exactly as its deployments name it."""
    if not ADDRESS_RE.fullmatch(value or ""):
        raise CandidateError(
            f"no staging address {value!r}: set the repository variable MINTLIFY_STAGING_URL "
            "to the address Mintlify's deployments of staging name, such as "
            "https://example-staging.mintlify.site, with no trailing slash"
        )
    return value


def by_mintlify(entry) -> bool:
    creator = entry.get("creator") if isinstance(entry, dict) else None
    return (
        isinstance(creator, dict)
        and creator.get("login") == MINTLIFY_BOT
        and creator.get("type") == "Bot"
    )


def newest(entries: list[dict]) -> dict:
    return max(entries, key=lambda e: e["id"] if isinstance(e.get("id"), int) else -1)


def check_deployed(deployments: Path | None, commit: str, address: str) -> None:
    """Mintlify's newest deployment of `commit` must have put it live at `address`.

    A push is not a deployment, and the same commit can be deployed to another
    preview, so neither a successful check nor an older deployment is enough.
    """
    try:
        listed = json.loads(deployments.read_text()) if deployments else []
    except (OSError, ValueError):
        listed = []
    mintlify = [
        d for d in (listed if isinstance(listed, list) else [])
        if by_mintlify(d) and d.get("sha") == commit
    ]
    if not mintlify:
        raise CandidateError(
            f"Mintlify has not reported deploying staging commit {commit}: wait until its "
            f"'{MINTLIFY_CHECK}' check on that commit succeeds, review {address}, then run "
            "this again"
        )
    # Anyone who can write deployments can add a status, so only Mintlify's count.
    deployment = newest(mintlify)
    statuses = [s for s in deployment.get("statuses") or [] if by_mintlify(s)]
    status = newest(statuses) if statuses else {}
    state, live_at = status.get("state"), status.get("environment_url")
    if state != "success" or live_at != address:
        raise CandidateError(
            f"Mintlify's newest deployment of staging commit {commit} is "
            f"{state or 'unfinished'} at {live_at or 'no address'}, not live at {address}"
        )


def describe(found: dict | None) -> str:
    if found is None:
        return "no candidate"
    return f"candidate {found['site_sha256']} of the {found['site']} site from run {found['run']}"


def promotion(args: argparse.Namespace, run: dict, site: Site, expected: str, head: str | None):
    """Production takes a candidate it has published before, to roll back, or the
    candidate that staging shows once Mintlify has deployed it there."""
    # `assembled` takes only the production deploy key, so its history is evidence.
    earlier = earlier_publication(head, args.run_id, expected, site)
    if earlier is not None:
        commit, sources, refs = earlier
        return tree_of(commit), sources, refs, f"earlier publication {commit}"
    address = staging_address(args.staging_url)
    staging = remote_head(args.remote, TARGETS["staging"])
    staged = publication(staging)
    if staged is None or (staged["run"], staged["site_sha256"], staged["site"]) != (
        args.run_id, expected, site.name
    ):
        raise CandidateError(
            f"staging shows {describe(staged)}, not candidate {expected} from run "
            f"{args.run_id}, and production has not published that before: publish it to "
            "staging and review it there first"
        )
    # Anyone who can push to staging can write its trailers; only the run's own
    # record shows that the run built these files.
    if not has_artifact(args.candidate):
        raise CandidateError(
            f"run {args.run_id} has no candidate artifact, so nothing shows it built what "
            "staging shows; publish a newer candidate to staging"
        )
    sources, refs = check_artifact(args.candidate, run, expected, site)
    check_deployed(args.staging_deployments, staging, address)
    return tree_of(staging), sources, refs, f"staging {staging}"


def skipped(target: str, reason: str) -> dict:
    return {"target": target, "branch": TARGETS[target], "published": False, "reason": reason}


def shows(commit: str, prefix: str, expected: str) -> bool:
    """Whether the commit's files, not just its message, are the candidate."""
    try:
        return site_checksum(tree_hashes(tree_of(commit), prefix)) == expected
    except CandidateError:
        return False


def follow_main(args: argparse.Namespace, prefix: str) -> tuple | dict:
    """By itself, staging takes main's newest versioned candidate, and only once
    it shows the versioned site: the first one is published there by hand.

    The run that woke this is not trusted to be the newest; main is read now.
    """
    head = remote_head(args.remote, TARGETS["staging"])
    staged = publication(head)
    if staged is None or staged["site"] != VERSIONED.name:
        return skipped(
            "staging",
            f"staging shows {describe(staged)}; it follows main's versioned candidates by "
            "itself only once one is published there by hand",
        )
    main = remote_head(args.remote, "main")
    if not args.run_id:
        return skipped("staging", f"main's newest commit {main} has no successful Compose run yet")
    run = load_run(args.run, args.run_id)
    site = check_run(run, args.run_id, args.repository, "staging")
    if site is not VERSIONED:
        raise CandidateError(
            f"run {args.run_id} built the {site.name} site, not the versioned site"
        )
    if run.get("head_sha") != main:
        return skipped(
            "staging",
            f"run {args.run_id} built {run.get('head_sha')}, but main is now {main}; the "
            "Compose run of that commit publishes it",
        )
    if not has_artifact(args.candidate):
        raise CandidateError(
            f"run {args.run_id}, of main's newest commit, has no candidate artifact"
        )
    expected = read_record(args.candidate).get("site_sha256")
    if not isinstance(expected, str) or not CHECKSUM_RE.fullmatch(expected):
        raise CandidateError(f"the candidate of run {args.run_id} names no site checksum")
    check_artifact(args.candidate, run, expected, site)
    # A push to main that leaves the site as it was needs no new deployment. Only
    # staging's files show that: its message, sources and all, can sit on others.
    if staged["site_sha256"] == expected and shows(head, prefix, expected):
        return skipped(
            "staging",
            f"staging already shows candidate {expected}, from run {staged['run']}",
        )
    return run, site, expected, head


def publish(args: argparse.Namespace) -> dict:
    branch = TARGETS[args.target]
    prefix = content_prefix(args.content_dir)
    if args.automatic:
        followed = follow_main(args, prefix)
        if isinstance(followed, dict):
            return followed
        run, site, expected, head = followed
    else:
        if not re.fullmatch(r"[0-9]+", args.run_id):
            raise CandidateError(f"run ID {args.run_id!r} is not a number")
        expected = args.site_sha256.strip()
        if not CHECKSUM_RE.fullmatch(expected):
            raise CandidateError(f"{args.site_sha256!r} is not a lowercase SHA-256")
        run = load_run(args.run, args.run_id)
        site = check_run(run, args.run_id, args.repository, args.target)
        if site is VERSIONED:
            on_main(args.remote, run.get("head_sha"), args.run_id)
        head = remote_head(args.remote, branch)

    if args.target == "production":
        # What production was when the run was requested, before its approval.
        if args.production_was is None:
            raise CandidateError(
                "production needs --production-was: the `assembled` commit when the run was "
                "requested"
            )
        if (head or "") != args.production_was:
            raise CandidateError(
                f"production has changed since this run was requested: it was "
                f"{args.production_was or 'empty'} and is now {head}; start a new run to publish "
                "this candidate"
            )
        tree, sources, refs, origin = promotion(args, run, site, expected, head)
    elif has_artifact(args.candidate):
        tree, sources, refs = from_artifact(args.candidate, run, expected, site, prefix)
        origin = "the run's artifact"
    else:
        earlier = earlier_publication(head, args.run_id, expected, site)
        if earlier is None:
            raise CandidateError(
                f"run {args.run_id} has no candidate artifact, and {branch} has no "
                f"earlier publication of candidate {expected} from it"
            )
        commit, sources, refs = earlier
        tree = tree_of(commit)
        origin = f"earlier publication {commit}"

    if sources[DOCS_ROOT] != run.get("head_sha"):
        raise CandidateError(
            f"the candidate was built from {DOCS_ROOT} {sources[DOCS_ROOT]}, "
            f"but run {args.run_id} checked out {run.get('head_sha')}"
        )

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
        f"Candidate-Site: {site.name}",
        "Sources: " + " ".join(f"{k}={v}" for k, v in sources.items()),
    ]
    if refs:
        lines.append("Refs: " + " ".join(f"{k}={v}" for k, v in refs.items()))
    lines.append(f"Published-From: {origin}")
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
        "published": True,
        "commit": commit,
        "parent": head,
        "site": site.name,
        "site_sha256": expected,
        "files": len(published),
        "run_id": args.run_id,
        "sources": sources,
        "refs": refs,
        "from": origin,
    }


def publish_summary(result: dict) -> str:
    if not result["published"]:
        return f"Nothing published to `{result['branch']}`: {result['reason']}.\n"
    return (
        f"Published candidate `{result['site_sha256']}` of the {result['site']} site from run "
        f"{result['run_id']} ({result['from']}) to `{result['branch']}` as "
        f"`{result['commit']}`. Mintlify deploys it next: the commit's "
        f"'{MINTLIFY_CHECK}' check says when the site shows it.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    commands = parser.add_subparsers(dest="command", required=True)

    rec = commands.add_parser("record", help="record a checked site as a candidate")
    rec.add_argument("site", type=Path)
    rec.add_argument("--sources", required=True, help="the build's sources line")
    rec.add_argument("--ref", action="append", default=[], metavar="ROOT=REF")
    rec.add_argument("--site", dest="kind", choices=SITES, default=ASSEMBLED.name)
    rec.add_argument("--output", type=Path, required=True)
    rec.add_argument("--summary", type=Path, help="append a Markdown summary to this file")

    pub = commands.add_parser("publish", help="publish one recorded candidate")
    pub.add_argument("--target", choices=TARGETS, required=True)
    pub.add_argument(
        "--automatic", action="store_true",
        help="staging only: publish main's newest versioned candidate, if staging follows main",
    )
    pub.add_argument("--run-id", default="")
    pub.add_argument("--site-sha256", default="")
    pub.add_argument("--run", type=Path, help="the run as the API returns it")
    pub.add_argument("--repository", required=True, help="OWNER/NAME of this repository")
    pub.add_argument("--content-dir", required=True)
    pub.add_argument("--candidate", type=Path, help="the downloaded candidate artifact")
    pub.add_argument(
        "--staging-deployments", type=Path,
        help="production: the deployments of staging's commit, as the API returns them, "
        "each with its `statuses`",
    )
    pub.add_argument(
        "--staging-url", help="production: the address Mintlify serves staging at"
    )
    pub.add_argument(
        "--production-was", metavar="COMMIT",
        help="production: the `assembled` commit when the run was requested, or empty",
    )
    pub.add_argument("--remote", default="origin")
    pub.add_argument("--summary", type=Path, help="append a Markdown summary to this file")
    args = parser.parse_args()

    try:
        if args.command == "record":
            manifest = record(args.site, args.sources, args.ref, args.output, SITES[args.kind])
            print(
                f"candidate {manifest['site_sha256']}: {manifest['files']} files of the "
                f"{args.kind} site from "
                + " ".join(f"{k}={v}" for k, v in manifest["sources"].items())
            )
            if args.summary:
                with args.summary.open("a") as summary:
                    summary.write(record_summary(manifest))
            return 0
        if args.automatic and args.target != "staging":
            raise CandidateError("only staging is published automatically")
        result = publish(args)
    except CandidateError as exc:
        print(f"candidate: refused: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    if args.summary:
        with args.summary.open("a") as summary:
            summary.write(publish_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
