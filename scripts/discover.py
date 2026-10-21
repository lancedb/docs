"""
Find each product's eligible releases, and prepare a documentation pull
request for each one that has no saved edition yet.

An Enterprise release is eligible once it is published and qualified: its
`vX.Y.Z` tag, which the Enterprise release pipeline pushes last, and an
annotated `release-qualified/vX.Y.Z` tag name the same commit. An OSS release
is eligible once its GitHub release is published and not a prerelease.
discover.yaml names the first release of each product, since there is no
backfill. Editions are prepared by `edition.py prepare`, as by hand. See
"Discovering releases" in README.md.

Usage:

    python scripts/discover.py [--lancedb REPO] [--sophon REPO] [--config FILE] [--summary FILE]
    python scripts/discover.py --open-prs [--lancedb REPO] [--sophon REPO] [--summary FILE]

Without --open-prs discovery only reports: it fetches release tags and commits
into the two repositories, and pushes, opens and starts nothing. --open-prs is
for the Discover releases workflow, on main.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compose  # noqa: E402
import edition  # noqa: E402

REPO_ROOT = edition.REPO_ROOT
CONFIG = "discover.yaml"
DOCS = edition.SOURCES["build"][0]
LANCEDB = edition.SOURCES["lancedb"][0]
NAMES = {"enterprise": "Enterprise", "oss": "OSS"}
BRANCH = "editions/{product}-{version}"
PRERELEASE_RE = re.compile(r"v(\d+\.\d+\.\d+)-[0-9A-Za-z.-]+")
# Pushes and pull requests made with the workflow's token start no workflows,
# so discovery starts the pull request checks on its branch itself.
CHECKS = {"compose.yml": "Compose the versioned site", "assemble.yml": "Test the assembler"}
BOT = ("github-actions[bot]", "41898282+github-actions[bot]@users.noreply.github.com")
# The report's sections, in order.
STATES = {
    "prepare": "To prepare",
    "resume": "Pushed, pull request not opened",
    "open": "Pull request open",
    "saved": "Saved",
    "declined": "Closed without merging",
    "waiting": "Not eligible yet",
    "skipped": "Not prepared",
    "blocked": "Stopped for review",
}


class DiscoveryError(Exception):
    """Something that stops discovery, or one release, with its reason."""


class Waiting(DiscoveryError):
    """A release that becomes ready without anyone acting on it."""


@dataclass(frozen=True)
class Tag:
    object: str
    commit: str


@dataclass
class Finding:
    """A release, or a group of them, and what discovery does about it."""

    product: str
    version: str
    state: str
    reason: str
    commit: str | None = None
    qualification: str | None = None
    pages: tuple[str, str] | None = None
    why: str = ""
    record: dict | None = None
    preferred: str = ""
    pull: dict | None = None
    notes: list[str] = field(default_factory=list)
    done: str | None = None
    failed: bool = False

    @property
    def name(self) -> str:
        return f"{NAMES[self.product]} {self.version}".strip()

    @property
    def branch(self) -> str:
        return BRANCH.format(product=self.product, version=self.version)


@dataclass
class Main:
    """lancedb/docs main as discovery read it, checked out at `path`."""

    commit: str
    path: Path
    records: dict
    problems: dict
    preferred: dict


def key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


# configuration


def load_config(path: Path) -> dict:
    where = path.name
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DiscoveryError(f"{where}: unreadable: {exc}") from None
    if not isinstance(raw, dict) or set(raw) != {"products", "holds"}:
        raise DiscoveryError(f"{where} sets `products` and `holds`, and nothing else")
    products = raw["products"]
    if not isinstance(products, dict) or set(products) != set(edition.PRODUCTS):
        raise DiscoveryError(f"{where}: products sets {' and '.join(edition.PRODUCTS)}, once each")
    settings = {}
    for name in edition.PRODUCTS:
        product = products[name]
        allowed = {"from", "skip"} | ({"enterprise_pages"} if name == "oss" else set())
        if not isinstance(product, dict) or not set(product) <= allowed:
            raise DiscoveryError(f"{where}: {name} sets only {', '.join(sorted(allowed))}")
        start = product.get("from")
        if start is not None and not (isinstance(start, str) and edition.VERSION_RE.fullmatch(start)):
            raise DiscoveryError(f"{where}: {name}.from is {start!r}, not a version such as 1.2.3")
        skip = product.get("skip") or {}
        if not isinstance(skip, dict) or not all(
            isinstance(v, str) and edition.VERSION_RE.fullmatch(v) and isinstance(r, str) and r.strip()
            for v, r in skip.items()
        ):
            raise DiscoveryError(f"{where}: {name}.skip maps versions such as 1.2.3 to why discovery leaves them alone")
        settings[name] = {"from": start, "skip": skip, "enterprise_pages": product.get("enterprise_pages")}
    pages = settings["oss"]["enterprise_pages"]
    if pages not in (None, "preferred"):
        raise DiscoveryError(f"{where}: oss.enterprise_pages is {pages!r}, not `preferred`")
    if settings["oss"]["from"] and pages is None:
        raise DiscoveryError(
            f"{where}: oss.from is set, so set oss.enterprise_pages too: an OSS edition includes the "
            "Enterprise pages, and discovery does not choose where they come from"
        )
    holds = raw["holds"] or []
    if not isinstance(holds, list) or not all(isinstance(h, str) and h.strip() for h in holds):
        raise DiscoveryError(f"{where}: holds lists one line of text for each hold")
    return {"products": settings, "holds": holds}


# git


def remote_tags(repo: Path, *patterns: str) -> dict[str, Tag]:
    """The tags `repo`'s origin has now, by name."""
    objects, peeled = {}, {}
    for line in edition.text(repo, "ls-remote", "--tags", "origin", *patterns).splitlines():
        sha, ref = line.split("\t")
        name = ref.removeprefix("refs/tags/")
        if name.endswith("^{}"):
            peeled[name[:-3]] = sha
        else:
            objects[name] = sha
    return {name: Tag(sha, peeled.get(name, sha)) for name, sha in objects.items()}


def remote_heads(repo: Path) -> dict[str, str]:
    """Discovery's branches as `repo`'s origin has them now."""
    listing = edition.text(repo, "ls-remote", "--heads", "origin", "refs/heads/editions/*")
    return {ref.removeprefix("refs/heads/"): sha for sha, ref in (line.split("\t") for line in listing.splitlines())}


def fetch(repo: Path, *refspecs: str) -> None:
    """Fetch from origin. A new repository, as the workflow makes, is kept
    shallow and blob-less: preparation fetches the blobs it reads on demand."""
    new = not edition.text(repo, "for-each-ref", "--count=1")
    shallow = new or edition.text(repo, "rev-parse", "--is-shallow-repository") == "true"
    promisor = subprocess.run(["git", "-C", str(repo), "config", "--get", "remote.origin.promisor"],
                              capture_output=True).stdout.strip() == b"true"
    edition.git(repo, "fetch", "--quiet", "--no-tags", *(["--depth=1"] if shallow else []),
                *(["--filter=blob:none"] if new or promisor else []), "origin", *refspecs)


def tag_refspecs(*names: str) -> list[str]:
    return [f"+refs/tags/{name}:refs/tags/{name}" for name in names]


def prefetch(repo: Path, commit: str, folder: str) -> None:
    """Fetch a folder's missing blobs at once; reading them would fetch each on its own."""
    listing = edition.git(repo, "ls-tree", "-r", "-z", commit, "--", folder).decode()
    blobs = [entry.split()[2] for entry in listing.split("\0") if entry and entry.split()[1] == "blob"]
    present = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "--batch-check"], input="".join(f"{b}\n" for b in blobs).encode(),
        env={**os.environ, "GIT_NO_LAZY_FETCH": "1"}, capture_output=True,
    ).stdout.decode()
    missing = [line.split()[0] for line in present.splitlines() if line.endswith(" missing")]
    if missing:
        # How git itself fetches a partial clone's missing objects.
        edition.git(repo, "-c", "fetch.negotiationAlgorithm=noop", "fetch", "--quiet", "--no-tags",
                    "--no-write-fetch-head", "--recurse-submodules=no", "--filter=blob:none", "origin", "--stdin",
                    data="".join(f"{b}\n" for b in missing).encode())


def add_worktree(repo: Path, commit: str) -> Path:
    path = Path(tempfile.mkdtemp(prefix="discover-")).resolve()
    path.rmdir()
    edition.git(repo, "worktree", "add", "--quiet", "--detach", str(path), commit)
    return path


def remove_worktree(repo: Path, path: Path) -> None:
    subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(path)], capture_output=True)
    shutil.rmtree(path, ignore_errors=True)
    subprocess.run(["git", "-C", str(repo), "worktree", "prune"], capture_output=True)


# GitHub


class GitHub:
    """GitHub's REST API through `gh`, which authenticates with GH_TOKEN. Only --open-prs writes."""

    def __init__(self, writes: bool = False):
        self.writes = writes

    def get(self, endpoint: str):
        """The response, or None if there is no such thing."""
        return self._call("GET", endpoint)

    def post(self, endpoint: str, payload: dict):
        if not self.writes:
            raise DiscoveryError(f"a dry run writes nothing, so it did not post to {endpoint}")
        return self._call("POST", endpoint, payload)

    def _call(self, method: str, endpoint: str, payload: dict | None = None):
        data = None if payload is None else json.dumps(payload).encode()
        result = subprocess.run(
            ["gh", "api", "--method", method, endpoint, *([] if data is None else ["--input", "-"])],
            input=data, capture_output=True,
        )
        if result.returncode:
            message = result.stderr.decode(errors="replace").strip()
            if method == "GET" and "(HTTP 404)" in message:
                return None
            raise DiscoveryError(f"GitHub {method} {endpoint.split('?')[0]}: {message}")
        return json.loads(result.stdout) if result.stdout.strip() else None


def pulls(github: GitHub, branch: str) -> list[dict]:
    """Pull requests from `branch`, newest first."""
    owner = DOCS.split("/")[0]
    found = github.get(f"repos/{DOCS}/pulls?state=all&head={owner}:{branch}&per_page=100") or []
    return sorted(found, key=lambda pull: pull["number"], reverse=True)


def start_checks(github: GitHub, branch: str, workflows) -> None:
    """Start each workflow on `branch`; one that fails to start does not stop the others."""
    failed = []
    for workflow in workflows:
        try:
            github.post(f"repos/{DOCS}/actions/workflows/{workflow}/dispatches", {"ref": branch})
        except DiscoveryError as exc:
            failed.append(f"{workflow} ({exc})")
    if failed:
        raise DiscoveryError(f"could not start {', '.join(failed)} on {branch}; the next run starts what has not run")


def unstarted(github: GitHub, commit: str) -> list[str]:
    """The check workflows with no run of `commit`, whatever became of the runs there are."""
    return [workflow for workflow in CHECKS if not (
        github.get(f"repos/{DOCS}/actions/workflows/{workflow}/runs?head_sha={commit}&per_page=1") or {}
    ).get("total_count")]


# releases


def stable_and_prereleases(tags: dict[str, Tag], start: str) -> tuple[dict[str, Tag], int]:
    """Stable `vX.Y.Z` tags by version, and how many prerelease tags are from `start` on."""
    stable, prereleases = {}, 0
    for name, tag in tags.items():
        if name.startswith("v") and edition.VERSION_RE.fullmatch(name[1:]):
            stable[name[1:]] = tag
        elif (match := PRERELEASE_RE.fullmatch(name)) and key(match[1]) >= key(start):
            prereleases += 1
    return stable, prereleases


def summaries(product: str, earlier: list[str], prereleases: int, start: str) -> list[Finding]:
    found = []
    if earlier:
        span = earlier[0] if len(earlier) == 1 else f"{earlier[0]} to {earlier[-1]}"
        found.append(Finding(product, "", "skipped", (
            f"{plural(len(earlier), 'release')} before {start} ({span}): discovery starts at {start}, with no "
            "backfill"
        )))
    if prereleases:
        # Counted, not named: prerelease names can tell of releases to come.
        found.append(Finding(product, "", "skipped", (
            f"{plural(prereleases, 'prerelease')} from {start} on: only stable releases get editions"
        )))
    return found


def list_enterprise(tags: dict[str, Tag], start: str) -> list[Finding]:
    releases, prereleases = stable_and_prereleases(tags, start)
    qualified = {}
    for name, tag in tags.items():
        version = name.removeprefix("release-qualified/v")
        if version != name and edition.VERSION_RE.fullmatch(version):
            qualified[version] = tag
    found, earlier = [], []
    for version in sorted(set(releases) | set(qualified), key=key):
        if key(version) < key(start):
            earlier.append(version)
            continue
        release, qualification = releases.get(version), qualified.get(version)
        tag, qtag = f"v{version}", edition.QUALIFIED_TAG.format(version=version)
        if release is None:
            found.append(Finding("enterprise", version, "waiting", (
                f"{qtag} exists, but {tag} does not yet: the Enterprise release pipeline tags a release last, "
                "once it is published")))
            continue
        # Preparation checks the qualification tag itself, as by hand.
        why = f"{tag} and the annotated {qtag} tag name the same commit, so the release is published and qualified"
        found.append(Finding("enterprise", version, "prepare", why, commit=release.commit,
                             qualification=qualification.object if qualification else None, why=why))
    return found + summaries("enterprise", earlier, prereleases, start)


def list_oss(tags: dict[str, Tag], start: str, github: GitHub) -> list[Finding]:
    releases, prereleases = stable_and_prereleases(tags, start)
    found, earlier = [], []
    for version in sorted(releases, key=key):
        if key(version) < key(start):
            earlier.append(version)
            continue
        finding = Finding("oss", version, "prepare", "", commit=releases[version].commit)
        published = github.get(f"repos/{LANCEDB}/releases/tags/v{version}")
        if published is None or published.get("draft"):
            finding.state, finding.reason = "waiting", f"v{version} has no published GitHub release yet"
        elif published.get("prerelease"):
            finding.state, finding.reason = "skipped", f"the GitHub release v{version} is marked as a prerelease"
        else:
            finding.reason = finding.why = (
                f"the GitHub release v{version} of {LANCEDB} is published and not a prerelease")
        found.append(finding)
    return found + summaries("oss", earlier, prereleases, start)


# lancedb/docs main, branches and pull requests


def read_main(path: Path, commit: str) -> Main:
    records, problems = {}, {}
    for file in sorted((path / "releases").glob("*.yaml")):
        product, _, version = file.stem.partition("-")
        try:
            record = edition.load_record(file)
        except edition.EditionError as exc:
            problems[(product, version)] = f"main's record is unreadable: {exc}"
            continue
        records[(record["product"], record["version"])] = record
    try:
        config = compose.load_config(path / compose.CONFIG)
    except compose.ComposeError as exc:
        raise DiscoveryError(f"on main, {exc}") from None
    preferred = {product["product"]: product.get("preferred") for product in config["products"]}
    return Main(commit, path, records, problems, preferred)


def fixed(record: object) -> tuple:
    """What a release's tags fix in its record. The rest follows from them,
    except an OSS edition's Enterprise pages, chosen when it is prepared."""
    release = record.get("release") if isinstance(record, dict) else None
    if not isinstance(release, dict):
        return None, None
    qualification = release.get("qualification")
    return release.get("commit"), qualification.get("object") if isinstance(qualification, dict) else None


def differences(record: object, finding: Finding) -> str:
    commit, qualification = fixed(record)
    if commit is None:
        return "it has no readable record"
    changes = []
    if commit != finding.commit:
        changes.append(f"v{finding.version} was {commit} and is now {finding.commit}")
    if qualification != finding.qualification:
        qtag = edition.QUALIFIED_TAG.format(version=finding.version)
        changes.append(f"{qtag} was tag object {qualification} and is now {finding.qualification or 'missing'}")
    return "; ".join(changes)


def branch_file(repo: Path, commit: str, path: str) -> object:
    """A YAML file at a commit of a discovery branch, or None."""
    try:
        fetch(repo, commit)
        return yaml.safe_load(edition.git(repo, "show", f"{commit}:{path}"))
    except (edition.EditionError, yaml.YAMLError):
        return None


def record_file(finding: Finding) -> str:
    return f"releases/{finding.product}-{finding.version}.yaml"


def classify(finding: Finding, main: Main, github: GitHub, repo: Path, heads: dict, skip: dict) -> None:
    """Settle what to do about a listed release from main, its pull requests and its branch."""
    version = finding.version
    if version in skip:
        finding.state, finding.reason = "skipped", f"{CONFIG} skips it: {skip[version]}"
        return
    if (finding.product, version) in main.problems:
        finding.state, finding.reason = "blocked", main.problems[(finding.product, version)]
        return
    saved = main.records.get((finding.product, version))
    if saved:
        if finding.state != "prepare":
            finding.state, finding.reason = "blocked", f"saved, but now {finding.reason}"
        elif fixed(saved) != (finding.commit, finding.qualification):
            finding.state = "blocked"
            finding.reason = (f"saved from other release inputs: {differences(saved, finding)}. An edition is "
                              "never replaced, so review the release")
        else:
            count = len(saved["corrections"])
            finding.state = "saved"
            finding.reason = "saved from these release inputs" + (
                f", with {plural(count, 'correction')}" if count else "")
        return
    found = pulls(github, finding.branch)
    open_pull = next((pull for pull in found if pull["state"] == "open"), None)
    if finding.state != "prepare":
        if open_pull:
            finding.state, finding.pull = "blocked", open_pull
            finding.reason = f"pull request #{open_pull['number']} is open, but {finding.reason}"
        return
    if open_pull:
        finding.pull = open_pull
        record = branch_file(repo, open_pull["head"]["sha"], record_file(finding))
        if fixed(record) == (finding.commit, finding.qualification):
            finding.state, finding.reason = "open", f"pull request #{open_pull['number']} is open for review"
        else:
            finding.state = "blocked"
            finding.reason = (f"pull request #{open_pull['number']} does not hold the edition of these release "
                              f"inputs: {differences(record, finding)}")
        return
    if found:
        finding.pull = found[0]
        if found[0].get("merged_at"):
            finding.state = "blocked"
            finding.reason = (f"pull request #{found[0]['number']} was merged, but main has no record of "
                              "this release; review what removed it")
        else:
            finding.state = "declined"
            finding.reason = (f"pull request #{found[0]['number']} was closed without merging; reopen it "
                              "to prepare this release again")
        return
    if finding.branch in heads:
        record = branch_file(repo, heads[finding.branch], record_file(finding))
        if fixed(record) == (finding.commit, finding.qualification):
            finding.state = "resume"
            finding.reason = f"{finding.branch} was pushed, but its pull request was not opened"
        else:
            finding.state = "blocked"
            finding.reason = (f"{finding.branch} exists without a pull request, and does not hold the edition "
                              "of these release inputs; review or delete it")


# preparation


def enterprise_pages(main: Main, tags: dict[str, Tag]) -> tuple[str, str]:
    """The tag and commit of the preferred Enterprise edition's release."""
    version = main.preferred.get("enterprise")
    if version is None:
        raise Waiting("an OSS edition takes its Enterprise pages from the preferred Enterprise edition, and "
                      "compose.yaml names none yet")
    record = main.records.get(("enterprise", version))
    if record is None:
        raise DiscoveryError(f"compose.yaml prefers Enterprise {version}, but main has no readable record of it")
    ref, commit = record["release"]["tag"], record["release"]["commit"]
    tag = tags.get(ref)
    if tag is None or tag.commit != commit:
        raise DiscoveryError(
            f"the preferred Enterprise edition {version} was saved from {commit}, but {ref} now names "
            f"{tag.commit if tag else 'nothing'}; review it before preparing OSS editions with its pages"
        )
    return ref, commit


def prepare_args(finding: Finding, main: Main, lancedb: Path, sophon: Path) -> list[str]:
    return [
        "prepare", finding.product, finding.version, *(["--enterprise", finding.pages[0]] if finding.pages else []),
        "--lancedb", str(lancedb), "--sophon", str(sophon), "--docs", main.commit,
    ]


def unmoved(finding: Finding, record: dict, where: str) -> None:
    """Stop unless `record` reads the tags as discovery checked them: the
    release, its qualification, and the Enterprise pages as their edition
    records them."""
    release = record["release"]
    read = [(f"v{finding.version}", release.get("commit"), finding.commit)]
    if finding.qualification:
        read.append((edition.QUALIFIED_TAG.format(version=finding.version),
                     (release.get("qualification") or {}).get("object"), finding.qualification))
    if finding.pages:
        read.append((finding.pages[0], (release.get("enterprise_pages") or {}).get("commit"), finding.pages[1]))
    moved = [f"{name} at {found or 'nothing'} rather than {checked}"
             for name, found, checked in read if found != checked]
    if moved:
        raise DiscoveryError(f"{where} {'; '.join(moved)}: a tag moved while discovery ran, so nothing is prepared "
                             "from it. Review the release")


def resolve(finding: Finding, main: Main, lancedb: Path, sophon: Path, sophon_tags) -> None:
    """Fetch what preparation reads, and the record it would write, from the
    tags as discovery checked them."""
    tag = f"v{finding.version}"
    if finding.product == "enterprise":
        qualified = [edition.QUALIFIED_TAG.format(version=finding.version)] if finding.qualification else []
        fetch(sophon, *tag_refspecs(tag, *qualified))
        # What the fetched tag bundles; if the tag has moved, the record shows it below.
        bundled = edition.bundled_lancedb(sophon, f"refs/tags/{tag}^{{commit}}", f"lancedb/sophon {tag}")
        try:
            fetch(lancedb, bundled)
        except edition.EditionError as exc:
            raise DiscoveryError(f"{tag} bundles LanceDB {bundled}, which {LANCEDB} does not serve: {exc}") from None
    else:
        finding.pages = enterprise_pages(main, sophon_tags())
        fetch(lancedb, *tag_refspecs(tag))
        fetch(sophon, *tag_refspecs(finding.pages[0]))
    record = edition.resolve_release(edition.parse(prepare_args(finding, main, lancedb, sophon)), main.path)
    unmoved(finding, record, "preparation would read")
    for name, repo in (("lancedb", lancedb), ("enterprise", sophon)):
        source = record["sources"][name]
        if not edition.text(repo, "ls-tree", "-d", source["commit"], "--", source["path"]):
            raise DiscoveryError(
                f"{source['repository']} {source['commit']} has no {source['path']}: an edition is built only "
                "from a release's own documentation sources"
            )
    finding.record = record


def preferred_note(main: Main, finding: Finding) -> tuple[bool, str]:
    """Whether the pull request sets the preferred edition, and what it says about it."""
    current, name = main.preferred.get(finding.product), NAMES[finding.product]
    if current is None:
        return True, (f"sets the preferred {name} edition to {finding.version} in compose.yaml, since Compose "
                      f"needs one once {name} has an edition; review it")
    if key(finding.version) > key(current):
        return False, (f"leaves the preferred {name} edition at {current}. This release is newer: to make it "
                       "the default, change `preferred` in compose.yaml in this pull request")
    return False, f"leaves the preferred {name} edition at {current}, a newer release"


def set_preferred(path: Path, product: str, version: str) -> None:
    """Set an unset `preferred` in compose.yaml, keeping its comments."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if line.strip() == f"- product: {product}"), len(lines))
    for i in range(start + 1, len(lines)):
        if lines[i].lstrip().startswith("- product:"):
            break
        if lines[i].strip() == "preferred:":
            lines[i] = lines[i].replace("preferred:", f"preferred: {version}", 1)
            path.write_text("".join(lines), encoding="utf-8")
            break
    products = compose.load_config(path)["products"]
    if next(p for p in products if p["product"] == product).get("preferred") != version:
        raise DiscoveryError(f"could not set the preferred {NAMES[product]} edition in compose.yaml; set it by hand")


def prepare_branch(finding: Finding, main: Main, lancedb: Path, sophon: Path, repo: Path) -> str:
    """The commit that saves the edition on main, made as by hand from the
    sources discovery checked."""
    for name, source_repo in (("lancedb", lancedb), ("enterprise", sophon)):
        source = finding.record["sources"][name]
        prefetch(source_repo, source["commit"], source["path"])
    path = add_worktree(repo, main.commit)
    try:
        edition.prepare(edition.parse(prepare_args(finding, main, lancedb, sophon)), path)
        record = edition.load_record(path / record_file(finding))
        unmoved(finding, record, "preparation read")
        if preferred_note(main, finding)[0]:
            set_preferred(path / compose.CONFIG, finding.product, finding.version)
        _lines, problems = edition.check(path, base=main.commit)
        if problems:
            raise DiscoveryError("`edition.py check` fails on the new branch: " + "; ".join(problems))
        finding.record = record
        edition.git(path, "add", "--all", "--", "releases", "editions", compose.CONFIG)
        bot = {"GIT_AUTHOR_NAME": BOT[0], "GIT_AUTHOR_EMAIL": BOT[1], "GIT_COMMITTER_NAME": BOT[0],
               "GIT_COMMITTER_EMAIL": BOT[1]}
        command = " ".join(["edition.py prepare", finding.product, finding.version,
                            *(["--enterprise", finding.pages[0]] if finding.pages else [])])
        result = subprocess.run(
            ["git", "-C", str(path), "commit", "--quiet", "-m", title(finding),
             "-m", f"Prepared by release discovery with `{command}`."],
            env={**os.environ, **bot}, capture_output=True,
        )
        if result.returncode:
            raise DiscoveryError(f"could not commit the edition: {result.stderr.decode(errors='replace').strip()}")
        return edition.text(path, "rev-parse", "HEAD")
    finally:
        remove_worktree(repo, path)


def push(repo: Path, commit: str, branch: str) -> bool:
    """Create `branch` at `commit`; False if it exists, since it may hold someone's work."""
    result = subprocess.run(
        ["git", "-C", str(repo), "push", "--quiet", f"--force-with-lease=refs/heads/{branch}:", "origin",
         f"{commit}:refs/heads/{branch}"],
        capture_output=True,
    )
    if result.returncode == 0:
        return True
    if branch in remote_heads(repo):
        return False
    raise DiscoveryError(f"could not push {branch}: {result.stderr.decode(errors='replace').strip()}")


# pull requests


def title(finding: Finding) -> str:
    return f"docs: save the {finding.name} edition"


def inputs(record: dict) -> list[str]:
    release, sources = record["release"], record["sources"]
    lines = [f"release `{release['tag']}` at `{release['commit']}`"]
    if "qualification" in release:
        lines.append(f"qualification `{release['qualification']['tag']}`, tag object "
                     f"`{release['qualification']['object']}`")
    if "bundled_lancedb" in release:
        lines.append(f"bundled LanceDB `{release['bundled_lancedb']}`")
    if "enterprise_pages" in release:
        lines.append(f"Enterprise pages from `{release['enterprise_pages']['ref']}` at "
                     f"`{release['enterprise_pages']['commit']}`")
    lines.append(f"docs `main` at `{sources['build']['commit']}`")
    return lines


def body(finding: Finding, record: dict, preferred: str, holds: list[str], resumed: bool) -> str:
    prepared = record["prepared"]
    built = f"{prepared['files']} files, checksum `{prepared['sha256']}`"
    checks = (
        f"An earlier discovery run prepared it ({built}) and pushed this branch, but did not open this pull "
        "request." if resumed else
        f"`edition.py prepare` built it with the assembler ({built}), and `edition.py check` passed against main."
    )
    lines = [
        f"Release discovery prepared the {finding.name} edition for review. Merging it saves the edition; the "
        "bot does not merge or publish.",
        "",
        f"- **Eligible:** {finding.why}.",
        f"- **Sources**, as `{record_file(finding)}` records them:",
        *[f"  - {line}" for line in inputs(record)],
        f"- **Checks:** {checks} Discovery started Compose and Assemble on this branch, whose pushes start "
        "no checks by themselves; their results are below.",
        f"- **Preferred edition:** {preferred[0].upper()}{preferred[1:]}.",
        "- **Publication holds:**" + ("".join(f"\n  - {hold}" for hold in holds) if holds else f" none in {CONFIG}."),
    ]
    return "\n".join(lines) + "\n"


def open_pull(github: GitHub, finding: Finding, text: str) -> dict:
    try:
        return github.post(f"repos/{DOCS}/pulls", {
            "title": title(finding), "head": finding.branch, "base": "main", "body": text,
        })
    except DiscoveryError:
        # Another run may have opened it meanwhile.
        existing = next((pull for pull in pulls(github, finding.branch) if pull["state"] == "open"), None)
        if existing is None:
            raise
        return existing


def act(finding: Finding, main: Main, repo: Path, lancedb: Path, sophon: Path, github: GitHub,
        holds: list[str]) -> None:
    """Prepare, push and open what the finding needs, picking up where an earlier run stopped."""
    resumed = finding.state == "resume"
    if finding.state == "prepare":
        commit = prepare_branch(finding, main, lancedb, sophon, repo)
        if not push(repo, commit, finding.branch):
            finding.notes.append(f"{finding.branch} appeared while this run prepared it, so its commits are kept")
            classify(finding, main, github, repo, remote_heads(repo), {})
            if finding.state in ("resume", "open"):
                act(finding, main, repo, lancedb, sophon, github, holds)
            return
    if finding.state in ("prepare", "resume"):
        commit = remote_heads(repo).get(finding.branch)
        record = finding.record if not resumed else branch_file(repo, commit, record_file(finding))
        if not isinstance(record, dict) or "prepared" not in record:
            raise DiscoveryError(f"{finding.branch} has no readable record")
        preferred = preferred_note(main, finding)[1]
        if resumed:
            # What the branch holds, which main may have overtaken since.
            products = (branch_file(repo, commit, compose.CONFIG) or {}).get("products") or []
            wanted = next((p.get("preferred") for p in products
                           if isinstance(p, dict) and p.get("product") == finding.product), None)
            if wanted != main.preferred.get(finding.product):
                name = NAMES[finding.product]
                preferred = f"sets the preferred {name} edition to {wanted} in compose.yaml; review it"
        pull = open_pull(github, finding, body(finding, record, preferred, holds, resumed))
        finding.state, finding.pull, finding.done = "open", pull, f"opened pull request #{pull['number']}"
        start_checks(github, finding.branch, CHECKS)
        finding.done += " and started its checks"
    elif finding.state == "open" and (missing := unstarted(github, finding.pull["head"]["sha"])):
        start_checks(github, finding.branch, missing)
        finding.done = f"started {' and '.join(missing)}, which had not run on its head"


# discovery


def discover(repo: Path, lancedb: Path, sophon: Path, github: GitHub, config: Path | None = None,
             open_prs: bool = False) -> tuple[list[Finding], str]:
    settings = load_config(config or repo / CONFIG)
    commit = edition.docs_main(repo)
    if open_prs:
        head = edition.text(repo, "rev-parse", "HEAD")
        if head != commit or edition.text(repo, "status", "--porcelain", "--untracked-files=no"):
            raise DiscoveryError(
                f"--open-prs runs on a clean checkout of {DOCS} main's newest commit, {commit}, but this "
                f"checkout is at {head}; nothing was prepared, so run discovery again"
            )
    path = add_worktree(repo, commit)
    tags = {}

    def sophon_tags() -> dict[str, Tag]:
        if "sophon" not in tags:
            tags["sophon"] = remote_tags(sophon, "refs/tags/v*", "refs/tags/release-qualified/v*")
        return tags["sophon"]

    try:
        main = read_main(path, commit)
        heads = remote_heads(repo)
        findings = []
        for product in ("enterprise", "oss"):
            options = settings["products"][product]
            if options["from"] is None:
                findings.append(Finding(product, "", "skipped",
                                        f"discovery is off: {CONFIG} names no release to start from"))
                continue
            if product == "enterprise":
                listed = list_enterprise(sophon_tags(), options["from"])
            else:
                listed = list_oss(remote_tags(lancedb, "refs/tags/v*"), options["from"], github)
            for finding in listed:
                if not finding.version:
                    continue
                classify(finding, main, github, repo, heads, options["skip"])
                if finding.state != "prepare":
                    continue
                try:
                    resolve(finding, main, lancedb, sophon, sophon_tags)
                    finding.preferred = preferred_note(main, finding)[1]
                except Waiting as exc:
                    finding.state, finding.reason = "waiting", str(exc)
                except (DiscoveryError, edition.EditionError) as exc:
                    finding.state, finding.reason = "blocked", str(exc)
            findings += listed
        if open_prs:
            for finding in findings:
                if finding.state in ("prepare", "resume", "open"):
                    try:
                        act(finding, main, repo, lancedb, sophon, github, settings["holds"])
                    except (DiscoveryError, edition.EditionError, compose.ComposeError) as exc:
                        finding.failed = True
                        finding.done = f"{finding.done}, then failed: {exc}" if finding.done else f"failed: {exc}"
    finally:
        remove_worktree(repo, path)
    return findings, commit


def report(findings: list[Finding], commit: str, open_prs: bool) -> str:
    lines = [f"# Release discovery{'' if open_prs else ' (dry run)'}", "", f"Read {DOCS} main at `{commit}`."
             + ("" if open_prs else " A dry run pushes, opens and starts nothing.")]
    for state, heading in STATES.items():
        group = [finding for finding in findings if finding.state == state]
        if not group:
            continue
        lines += ["", f"## {heading}", ""]
        for finding in group:
            lines.append(f"- **{finding.name}**: {finding.reason}.")
            if state == "prepare" and finding.record and not open_prs:
                lines += [f"  - {line}" for line in inputs(finding.record)]
                lines.append(f"  - Would prepare it on `{finding.branch}`, open its pull request and start its "
                             f"checks. The pull request {finding.preferred}.")
            if finding.pull and state in ("open", "declined", "blocked") and finding.pull.get("html_url"):
                lines.append(f"  - {finding.pull['html_url']}")
            lines += [f"  - {note}" for note in finding.notes]
            if finding.done:
                lines.append(f"  - {finding.done[0].upper()}{finding.done[1:]}.")
    return "\n".join(lines)


def failed(findings: list[Finding]) -> bool:
    return any(finding.state == "blocked" or finding.failed for finding in findings)


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--lancedb", type=Path, default=REPO_ROOT.parent / "lancedb",
                        help="a lancedb/lancedb repository to fetch release tags and commits into "
                             "(default: ../lancedb)")
    parser.add_argument("--sophon", type=Path, default=REPO_ROOT.parent / "sophon",
                        help="a lancedb/sophon repository to fetch release tags and commits into (default: ../sophon)")
    parser.add_argument("--config", type=Path,
                        help=f"read this file instead of {CONFIG}, to try a configuration in a dry run")
    parser.add_argument("--open-prs", action="store_true",
                        help="prepare editions and open their pull requests, as the Discover releases workflow does")
    parser.add_argument("--summary", type=Path, help="also append the report to this file")
    args = parser.parse_args(argv)
    if args.open_prs and args.config:
        parser.error(f"--open-prs reads {CONFIG}; --config is for trying a configuration in a dry run")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    try:
        findings, commit = discover(REPO_ROOT, args.lancedb.resolve(), args.sophon.resolve(),
                                    GitHub(writes=args.open_prs), args.config, args.open_prs)
    except (DiscoveryError, edition.EditionError) as exc:
        print(f"discover: {exc}", file=sys.stderr)
        return 1
    text = report(findings, commit, args.open_prs)
    print(text)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 1 if failed(findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
