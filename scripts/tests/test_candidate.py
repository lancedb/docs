"""Tests for recording a candidate and publishing it.

Every test publishes into a disposable bare repository standing in for
lancedb/docs, from a clone of it standing in for the workflow's checkout, and
runs `candidate.py` the way the workflows do. Nothing here reaches a real
remote.

Run with `make test-assemble`.
"""

import collections
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1]
CANDIDATE = SCRIPTS / "candidate.py"
WORKFLOWS = SCRIPTS.parent / ".github" / "workflows"
REPOSITORY = "lancedb/docs"
WORKFLOW = ".github/workflows/assemble.yml"
COMPOSE = ".github/workflows/compose.yml"
STAGING_URL = "https://example-staging.mintlify.site"
ANOTHER_PREVIEW = "https://example-some-other-preview.mintlify.site"
MINTLIFY = {"login": "mintlify[bot]", "type": "Bot"}
ACTIONS = {"login": "github-actions[bot]", "type": "Bot"}

SITE = {
    "docs.json": b'{"name": "LanceDB"}\n',
    "index.mdx": b"# Home\n",
    # Published byte for byte: no line-ending conversion ...
    "guide/page.mdx": b"line one\r\nline two\r\n",
    # ... and no ignore rules, even the site's own.
    ".gitignore": b"*.png\n",
    # Mintlify's own ignore file is part of the site: it keeps draft pages out.
    ".mintignore": b"drafts/page.mdx\n",
    "static/logo.png": bytes(range(256)),
    ".cursor/rules/.cursorrules": b"hidden, and published\n",
}


def versioned_site(release: str = "8.1.0") -> dict[str, bytes]:
    """A composed site, as compose.py writes one, with one Enterprise edition."""
    path = "enterprise/v" + release.replace(".", "-")
    group = {"group": "Start", "pages": [f"{path}/index"]}
    version = {"version": release, "default": True, "tag": "Latest",
               "tabs": [{"tab": "Docs", "groups": [group]}]}
    navigation = {"products": [{"product": "Enterprise", "versions": [version]}]}
    return {
        "docs.json": json.dumps({"name": "LanceDB", "navigation": navigation}).encode(),
        ".mintignore": f"/{path}/drafts/page.mdx\n".encode(),
        f"{path}/index.mdx": f"# Enterprise {release}\n".encode(),
        f"{path}/about-this-version.mdx": f"Release {release}.\n".encode(),
    }


def environment(**extra: str) -> dict[str, str]:
    """No GitHub Actions context and no user git configuration, unless given."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GITHUB_", "GIT_"))}
    env.update(
        GIT_AUTHOR_NAME="Test",
        GIT_AUTHOR_EMAIL="test@example.com",
        GIT_COMMITTER_NAME="Test",
        GIT_COMMITTER_EMAIL="test@example.com",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_TERMINAL_PROMPT="0",
    )
    env.update(extra)
    return env


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, env=environment(), check=True, capture_output=True, text=True
    ).stdout.strip()


def write_files(root: Path, files: dict[str, bytes]) -> None:
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def commit(repo: Path, files: dict[str, bytes], message: str) -> str:
    write_files(repo, files)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def refs(remote: Path) -> dict[str, str]:
    lines = git(remote, "for-each-ref", "--format=%(refname) %(objectname)").splitlines()
    return dict(line.split() for line in lines)


def changed(before: dict[str, str], after: dict[str, str]) -> set[str]:
    return {r for r in set(before) | set(after) if before.get(r) != after.get(r)}


def published(remote: Path, rev: str, prefix: str = "docs/") -> dict[str, bytes]:
    names = git(remote, "ls-tree", "-r", "--name-only", rev).splitlines()
    assert all(name.startswith(prefix) for name in names), names
    return {
        name[len(prefix) :]: subprocess.run(
            ["git", "cat-file", "blob", f"{rev}:{name}"],
            cwd=remote, env=environment(), check=True, capture_output=True,
        ).stdout
        for name in names
    }


def checksum(files: dict[str, bytes]) -> str:
    listing = "".join(
        f"{hashlib.sha256(files[p]).hexdigest()}  {p}\n" for p in sorted(files)
    )
    return hashlib.sha256(listing.encode()).hexdigest()


@pytest.fixture
def remote(tmp_path: Path) -> dict:
    """A bare lancedb/docs with `main`, a previously published `assembled` and
    `staging` and `deploy-freeze`, and a clone of it to publish from."""
    seed = tmp_path / "seed"
    seed.mkdir()
    git(seed, "init", "-q", "-b", "main")
    docs_commit = commit(seed, {"README.md": b"docs\n"}, "main")
    git(seed, "checkout", "-q", "--orphan", "assembled")
    git(seed, "rm", "-q", "-rf", ".")
    old = commit(seed, {"docs.json": b"{}\n", "index.mdx": b"# Old\n"}, "assemble 1234567")
    git(seed, "checkout", "-q", "--orphan", "staging")
    git(seed, "rm", "-q", "-rf", ".")
    staging = commit(seed, {"docs/docs.json": b"{}\n"}, "staging as it is")
    git(seed, "checkout", "-q", "-b", "deploy-freeze", "main")
    freeze = commit(seed, {"docs/docs.json": b"{}\n"}, "production as it is")
    bare = tmp_path / "docs.git"
    git(tmp_path, "clone", "-q", "--bare", str(seed), str(bare))
    workspace = tmp_path / "checkout"
    git(tmp_path, "clone", "-q", str(bare), str(workspace))
    return {
        "seed": seed,
        "bare": bare,
        "workspace": workspace,
        "docs_commit": docs_commit,
        "assembled": old,
        "staging": staging,
        "deploy_freeze": freeze,
    }


def advance_main(tmp_path: Path, remote: dict, name: str) -> str:
    """Merge another commit into main, as a pull request would."""
    clone = tmp_path / f"main-{name}"
    git(tmp_path, "clone", "-q", "-b", "main", str(remote["bare"]), str(clone))
    head = commit(clone, {f"{name}.md": b"merged\n"}, f"merge {name}")
    git(clone, "push", "-q", "origin", "main")
    return head


SOURCES = {"lancedb": "1" * 40, "enterprise": "2" * 40}


def sources_line(docs_commit: str, **override: str) -> str:
    return " ".join(f"{k}={v}" for k, v in {**SOURCES, "build": docs_commit, **override}.items())


class Candidate:
    """One recorded candidate: its artifact directory and its run's evidence."""

    def __init__(
        self, base: Path, docs_commit: str, files=None, run_id="1001", sources=None,
        site="assembled", **run
    ):
        self.dir = base / f"candidate-{run_id}"
        self.site = self.dir / "site"
        self.run_json = base / f"run-{run_id}.json"
        self.run_id = run_id
        self.docs_commit = docs_commit
        self.kind = site
        if site == "versioned":
            self.sources = sources or f"build={docs_commit}"
            self.refs = ()
            files = versioned_site() if files is None else files
        else:
            self.sources = sources or sources_line(docs_commit)
            self.refs = ("lancedb=main", "enterprise=main")
            files = SITE if files is None else files
        if isinstance(files, Path):
            shutil.copytree(files, self.site)
        else:
            write_files(self.site, files)
        self.files = {
            p.relative_to(self.site).as_posix(): p.read_bytes()
            for p in self.site.rglob("*") if p.is_file()
        }
        self.run = {
            "id": int(run_id),
            "path": COMPOSE if site == "versioned" else WORKFLOW,
            "event": "push",
            "head_branch": "main",
            "head_sha": docs_commit,
            "status": "completed",
            "conclusion": "success",
            "repository": {"full_name": REPOSITORY},
            **run,
        }
        self.run_json.write_text(json.dumps(self.run))
        self.manifest = self.record()
        self.checksum = self.manifest["site_sha256"]

    def record(self, refs=None, sources=None, context=True):
        env = environment()
        if context:
            env.update(
                GITHUB_REPOSITORY=REPOSITORY,
                GITHUB_RUN_ID=self.run_id,
                GITHUB_RUN_ATTEMPT="1",
                GITHUB_EVENT_NAME=self.run["event"],
                GITHUB_REF=f"refs/heads/{self.run['head_branch']}",
                GITHUB_SHA=self.docs_commit,
                GITHUB_WORKFLOW_REF=f"{REPOSITORY}/{self.run['path']}@refs/heads/main",
            )
        command = [sys.executable, str(CANDIDATE), "record", str(self.site), "--site", self.kind]
        command += ["--sources", sources or self.sources]
        for ref in self.refs if refs is None else refs:
            command += ["--ref", ref]
        command += ["--output", str(self.dir / "candidate.json")]
        subprocess.run(command, env=env, check=True, capture_output=True, text=True)
        self.manifest = json.loads((self.dir / "candidate.json").read_text())
        return self.manifest

    def edit_run(self, **fields) -> None:
        self.run.update(fields)
        self.run_json.write_text(json.dumps(self.run))

    def edit_manifest(self, **fields) -> None:
        self.manifest.update(fields)
        (self.dir / "candidate.json").write_text(json.dumps(self.manifest))


def publish(
    remote: dict,
    candidate: Candidate | None,
    *,
    target: str = "staging",
    run_id: str | None = None,
    site_sha256: str | None = None,
    content_dir: str = "/docs",
    artifact: Path | None | bool = True,
    automatic: bool = False,
    deployments: Path | None = None,
    staging_url: str = STAGING_URL,
    production_was: str | None = None,
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    """Run the publish command as the Publish workflow does. Production is
    requested as it is now, unless `production_was` says otherwise."""
    command = [sys.executable, str(CANDIDATE), "publish", "--target", target]
    if automatic:
        command += ["--automatic"]
    command += ["--run-id", candidate.run_id if run_id is None else run_id]
    if not automatic:
        command += ["--site-sha256", site_sha256 or candidate.checksum]
    if candidate is not None:
        command += ["--run", str(candidate.run_json)]
    command += ["--repository", REPOSITORY, "--content-dir", content_dir]
    if artifact is True:
        command += ["--candidate", str(candidate.dir)]
    elif artifact:
        command += ["--candidate", str(artifact)]
    if deployments is not None:
        command += ["--staging-deployments", str(deployments)]
    if target == "production":
        if production_was is None:
            production_was = git(remote["bare"], "for-each-ref", "--format=%(objectname)",
                                 "refs/heads/assembled")
        command += ["--staging-url", staging_url, "--production-was", production_was]
    return subprocess.run(
        command, cwd=remote["workspace"], env=env or environment(), capture_output=True, text=True
    )


def status(state="success", address=STAGING_URL, id=1, creator=MINTLIFY) -> dict:
    """A deployment status, as the API returns it."""
    entry = {"id": id, "state": state, "environment": "staging - docs", "creator": creator}
    if address is not None:
        entry["environment_url"] = address
    return entry


def deployment(commit_sha: str, address=STAGING_URL, *, id=1, creator=MINTLIFY,
               statuses: list | None = None) -> dict:
    """A deployment of a commit with its statuses, as the Publish workflow reads it.
    By default Mintlify's, live at the staging address."""
    return {
        "id": id, "sha": commit_sha, "ref": commit_sha, "environment": "staging - docs",
        "payload": {"branch": "staging"}, "creator": creator,
        "statuses": [status(address=address)] if statuses is None else statuses,
    }


def deployed(base: Path, commit_sha: str, name: str = "deployed", *entries: dict) -> Path:
    """The deployments of a commit: by default, Mintlify's one to staging."""
    path = base / f"deployments-{name}.json"
    path.write_text(json.dumps(list(entries) or [deployment(commit_sha)]))
    return path


def staged(remote: dict, candidate: Candidate) -> str:
    """Publish the candidate to staging by hand and return the commit."""
    result = publish(remote, candidate)
    assert result.returncode == 0, result.stderr
    return git(remote["bare"], "rev-parse", "staging")


def promote(remote: dict, candidate: Candidate, deployments: Path | None, **options):
    return publish(remote, candidate, target="production", deployments=deployments, **options)


# --------------------------------------------------------------------------- #
# recording
# --------------------------------------------------------------------------- #


def test_record_lists_every_file_and_names_the_candidate(tmp_path, remote):
    candidate = Candidate(tmp_path, remote["docs_commit"])
    manifest = candidate.manifest

    assert manifest["file_sha256"] == {
        rel: hashlib.sha256(data).hexdigest() for rel, data in sorted(SITE.items())
    }
    assert manifest["files"] == len(SITE)
    assert manifest["site_sha256"] == checksum(SITE)
    assert manifest["site"] == "assembled"
    assert manifest["sources"] == {**SOURCES, "build": remote["docs_commit"]}
    assert manifest["refs"] == {"lancedb": "main", "enterprise": "main"}
    assert manifest["run"]["run_id"] == "1001"
    assert manifest["run"]["sha"] == remote["docs_commit"]


def test_record_of_the_versioned_site_names_only_its_commit(tmp_path, remote):
    candidate = Candidate(tmp_path, remote["docs_commit"], site="versioned")

    assert candidate.manifest["site"] == "versioned"
    assert candidate.manifest["sources"] == {"build": remote["docs_commit"]}
    assert candidate.manifest["refs"] == {}
    assert candidate.checksum == checksum(versioned_site())
    with pytest.raises(subprocess.CalledProcessError) as failure:
        candidate.record(sources=sources_line(remote["docs_commit"]))
    assert "sources must name exactly build: missing [], unexpected ['enterprise', 'lancedb']" in (
        failure.value.stderr
    )


@pytest.mark.skipif(not shutil.which("sha256sum"), reason="needs GNU coreutils")
def test_checksum_is_recomputable_with_standard_tools(tmp_path, remote):
    candidate = Candidate(tmp_path, remote["docs_commit"])
    shell = (
        "find . -type f -printf '%P\\0' | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum"
    )
    result = subprocess.run(
        ["sh", "-c", shell], cwd=candidate.site, check=True, capture_output=True, text=True
    )
    assert result.stdout.split()[0] == candidate.checksum


@pytest.mark.parametrize(
    "build",
    ["a" * 40 + "+uncommitted", "unversioned", "abc1234"],
)
def test_record_refuses_a_source_that_is_not_a_clean_commit(tmp_path, remote, build):
    candidate = Candidate(tmp_path, remote["docs_commit"])
    with pytest.raises(subprocess.CalledProcessError) as failure:
        candidate.record(sources=sources_line(remote["docs_commit"], build=build))
    assert "is not a clean commit" in failure.value.stderr


def test_record_refuses_a_symlink(tmp_path, remote):
    candidate = Candidate(tmp_path, remote["docs_commit"])
    (candidate.site / "linked.mdx").symlink_to(candidate.site / "index.mdx")
    with pytest.raises(subprocess.CalledProcessError) as failure:
        candidate.record()
    assert "linked.mdx is a symlink" in failure.value.stderr


# --------------------------------------------------------------------------- #
# publishing to staging by hand
# --------------------------------------------------------------------------- #


def test_staging_publishes_exactly_the_candidate(tmp_path, remote):
    candidate = Candidate(tmp_path, remote["docs_commit"])
    before = refs(remote["bare"])

    result = publish(remote, candidate)

    assert result.returncode == 0, result.stderr
    after = refs(remote["bare"])
    assert changed(before, after) == {"refs/heads/staging"}, "only staging may change"
    head = after["refs/heads/staging"]
    assert git(remote["bare"], "rev-parse", f"{head}^") == remote["staging"]
    assert published(remote["bare"], head) == SITE
    message = git(remote["bare"], "log", "-1", "--format=%B", head)
    assert f"Candidate-Site-SHA256: {candidate.checksum}" in message
    assert "Candidate-Run: 1001" in message
    assert "Candidate-Site: assembled" in message
    assert f"Sources: {sources_line(remote['docs_commit'])}" in message
    assert "Refs: lancedb=main enterprise=main" in message
    assert "Published-From: the run's artifact" in message
    assert "Publication:" not in message
    assert json.loads(result.stdout)["commit"] == head


@pytest.mark.parametrize("content_dir", ["/", "."])
def test_root_content_directory(tmp_path, remote, content_dir):
    candidate = Candidate(tmp_path, remote["docs_commit"])

    result = publish(remote, candidate, content_dir=content_dir)

    assert result.returncode == 0, result.stderr
    assert published(remote["bare"], "staging", prefix="") == SITE


def test_staging_writes_only_the_staging_branch(tmp_path, remote):
    # Staging also takes a manual run on another branch that read pull
    # request heads: a hosted preview before anything merges.
    candidate = Candidate(
        tmp_path, remote["docs_commit"], event="workflow_dispatch", head_branch="jack/a3-move"
    )
    candidate.record(refs=("lancedb=refs/pull/4122/head", "enterprise=refs/pull/7681/head"))
    candidate.checksum = candidate.manifest["site_sha256"]
    before = refs(remote["bare"])

    result = publish(remote, candidate)

    assert result.returncode == 0, result.stderr
    after = refs(remote["bare"])
    assert changed(before, after) == {"refs/heads/staging"}
    assert published(remote["bare"], "staging") == SITE


def rewrite_record_for_a_changed_file(c):
    (c.site / "index.mdx").write_bytes(b"# Changed\n")
    hashes = dict(c.manifest["file_sha256"])
    hashes["index.mdx"] = hashlib.sha256(b"# Changed\n").hexdigest()
    listing = "".join(f"{hashes[p]}  {p}\n" for p in sorted(hashes))
    c.edit_manifest(file_sha256=hashes, site_sha256=hashlib.sha256(listing.encode()).hexdigest())


def edit_record_but_not_its_checksum(c):
    (c.site / "index.mdx").write_bytes(b"# Changed\n")
    hashes = dict(c.manifest["file_sha256"])
    hashes["index.mdx"] = hashlib.sha256(b"# Changed\n").hexdigest()
    c.edit_manifest(file_sha256=hashes)


REFUSALS = {
    "no run evidence": (lambda c: c.run_json.unlink(), {}, "no evidence of run 1001"),
    "run failed": (lambda c: c.edit_run(conclusion="failure"), {}, "has not succeeded"),
    "run cancelled": (lambda c: c.edit_run(conclusion="cancelled"), {}, "has not succeeded"),
    "run in progress": (
        lambda c: c.edit_run(status="in_progress", conclusion=None), {}, "has not succeeded"
    ),
    "another workflow's run": (
        lambda c: c.edit_run(path=".github/workflows/docs-ci.yml"),
        {},
        "ran .github/workflows/docs-ci.yml",
    ),
    "another repository's run": (
        lambda c: c.edit_run(repository={"full_name": "fork/docs"}), {}, "not a run of lancedb/docs"
    ),
    "evidence of another run": (lambda c: c.edit_run(id=999), {}, "describes run 999, not 1001"),
    "pull request run": (lambda c: c.edit_run(event="pull_request"), {}, "was a pull_request run"),
    "another checksum requested": (lambda c: None, {"site_sha256": "0" * 64}, "not " + "0" * 64),
    "file changed after recording": (
        lambda c: (c.site / "index.mdx").write_bytes(b"# Changed\n"), {}, "changed ['index.mdx']"
    ),
    "file added after recording": (
        lambda c: (c.site / "new.mdx").write_bytes(b"new\n"), {}, "extra ['new.mdx']"
    ),
    "file removed after recording": (
        lambda c: (c.site / "index.mdx").unlink(), {}, "missing ['index.mdx']"
    ),
    "record rewritten to match a changed file": (
        rewrite_record_for_a_changed_file, {}, "run 1001 recorded candidate"
    ),
    "record edited without its checksum": (
        edit_record_but_not_its_checksum, {}, "does not match its own file list"
    ),
    "artifact recorded by another run": (
        lambda c: c.edit_manifest(run={**c.manifest["run"], "run_id": "1002"}),
        {},
        "recorded by lancedb/docs run 1002",
    ),
    "artifact recorded outside a workflow run": (
        lambda c: c.record(context=False), {}, "recorded by None run None"
    ),
    "built from another docs commit": (
        lambda c: c.edit_run(head_sha="f" * 40), {}, "but run 1001 checked out " + "f" * 40
    ),
    "artifact without its record": (
        lambda c: (c.dir / "candidate.json").unlink(), {}, "holds no candidate.json"
    ),
    "no artifact and no earlier publication": (
        lambda c: shutil.rmtree(c.dir), {}, "has no candidate artifact"
    ),
    "content directory not set": (lambda c: None, {"content_dir": ""}, "no content directory"),
    "content directory escaping": (
        lambda c: None, {"content_dir": "../docs"}, "is not a plain relative path"
    ),
    "malformed run ID": (lambda c: None, {"run_id": "1001; true"}, "is not a number"),
    "malformed checksum": (lambda c: None, {"site_sha256": "ABC"}, "is not a lowercase SHA-256"),
    "run evidence that is not an object": (
        lambda c: c.run_json.write_text("[]"), {}, "the evidence of run 1001 is not a workflow run"
    ),
    "run evidence without a repository object": (
        lambda c: c.edit_run(repository="lancedb/docs"),
        {},
        "the evidence of run 1001 is not a workflow run",
    ),
    "a record that is not an object": (
        lambda c: (c.dir / "candidate.json").write_text("[]"), {}, "candidate.json is not a record"
    ),
    "a record without file hashes": (
        lambda c: c.edit_manifest(file_sha256=None), {}, "the record has no file hashes"
    ),
    "a record of the other site": (
        lambda c: c.edit_manifest(site="versioned"),
        {},
        "the candidate is a record of the versioned site, but run 1001 built the assembled site",
    ),
}


@pytest.mark.parametrize("case", REFUSALS)
def test_refused_evidence_publishes_nothing(tmp_path, remote, case):
    mutate, options, message = REFUSALS[case]
    candidate = Candidate(tmp_path, remote["docs_commit"])
    mutate(candidate)
    before = refs(remote["bare"])

    result = publish(remote, candidate, **options)

    assert result.returncode == 1, result.stdout
    assert message in result.stderr
    assert refs(remote["bare"]) == before


@pytest.mark.parametrize("site", ["assembled", "versioned"])
@pytest.mark.parametrize("target", ["staging", "production"])
def test_pull_request_candidates_are_refused_for_both_targets(tmp_path, remote, target, site):
    candidate = Candidate(tmp_path, remote["docs_commit"], site=site, event="pull_request")
    before = refs(remote["bare"])

    result = publish(remote, candidate, target=target)

    assert result.returncode == 1
    assert "was a pull_request run" in result.stderr
    assert refs(remote["bare"]) == before


# --------------------------------------------------------------------------- #
# provenance: exactly the site's roots, with each producer's ref
# --------------------------------------------------------------------------- #

PAIR = {"lancedb": "main", "enterprise": "main"}
# Sources and refs written into an otherwise valid record. `None` stands for
# the run's own docs commit.
PROVENANCE = {
    "build only": ({"build": None}, {}, "missing ['lancedb', 'enterprise'], unexpected []"),
    "Enterprise missing": (
        {"build": None, "lancedb": "1" * 40},
        {"lancedb": "main"},
        "missing ['enterprise'], unexpected []",
    ),
    "Enterprise source missing, both refs present": (
        {"build": None, "lancedb": "1" * 40}, PAIR, "sources must name exactly"
    ),
    "another producer instead of the pair": (
        {"build": None, "other": "3" * 40},
        {"other": "main"},
        "missing ['lancedb', 'enterprise'], unexpected ['other']",
    ),
    "another producer beside the pair": (
        {**SOURCES, "build": None, "other": "3" * 40}, PAIR, "missing [], unexpected ['other']"
    ),
    "Enterprise ref missing": (
        {**SOURCES, "build": None},
        {"lancedb": "main"},
        "refs must name exactly lancedb, enterprise: missing ['enterprise']",
    ),
    "a ref for the docs root": (
        {**SOURCES, "build": None}, {**PAIR, "build": "main"}, "missing [], unexpected ['build']"
    ),
    "a producer that is not a commit": (
        {**SOURCES, "build": None, "enterprise": "main"},
        PAIR,
        "source enterprise=main is not a clean commit",
    ),
    "a ref that is not a ref name": (
        {**SOURCES, "build": None}, {**PAIR, "enterprise": "main; true"}, "is not a plain ref name"
    ),
    "sources that are not a map": (["build"], PAIR, "are not a map of names to strings"),
    "versioned: a producer as well": (
        {**SOURCES, "build": None}, {}, "sources must name exactly build: missing [], unexpected"
    ),
    "versioned: a producer's ref": (
        {"build": None}, {"lancedb": "main"}, "refs must name exactly nothing"
    ),
}


@pytest.mark.parametrize("case", PROVENANCE)
def test_incomplete_provenance_publishes_nothing(tmp_path, remote, case):
    sources, recorded_refs, message = PROVENANCE[case]
    site = "versioned" if case.startswith("versioned") else "assembled"
    candidate = Candidate(tmp_path, remote["docs_commit"], site=site)
    if isinstance(sources, dict):
        sources = {name: commit or candidate.docs_commit for name, commit in sources.items()}
    candidate.edit_manifest(sources=sources, refs=recorded_refs)
    before = refs(remote["bare"])

    result = publish(remote, candidate)

    assert result.returncode == 1, result.stdout
    assert message in result.stderr
    assert refs(remote["bare"]) == before


@pytest.mark.parametrize(
    "sources, recorded_refs, message",
    [
        (lambda docs: f"lancedb={'1' * 40} build={docs}", ("lancedb=main", "enterprise=main"),
         "missing ['enterprise']"),
        (lambda docs: sources_line(docs, other="3" * 40), ("lancedb=main", "enterprise=main"),
         "unexpected ['other']"),
        (lambda docs: sources_line(docs), ("lancedb=main",), "missing ['enterprise']"),
        (lambda docs: sources_line(docs), ("lancedb=main", "enterprise=main", "other=main"),
         "unexpected ['other']"),
    ],
)
def test_record_requires_every_source_and_producer_ref(
    tmp_path, remote, sources, recorded_refs, message
):
    candidate = Candidate(tmp_path, remote["docs_commit"])
    with pytest.raises(subprocess.CalledProcessError) as failure:
        candidate.record(refs=recorded_refs, sources=sources(remote["docs_commit"]))
    assert message in failure.value.stderr


def forge(tmp_path: Path, remote: dict, files: dict[str, bytes], message: str,
          branch: str = "staging") -> str:
    """Push a commit with these files and this message onto `branch`."""
    forger = tmp_path / f"forger-{branch}"
    git(tmp_path, "clone", "-q", "-b", branch, str(remote["bare"]), str(forger))
    git(forger, "rm", "-q", "-rf", ".")
    write_files(forger, files)
    git(forger, "add", "-A", "-f")
    git(forger, "commit", "-q", "-m", message)
    git(forger, "push", "-q", "origin", branch)
    return git(forger, "rev-parse", "HEAD")


@pytest.mark.parametrize(
    "sources_trailer, refs_trailer, message",
    [
        ("{all}", "lancedb=main enterprise=main", None),
        ("build={docs}", "lancedb=main enterprise=main", "missing ['lancedb', 'enterprise']"),
        (
            f"lancedb={'1' * 40} build={{docs}}",
            "lancedb=main enterprise=main",
            "missing ['enterprise']",
        ),
        (
            "{all}",
            "lancedb=main",
            "refs must name exactly lancedb, enterprise: missing ['enterprise']",
        ),
        (
            f"build={{docs}} other={'3' * 40}",
            "lancedb=main enterprise=main",
            "unexpected ['other']",
        ),
    ],
    ids=[
        "complete", "build only", "Enterprise missing", "Enterprise ref missing", "another producer"
    ],
)
def test_republishing_from_history_requires_complete_provenance(
    tmp_path, remote, sources_trailer, refs_trailer, message
):
    candidate = Candidate(tmp_path, remote["docs_commit"], run_id="1004")
    # An earlier commit with exactly the candidate's files, run and checksum.
    trailers = sources_trailer.format(
        all=sources_line(remote["docs_commit"]), docs=remote["docs_commit"]
    )
    forge(
        tmp_path,
        remote,
        {f"docs/{path}": data for path, data in SITE.items()},
        f"Publish candidate\n\nCandidate-Site-SHA256: {candidate.checksum}\n"
        f"Candidate-Run: 1004\nSources: {trailers}\nRefs: {refs_trailer}\n",
    )
    shutil.rmtree(candidate.dir)
    before = refs(remote["bare"])

    result = publish(remote, candidate)

    if message is None:
        assert result.returncode == 0, result.stderr
        assert published(remote["bare"], "staging") == SITE
    else:
        assert result.returncode == 1, result.stdout
        assert message in result.stderr
        assert refs(remote["bare"]) == before


# --------------------------------------------------------------------------- #
# the run's workflow: the bare path, or qualified with the run's own ref
# --------------------------------------------------------------------------- #

WORKFLOW_PATHS = {
    "bare path": (WORKFLOW, {}, None),
    "with its branch, as GitHub documents": (WORKFLOW + "@main", {}, None),
    "with its full branch ref": (WORKFLOW + "@refs/heads/main", {}, None),
    "with its commit": (WORKFLOW + "@{docs}", {}, None),
    "another workflow with a ref": (
        ".github/workflows/docs-ci.yml@main", {}, "ran .github/workflows/docs-ci.yml@main, not"
    ),
    "another repository's copy": (
        "octo-org/docs/.github/workflows/assemble.yml@main",
        {},
        "ran octo-org/docs/.github/workflows/assemble.yml@main",
    ),
    "a lookalike file": (
        WORKFLOW + ".bak@main", {}, "ran .github/workflows/assemble.yml.bak@main, not"
    ),
    "a lookalike directory": (
        ".github/workflows/x/assemble.yml", {}, "ran .github/workflows/x/assemble.yml, not"
    ),
    "another branch's copy": (
        WORKFLOW + "@jack/a3-move", {}, "from jack/a3-move, not from its own branch main"
    ),
    "another commit's copy": (
        WORKFLOW + "@" + "f" * 40, {}, "from " + "f" * 40 + ", not from its own"
    ),
    "an empty ref": (WORKFLOW + "@", {}, "ran .github/workflows/assemble.yml@, not"),
    "a ref holding another @": (
        WORKFLOW + "@main@other", {}, "ran .github/workflows/assemble.yml@main@other"
    ),
    "no path": (None, {}, "ran None, not"),
    "a path that is not text": ([WORKFLOW], {}, "not .github/workflows/assemble.yml"),
    "its branch, but a failed run": (
        WORKFLOW + "@main", {"conclusion": "failure"}, "has not succeeded"
    ),
    "its branch, but a pull request": (
        WORKFLOW + "@main", {"event": "pull_request"}, "was a pull_request run"
    ),
    # The record says which site it is, and so does the workflow that ran.
    "the Compose workflow, with a record of the assembled site": (
        COMPOSE + "@main", {}, "a record of the assembled site, but run 1001 built the versioned"
    ),
}


@pytest.mark.parametrize("case", WORKFLOW_PATHS)
def test_workflow_identity(tmp_path, remote, case):
    path, fields, message = WORKFLOW_PATHS[case]
    candidate = Candidate(tmp_path, remote["docs_commit"])
    if isinstance(path, str):
        path = path.format(docs=remote["docs_commit"])
    candidate.edit_run(path=path, **fields)
    before = refs(remote["bare"])

    result = publish(remote, candidate)

    after = refs(remote["bare"])
    if message is None:
        assert result.returncode == 0, result.stderr
        assert changed(before, after) == {"refs/heads/staging"}
    else:
        assert result.returncode == 1, result.stdout
        assert message in result.stderr
        assert after == before


def test_staging_takes_a_manual_run_named_with_its_own_branch(tmp_path, remote):
    candidate = Candidate(
        tmp_path,
        remote["docs_commit"],
        event="workflow_dispatch",
        head_branch="jack/a3-move",
        path=WORKFLOW + "@jack/a3-move",
    )
    candidate.record(refs=("lancedb=refs/pull/4122/head", "enterprise=refs/pull/7681/head"))
    candidate.checksum = candidate.manifest["site_sha256"]

    result = publish(remote, candidate)

    assert result.returncode == 0, result.stderr
    assert published(remote["bare"], "staging") == SITE


# The independent review's probes (documentation/reviews/candidate/pass8/
# probe_publication.py), case for case: before this fix the three malformed
# records were published and the documented path was refused.
REVIEW_PROBES = {
    "control": (lambda c: None, 0),
    "no_producer_sources": (
        lambda c: c.edit_manifest(sources={"build": c.docs_commit}, refs={}), 1
    ),
    "enterprise_source_missing": (
        lambda c: c.edit_manifest(
            sources={"build": c.docs_commit, "lancedb": "1" * 40}, refs={"lancedb": "main"}
        ),
        1,
    ),
    "unexpected_producer_instead_of_expected_pair": (
        lambda c: c.edit_manifest(
            sources={"build": c.docs_commit, "other": "3" * 40}, refs={"other": "main"}
        ),
        1,
    ),
    "github_documented_path_with_ref": (lambda c: c.edit_run(path=WORKFLOW + "@main"), 0),
}


@pytest.mark.parametrize("case", REVIEW_PROBES)
def test_independent_review_probes(tmp_path, remote, case):
    mutate, expected = REVIEW_PROBES[case]
    candidate = Candidate(tmp_path, remote["docs_commit"])
    mutate(candidate)
    before = refs(remote["bare"])

    result = publish(remote, candidate)

    after = refs(remote["bare"])
    assert result.returncode == expected, result.stderr
    assert changed(before, after) == ({"refs/heads/staging"} if expected == 0 else set())


def assemble_sources(base: Path, index: bytes) -> dict:
    """LanceDB, Enterprise and docs checkouts with the shape assemble.yaml reads."""
    navigation = {
        "navigation": {
            "tabs": [
                {
                    "tab": "Docs",
                    "groups": [{"group": "All", "pages": ["index", "enterprise/security"]}],
                }
            ]
        }
    }
    repos = {}
    for name, files in {
        "lancedb": {
            "docs/web/docs.json": json.dumps(navigation).encode(),
            "docs/web/.mintignore": b"drafts/page.mdx\n",
            "docs/web/drafts/page.mdx": b"A draft page\n",
            "docs/web/index.mdx": index,
            "docs/web/enterprise/security.mdx": b"Open-source security\n",
        },
        "enterprise_repo": {"docs/web/enterprise/security.mdx": b"Enterprise security\n"},
        "docs": {"docs/geneva/udfs.mdx": b"Geneva UDFs\n"},
    }.items():
        repo = base / name
        repo.mkdir(parents=True)
        git(repo, "init", "-q", "-b", "main")
        commit(repo, files, f"{name} sources")
        repos[name] = repo
    return repos


def assemble(base: Path, repos: dict, output: Path) -> str:
    """Run the assembler and return its sources line."""
    config = base / "assemble.yaml"
    config.write_text(
        f"""output: {output}
roots:
  - name: lancedb
    path: {repos["lancedb"] / "docs/web"}
    role: reference
  - name: enterprise
    path: {repos["enterprise_repo"] / "docs/web"}
    role: overlay
    private: true
  - name: build
    path: {repos["docs"] / "docs"}
    role: reference
"""
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "assemble.py"), "--config", str(config)],
        env=environment(), check=True, capture_output=True, text=True,
    )
    return next(
        line.removeprefix("sources: ")
        for line in result.stdout.splitlines()
        if line.startswith("sources: ")
    )


def test_publication_publishes_the_checked_output_not_newer_sources(tmp_path, remote):
    base = tmp_path.resolve()
    repos = assemble_sources(base / "sources", b"# Release one\n")
    sources = assemble(base, repos, base / "built")
    docs_commit = sources.split("build=")[1]
    candidate = Candidate(base, docs_commit, files=base / "built", run_id="2001", sources=sources)
    checked = candidate.files
    # A newer producer commit after the run, and what assembling it would give.
    commit(repos["lancedb"], {"docs/web/index.mdx": b"# Release two\n"}, "newer")
    newer = assemble(base, repos, base / "newer")
    assert newer != sources
    newer_files = {
        p.relative_to(base / "newer").as_posix(): p.read_bytes()
        for p in (base / "newer").rglob("*")
        if p.is_file()
    }
    assert checksum(newer_files) != candidate.checksum
    # Publication cannot read the sources at all.
    shutil.rmtree(base / "sources")

    result = publish(remote, candidate)

    assert result.returncode == 0, result.stderr
    assert published(remote["bare"], "staging") == checked
    assert checked["index.mdx"] == b"# Release one\n"
    assert checked["enterprise/security.mdx"] == b"Enterprise security\n"
    # The draft page stays in the files, and so does the ignore file that hides it.
    assert checked[".mintignore"] == b"drafts/page.mdx\n"
    assert checked["drafts/page.mdx"] == b"A draft page\n"

    # Nor can the run's ID carry the newer build's checksum.
    before = refs(remote["bare"])
    refused = publish(remote, candidate, site_sha256=checksum(newer_files))
    assert refused.returncode == 1
    assert "run 2001 recorded candidate" in refused.stderr
    assert refs(remote["bare"]) == before


def test_staging_republishes_an_earlier_candidate_after_its_artifact_expired(tmp_path, remote):
    first = Candidate(tmp_path, remote["docs_commit"], run_id="1001")
    second = Candidate(
        tmp_path, remote["docs_commit"], files={**SITE, "index.mdx": b"# Broken\n"}, run_id="1002"
    )
    assert publish(remote, first).returncode == 0
    first_commit = git(remote["bare"], "rev-parse", "staging")
    assert publish(remote, second).returncode == 0
    shutil.rmtree(first.dir)

    result = publish(remote, first)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["from"] == f"earlier publication {first_commit}"
    head = git(remote["bare"], "rev-parse", "staging")
    assert published(remote["bare"], head) == SITE
    assert git(remote["bare"], "rev-parse", f"{head}^{{tree}}") == git(
        remote["bare"], "rev-parse", f"{first_commit}^{{tree}}"
    )
    history = git(remote["bare"], "log", "--format=%s", "staging").splitlines()
    assert history[:3] == [
        f"Publish candidate {first.checksum[:12]} to staging",
        f"Publish candidate {second.checksum[:12]} to staging",
        f"Publish candidate {first.checksum[:12]} to staging",
    ]


def test_an_earlier_publication_must_still_match_its_checksum(tmp_path, remote):
    candidate = Candidate(tmp_path, remote["docs_commit"], run_id="1003")
    # Someone pushes a commit claiming to publish this candidate, with other files.
    forge(
        tmp_path,
        remote,
        {"docs/index.mdx": b"# Not the candidate\n"},
        f"Publish candidate\n\nCandidate-Site-SHA256: {candidate.checksum}\n"
        f"Candidate-Run: 1003\nSources: {sources_line(remote['docs_commit'])}\n"
        "Refs: lancedb=main enterprise=main\n",
    )
    shutil.rmtree(candidate.dir)
    before = refs(remote["bare"])

    result = publish(remote, candidate)

    assert result.returncode == 1
    assert "is not candidate" in result.stderr
    assert refs(remote["bare"]) == before


@pytest.mark.parametrize("target", ["staging", "production"])
def test_a_branch_moved_by_someone_else_is_never_overwritten(tmp_path, remote, target):
    candidate = Candidate(tmp_path, remote["docs_commit"], site="versioned")
    deployments = None
    if target == "production":
        deployments = deployed(tmp_path, staged(remote, candidate))
    branch = {"staging": "staging", "production": "assembled"}[target]
    # Another writer moves the branch just before this publication pushes.
    mover = tmp_path / "mover"
    git(tmp_path, "clone", "-q", "-b", branch, str(remote["bare"]), str(mover))
    moved = commit(mover, {"index.mdx": b"# Someone else\n"}, "someone else")
    shim = tmp_path / "bin"
    shim.mkdir()
    real_git = shutil.which("git")
    (shim / "git").write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = push ]; then "{real_git}" -C "{mover}" push -q origin {branch}; fi\n'
        f'exec "{real_git}" "$@"\n'
    )
    (shim / "git").chmod(0o755)

    result = publish(
        remote, candidate, target=target, deployments=deployments,
        env=environment(PATH=f"{shim}:{os.environ['PATH']}"),
    )

    assert result.returncode == 1
    assert "git push failed" in result.stderr
    assert git(remote["bare"], "rev-parse", branch) == moved


# --------------------------------------------------------------------------- #
# the versioned site: by hand to staging, then promoted from staging
# --------------------------------------------------------------------------- #


def test_the_versioned_site_is_published_only_from_main(tmp_path, remote):
    candidate = Candidate(
        tmp_path, remote["docs_commit"], site="versioned", event="workflow_dispatch",
        head_branch="contributor/new-edition", path=COMPOSE + "@contributor/new-edition",
    )
    before = refs(remote["bare"])

    result = publish(remote, candidate)

    assert result.returncode == 1
    assert (
        "the versioned site is published only from main; run 1001 built contributor/new-edition"
        in result.stderr
    )
    assert refs(remote["bare"]) == before


def tagged_main(tmp_path: Path, remote: dict) -> str:
    """A commit that never merged, pushed as a tag named main. GitHub reports a
    run of it with the head_branch main."""
    clone = tmp_path / "tagger"
    git(tmp_path, "clone", "-q", "-b", "main", str(remote["bare"]), str(clone))
    head = commit(clone, {"unreviewed.md": b"never merged\n"}, "not on main")
    git(clone, "tag", "main", head)
    git(clone, "push", "-q", "origin", "refs/tags/main")
    return head


def test_a_run_of_a_tag_named_main_is_never_published(tmp_path, remote):
    candidate = Candidate(tmp_path, tagged_main(tmp_path, remote), site="versioned",
                          event="workflow_dispatch")
    before = refs(remote["bare"])

    result = publish(remote, candidate)

    assert result.returncode == 1
    assert f"run 1001 built {candidate.docs_commit}, which main does not contain" in result.stderr
    assert refs(remote["bare"]) == before


def test_production_refuses_a_tag_named_main_even_when_staging_shows_it(tmp_path, remote):
    candidate = Candidate(tmp_path, tagged_main(tmp_path, remote), site="versioned",
                          event="workflow_dispatch")
    # Pushed to staging outside Publish, with a publication's trailers.
    staging = forge(
        tmp_path, remote, {f"docs/{path}": data for path, data in candidate.files.items()},
        f"Publish candidate\n\nCandidate-Site-SHA256: {candidate.checksum}\nCandidate-Run: 1001\n"
        f"Candidate-Site: versioned\nSources: build={candidate.docs_commit}\n",
    )
    before = refs(remote["bare"])

    result = promote(remote, candidate, deployed(tmp_path, staging))

    assert result.returncode == 1
    assert "which main does not contain" in result.stderr
    assert refs(remote["bare"]) == before


@pytest.mark.parametrize("checkout", ["full", "shallow"])
def test_an_earlier_commit_of_main_is_published_and_promoted(tmp_path, remote, checkout):
    candidate = Candidate(tmp_path, remote["docs_commit"], site="versioned", event="workflow_dispatch")
    advance_main(tmp_path, remote, "later")
    if checkout == "shallow":
        shutil.rmtree(remote["workspace"])
        git(tmp_path, "clone", "-q", "--depth", "1", "-b", "main", f"file://{remote['bare']}",
            str(remote["workspace"]))
        assert git(remote["workspace"], "rev-parse", "--is-shallow-repository") == "true"

    result = promote(remote, candidate, deployed(tmp_path, staged(remote, candidate)))

    assert result.returncode == 0, result.stderr
    assert published(remote["bare"], "assembled") == candidate.files


def test_a_versioned_candidate_lists_its_products(tmp_path, remote):
    files = versioned_site()
    files["docs.json"] = b'{"name": "LanceDB", "navigation": {"tabs": []}}'
    candidate = Candidate(tmp_path, remote["docs_commit"], files=files, site="versioned")
    before = refs(remote["bare"])

    result = publish(remote, candidate)

    assert result.returncode == 1
    assert "docs.json lists no products and versions" in result.stderr
    assert refs(remote["bare"]) == before


def test_production_promotes_exactly_what_staging_shows(tmp_path, remote):
    candidate = Candidate(tmp_path, remote["docs_commit"], site="versioned")
    staging = staged(remote, candidate)
    before = refs(remote["bare"])

    result = promote(remote, candidate, deployed(tmp_path, staging))

    assert result.returncode == 0, result.stderr
    after = refs(remote["bare"])
    assert changed(before, after) == {"refs/heads/assembled"}, "only assembled may change"
    head = after["refs/heads/assembled"]
    assert git(remote["bare"], "rev-parse", f"{head}^") == remote["assembled"]
    assert published(remote["bare"], head) == candidate.files
    assert git(remote["bare"], "rev-parse", f"{head}^{{tree}}") == git(
        remote["bare"], "rev-parse", f"{staging}^{{tree}}"
    )
    message = git(remote["bare"], "log", "-1", "--format=%B", head)
    assert f"Candidate-Site-SHA256: {candidate.checksum}" in message
    assert "Candidate-Site: versioned" in message
    assert f"Sources: build={remote['docs_commit']}" in message
    assert "Refs:" not in message
    assert f"Published-From: staging {staging}" in message


def stage_newer(tmp_path, remote, candidate):
    newer = Candidate(
        tmp_path, advance_main(tmp_path, remote, "newer"), files=versioned_site("8.2.0"),
        run_id="1002", site="versioned",
    )
    staged(remote, newer)
    return newer


PROMOTION_REFUSALS = {
    "never on staging": (
        lambda tmp_path, remote, c: None, "staging shows no candidate, not candidate"
    ),
    "a newer build reached staging after review": (stage_newer, "staging shows candidate"),
    "artifact expired before promotion": (
        lambda tmp_path, remote, c: shutil.rmtree(c.dir),
        "has no candidate artifact, so nothing shows it built what staging shows",
    ),
    "artifact altered since staging": (
        lambda tmp_path, remote, c: next(c.site.rglob("index.mdx")).write_bytes(b"# Changed\n"),
        "the artifact's files differ from its record",
    ),
}


@pytest.mark.parametrize("case", PROMOTION_REFUSALS)
def test_production_refuses_what_staging_does_not_show(tmp_path, remote, case):
    after_staging, message = PROMOTION_REFUSALS[case]
    candidate = Candidate(tmp_path, remote["docs_commit"], site="versioned")
    if case != "never on staging":
        staged(remote, candidate)
    after_staging(tmp_path, remote, candidate)
    staging = git(remote["bare"], "rev-parse", "staging")
    before = refs(remote["bare"])

    result = promote(remote, candidate, deployed(tmp_path, staging))

    assert result.returncode == 1, result.stdout
    assert message in result.stderr
    assert refs(remote["bare"]) == before


NOT_LIVE = f"not live at {STAGING_URL}"
# What the Publish workflow read about the staging commit `s`, the staging
# address it was given, and what production then says, or None to publish.
DEPLOYMENT_EVIDENCE = {
    "Mintlify put it live at the staging address": (lambda s: [deployment(s)], {}, None),
    "Mintlify's success after its own earlier status": (
        lambda s: [deployment(s, statuses=[status("in_progress", None), status(id=2)])], {}, None
    ),
    "another account's newer deployment beside Mintlify's": (
        lambda s: [deployment(s), deployment(s, ANOTHER_PREVIEW, id=2, creator=ACTIONS)], {}, None
    ),
    "Mintlify put it live at another preview": (
        lambda s: [deployment(s, ANOTHER_PREVIEW)],
        {},
        f"is success at {ANOTHER_PREVIEW}, " + NOT_LIVE,
    ),
    "Mintlify's newest deployment of it went to another preview": (
        lambda s: [deployment(s), deployment(s, ANOTHER_PREVIEW, id=2)], {}, NOT_LIVE
    ),
    "a lookalike address": (
        lambda s: [deployment(s, STAGING_URL + ".example.com")], {}, NOT_LIVE
    ),
    "a status that names no address": (
        lambda s: [deployment(s, None)], {}, "is success at no address, " + NOT_LIVE
    ),
    "Mintlify still deploying it again": (
        lambda s: [deployment(s), deployment(s, id=2, statuses=[])],
        {},
        "is unfinished at no address",
    ),
    "Mintlify's deployment failed": (
        lambda s: [deployment(s, statuses=[status("failure", None)])],
        {},
        "is failure at no address",
    ),
    "a newer status at the staging address that is not a success": (
        lambda s: [deployment(s, statuses=[status(), status("failure", id=2)])],
        {},
        f"is failure at {STAGING_URL}, " + NOT_LIVE,
    ),
    "a success added by another account": (
        lambda s: [
            deployment(s, statuses=[status("failure", None), status(id=2, creator=ACTIONS)])
        ],
        {},
        "is failure at no address",
    ),
    "a deployment recorded by another account": (
        lambda s: [deployment(s, creator=ACTIONS)], {}, "Mintlify has not reported deploying"
    ),
    "a deployment of another commit": (
        lambda s: [deployment("e" * 40)], {}, "Mintlify has not reported deploying"
    ),
    "no deployments read": (lambda s: None, {}, "Mintlify has not reported deploying"),
    "no staging address set": (
        lambda s: [deployment(s)], {"staging_url": ""}, "no staging address"
    ),
    "a staging address with a trailing slash": (
        lambda s: [deployment(s, STAGING_URL + "/")],
        {"staging_url": STAGING_URL + "/"},
        "no staging address",
    ),
}


@pytest.mark.parametrize("case", DEPLOYMENT_EVIDENCE)
def test_production_requires_mintlify_to_report_staging_live_at_its_address(tmp_path, remote, case):
    evidence, options, message = DEPLOYMENT_EVIDENCE[case]
    candidate = Candidate(tmp_path, remote["docs_commit"], site="versioned")
    staging = staged(remote, candidate)
    entries = evidence(staging)
    deployments = None if entries is None else deployed(tmp_path, staging, "case", *entries)
    before = refs(remote["bare"])

    result = promote(remote, candidate, deployments, **options)

    if message is None:
        assert result.returncode == 0, result.stderr
        assert published(remote["bare"], "assembled") == candidate.files
    else:
        assert result.returncode == 1, result.stdout
        assert message in result.stderr
        assert refs(remote["bare"]) == before


def test_production_refuses_staging_files_that_are_not_the_candidate(tmp_path, remote):
    # Anyone who can push to staging can copy a candidate's trailers.
    candidate = Candidate(tmp_path, remote["docs_commit"], site="versioned")
    message = git(remote["bare"], "log", "-1", "--format=%B", staged(remote, candidate))
    forged = forge(tmp_path, remote, {"docs/docs.json": b"{}\n"}, message)
    before = refs(remote["bare"])

    result = promote(remote, candidate, deployed(tmp_path, forged))

    assert result.returncode == 1
    assert f"the tree from staging {forged} is not candidate" in result.stderr
    assert refs(remote["bare"]) == before


def test_production_takes_only_the_versioned_site(tmp_path, remote):
    candidate = Candidate(tmp_path, remote["docs_commit"])
    staging = staged(remote, candidate)
    before = refs(remote["bare"])

    result = promote(remote, candidate, deployed(tmp_path, staging))

    assert result.returncode == 1
    assert "production takes only the versioned site; run 1001 built the assembled site" in (
        result.stderr
    )
    assert refs(remote["bare"]) == before


def test_rollback_republishes_an_earlier_production_candidate_without_staging(tmp_path, remote):
    first = Candidate(tmp_path, remote["docs_commit"], site="versioned", run_id="1001")
    assert promote(remote, first, deployed(tmp_path, staged(remote, first), "1")).returncode == 0
    first_commit = git(remote["bare"], "rev-parse", "assembled")
    second = Candidate(
        tmp_path, advance_main(tmp_path, remote, "second"), files=versioned_site("8.2.0"),
        run_id="1002", site="versioned",
    )
    assert promote(remote, second, deployed(tmp_path, staged(remote, second), "2")).returncode == 0
    # Both artifacts have expired, and staging still shows the second candidate.
    shutil.rmtree(first.dir)
    shutil.rmtree(second.dir)

    result = promote(remote, first, None)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["from"] == f"earlier publication {first_commit}"
    head = git(remote["bare"], "rev-parse", "assembled")
    assert published(remote["bare"], head) == first.files
    assert git(remote["bare"], "log", "--format=%s", "assembled").splitlines()[:3] == [
        f"Publish candidate {first.checksum[:12]} to production",
        f"Publish candidate {second.checksum[:12]} to production",
        f"Publish candidate {first.checksum[:12]} to production",
    ]


@pytest.mark.parametrize("path", ["promoting from staging", "rolling back"])
def test_an_approval_given_after_production_changed_publishes_nothing(tmp_path, remote, path):
    first = Candidate(tmp_path, remote["docs_commit"], site="versioned", run_id="1001")
    second = stage_newer(tmp_path, remote, first)
    staging = git(remote["bare"], "rev-parse", "staging")
    if path == "rolling back":
        # Production published the second, then the first; now the second again.
        assert promote(remote, second, deployed(tmp_path, staging, "2")).returncode == 0
        first_staged = staged(remote, first)
        assert promote(remote, first, deployed(tmp_path, first_staged, "1")).returncode == 0
        candidate, deployments = second, None
    else:
        candidate, deployments = second, deployed(tmp_path, staging)
    # Requested while production showed this; approved after it changed.
    requested = git(remote["bare"], "rev-parse", "assembled")
    forge(tmp_path, remote, {"docs/index.mdx": b"# Promoted meanwhile\n"}, "another publication",
          branch="assembled")
    before = refs(remote["bare"])

    result = promote(remote, candidate, deployments, production_was=requested)

    assert result.returncode == 1
    assert f"production has changed since this run was requested: it was {requested}" in (
        result.stderr
    )
    assert refs(remote["bare"]) == before


def publication_group(job: str) -> str:
    return workflow("publish.yml")["jobs"][job]["concurrency"]["group"]


def test_staging_waits_while_production_checks_it_and_pushes(tmp_path, remote):
    # Two valid candidates: the first is promoted while the second is published
    # to staging. GitHub runs one job per concurrency group at a time; a lock
    # per group name, as publish.yml names them, plays that here.
    first = Candidate(tmp_path, remote["docs_commit"], site="versioned", run_id="1001")
    first_staged = staged(remote, first)
    deployments = deployed(tmp_path, first_staged)
    newer = Candidate(
        tmp_path, advance_main(tmp_path, remote, "newer"), files=versioned_site("8.2.0"),
        run_id="1002", site="versioned",
    )
    groups = collections.defaultdict(threading.Lock)
    # Production stops just before its push until it is told to go on.
    signals = tmp_path / "signals"
    signals.mkdir()
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "git").write_text(
        "#!/bin/sh\n"
        'if [ "$1" = push ]; then\n'
        f'  : > "{signals}/pushing"; i=0\n'
        f'  while [ ! -e "{signals}/go" ] && [ $i -lt 1200 ]; do sleep 0.05; i=$((i+1)); done\n'
        "fi\n"
        f'exec "{shutil.which("git")}" "$@"\n'
    )
    (shim / "git").chmod(0o755)
    results, started = {}, threading.Event()

    def production():
        with groups[publication_group("production")]:
            results["production"] = promote(
                remote, first, deployments, env=environment(PATH=f"{shim}:{os.environ['PATH']}")
            )

    def staging():
        with groups[publication_group("staging")]:
            started.set()
            results["staging"] = publish(remote, newer)

    promoting = threading.Thread(target=production)
    promoting.start()
    while not (signals / "pushing").exists() and promoting.is_alive():
        time.sleep(0.05)
    publishing = threading.Thread(target=staging)
    publishing.start()
    # In a group of its own, the staging publication would finish now.
    if started.wait(timeout=2):
        publishing.join()
    staging_at_push = git(remote["bare"], "rev-parse", "staging")
    (signals / "go").touch()
    promoting.join()
    publishing.join()

    assert results["production"].returncode == 0, results["production"].stderr
    assert staging_at_push == first_staged, "staging changed while production checked it"
    assert published(remote["bare"], "assembled") == first.files
    assert results["staging"].returncode == 0, results["staging"].stderr
    assert published(remote["bare"], "staging") == newer.files


def test_production_must_say_what_it_was_when_requested(tmp_path, remote):
    candidate = Candidate(tmp_path, remote["docs_commit"], site="versioned")
    deployments = deployed(tmp_path, staged(remote, candidate))
    command = [
        sys.executable, str(CANDIDATE), "publish", "--target", "production",
        "--run-id", candidate.run_id, "--site-sha256", candidate.checksum,
        "--run", str(candidate.run_json), "--repository", REPOSITORY, "--content-dir", "/docs",
        "--candidate", str(candidate.dir), "--staging-deployments", str(deployments),
        "--staging-url", STAGING_URL,
    ]
    before = refs(remote["bare"])

    result = subprocess.run(
        command, cwd=remote["workspace"], env=environment(), capture_output=True, text=True
    )

    assert result.returncode == 1
    assert "production needs --production-was" in result.stderr
    assert refs(remote["bare"]) == before


def test_a_rollback_needs_an_earlier_production_publication(tmp_path, remote):
    # Staged once and never approved, then replaced on staging.
    candidate = Candidate(tmp_path, remote["docs_commit"], site="versioned", run_id="1001")
    staged(remote, candidate)
    stage_newer(tmp_path, remote, candidate)
    shutil.rmtree(candidate.dir)
    before = refs(remote["bare"])

    result = promote(remote, candidate, None)

    assert result.returncode == 1
    assert "and production has not published that before" in result.stderr
    assert refs(remote["bare"]) == before


# --------------------------------------------------------------------------- #
# staging following main by itself
# --------------------------------------------------------------------------- #


def follow(remote, candidate, **options):
    return publish(remote, candidate, automatic=True, **options)


@pytest.mark.parametrize("shown", ["no publication", "the assembled site"])
def test_staging_follows_main_only_once_it_shows_the_versioned_site(tmp_path, remote, shown):
    if shown == "the assembled site":
        staged(remote, Candidate(tmp_path, remote["docs_commit"], run_id="1000"))
    candidate = Candidate(tmp_path, remote["docs_commit"], site="versioned")
    before = refs(remote["bare"])

    # The first versioned candidate is published to staging by hand.
    result = follow(remote, candidate)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["published"] is False
    assert "published there by hand" in json.loads(result.stdout)["reason"]
    assert refs(remote["bare"]) == before


def test_staging_follows_main_to_its_newest_versioned_candidate(tmp_path, remote):
    first = Candidate(tmp_path, remote["docs_commit"], site="versioned", run_id="1001")
    staged(remote, first)
    second = Candidate(
        tmp_path, advance_main(tmp_path, remote, "second"), files=versioned_site("8.2.0"),
        run_id="1002", site="versioned",
    )
    before = refs(remote["bare"])

    result = follow(remote, second)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["from"] == "the run's artifact"
    assert changed(before, refs(remote["bare"])) == {"refs/heads/staging"}
    assert published(remote["bare"], "staging") == second.files
    message = git(remote["bare"], "log", "-1", "--format=%B", "staging")
    assert f"Candidate-Site-SHA256: {second.checksum}" in message


def test_staging_is_left_alone_when_main_leaves_the_site_as_it_was(tmp_path, remote):
    first = Candidate(tmp_path, remote["docs_commit"], site="versioned", run_id="1001")
    staged(remote, first)
    # A newer commit of main, such as a README change, composes the same site.
    same = Candidate(
        tmp_path, advance_main(tmp_path, remote, "readme"), run_id="1002", site="versioned"
    )
    assert same.checksum == first.checksum
    before = refs(remote["bare"])

    result = follow(remote, same)

    assert result.returncode == 0, result.stderr
    reason = json.loads(result.stdout)["reason"]
    assert reason == f"staging already shows candidate {first.checksum}, from run 1001"
    assert refs(remote["bare"]) == before


def test_staging_is_left_alone_only_beside_a_verified_candidate(tmp_path, remote):
    candidate = Candidate(tmp_path, remote["docs_commit"], site="versioned")
    staged(remote, candidate)
    next(candidate.site.rglob("index.mdx")).write_bytes(b"# Changed\n")
    before = refs(remote["bare"])

    result = follow(remote, candidate)

    assert result.returncode == 1, result.stdout
    assert "the artifact's files differ from its record" in result.stderr
    assert refs(remote["bare"]) == before


@pytest.mark.parametrize("main", ["moved on", "where it was", "where it was, artifact expired"])
def test_staging_files_that_are_not_the_candidate_they_claim_are_replaced(tmp_path, remote, main):
    first = Candidate(tmp_path, remote["docs_commit"], site="versioned", run_id="1001")
    message = git(remote["bare"], "log", "-1", "--format=%B", staged(remote, first))
    # Other files under the publication's own message, its sources included.
    forged = forge(tmp_path, remote, {"docs/docs.json": b"{}\n"}, message)
    newest = first
    if main == "moved on":
        newest = Candidate(
            tmp_path, advance_main(tmp_path, remote, "readme"), run_id="1002", site="versioned"
        )
    if main.endswith("artifact expired"):
        shutil.rmtree(newest.dir)
    before = refs(remote["bare"])

    result = follow(remote, newest)

    if main.endswith("artifact expired"):
        assert result.returncode == 1, result.stdout
        assert "of main's newest commit, has no candidate artifact" in result.stderr
        assert refs(remote["bare"]) == before
    else:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["published"] is True
        assert git(remote["bare"], "rev-parse", "staging^") == forged
        assert published(remote["bare"], "staging") == newest.files


FOLLOW_SKIPS = {
    # An older run, finishing or re-run after main moved on, publishes nothing.
    "main has moved on": (
        lambda tmp_path, remote, c: advance_main(tmp_path, remote, "later"), {},
        "but main is now",
    ),
    # Its files, checked against the run's record, not its message.
    "staging already shows main's newest candidate": (
        lambda tmp_path, remote, c: None, {}, "staging already shows candidate ",
    ),
    "no successful run of main's newest commit yet": (
        lambda tmp_path, remote, c: None, {"run_id": ""}, "has no successful Compose run yet",
    ),
}


@pytest.mark.parametrize("case", FOLLOW_SKIPS)
def test_staging_following_main_never_goes_backwards(tmp_path, remote, case):
    prepare, options, reason = FOLLOW_SKIPS[case]
    candidate = Candidate(tmp_path, remote["docs_commit"], site="versioned")
    staged(remote, candidate)
    prepare(tmp_path, remote, candidate)
    before = refs(remote["bare"])

    result = follow(remote, candidate, **options)

    assert result.returncode == 0, result.stderr
    assert reason in json.loads(result.stdout)["reason"]
    assert refs(remote["bare"]) == before


FOLLOW_REFUSALS = {
    "main's newest run kept no artifact": (
        lambda c: shutil.rmtree(c.dir), "of main's newest commit, has no candidate artifact"
    ),
    "a run of the assembled site": (
        lambda c: c.edit_run(path=WORKFLOW), "built the assembled site, not the versioned site"
    ),
    "a failed run": (lambda c: c.edit_run(conclusion="failure"), "has not succeeded"),
    "an altered artifact": (
        lambda c: next(c.site.rglob("index.mdx")).write_bytes(b"# Changed\n"),
        "the artifact's files differ from its record",
    ),
}


@pytest.mark.parametrize("case", FOLLOW_REFUSALS)
def test_staging_following_main_refuses_what_it_cannot_verify(tmp_path, remote, case):
    mutate, message = FOLLOW_REFUSALS[case]
    first = Candidate(tmp_path, remote["docs_commit"], site="versioned", run_id="1001")
    staged(remote, first)
    second = Candidate(
        tmp_path, advance_main(tmp_path, remote, "second"), files=versioned_site("8.2.0"),
        run_id="1002", site="versioned",
    )
    mutate(second)
    before = refs(remote["bare"])

    result = follow(remote, second)

    assert result.returncode == 1, result.stdout
    assert message in result.stderr
    assert refs(remote["bare"]) == before


def test_only_staging_is_published_automatically(tmp_path, remote):
    candidate = Candidate(tmp_path, remote["docs_commit"], site="versioned")
    before = refs(remote["bare"])

    result = follow(remote, candidate, target="production")

    assert result.returncode == 1
    assert "only staging is published automatically" in result.stderr
    assert refs(remote["bare"]) == before


# --------------------------------------------------------------------------- #
# the workflows: what pull request code can reach
# --------------------------------------------------------------------------- #


def workflow(name: str) -> dict:
    loaded = yaml.safe_load((WORKFLOWS / name).read_text())
    # YAML 1.1 reads the `on:` key as true.
    loaded["on"] = loaded.pop(True)
    return loaded


def test_compose_runs_pull_request_code_without_write_access_or_secrets():
    compose = workflow("compose.yml")

    assert "pull_request" in compose["on"]
    assert compose["permissions"] == {"contents": "read"}
    assert all("permissions" not in job for job in compose["jobs"].values())
    assert "secrets." not in yaml.safe_dump(compose)


def test_publish_runs_only_this_repositorys_main_code_with_its_credentials():
    publish_yml = workflow("publish.yml")
    text = (WORKFLOWS / "publish.yml").read_text()
    jobs = publish_yml["jobs"]

    # Never a pull request's copy of the workflow.
    assert set(publish_yml["on"]) == {"workflow_dispatch", "workflow_run"}
    assert publish_yml["permissions"] == {"contents": "read"}
    # Only production holds the deploy key, behind its environment, and what
    # production was is read before the approval.
    assert jobs["production"]["environment"] == "production"
    assert jobs["production"]["needs"] == "request" and "environment" not in jobs["request"]
    assert jobs["request"]["permissions"] == {"contents": "read"}
    assert "secrets." not in yaml.safe_dump(jobs["staging"])
    assert text.count("secrets.ASSEMBLED_DEPLOY_KEY") == 2
    # A run that wakes staging is not trusted for what to publish.
    assert "workflow_run.id" not in text and "workflow_run.head_sha" not in text
    assert "github.event.workflow_run.event != 'pull_request'" in jobs["staging"]["if"]
    # Both targets publish in one group, and a publication waiting there,
    # by hand or not, is never cancelled.
    for job in ("staging", "production"):
        assert jobs[job]["concurrency"] == {
            "group": "publish", "queue": "max", "cancel-in-progress": False
        }
    assert jobs["production"]["permissions"] == {
        "contents": "read", "actions": "read", "deployments": "read"
    }
