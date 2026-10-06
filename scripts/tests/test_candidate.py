"""Tests for recording a candidate and publishing it.

Every test publishes into a disposable bare repository standing in for
lancedb/docs, from a clone of it standing in for the workflow's checkout, and
runs `candidate.py` the way the workflows do. Nothing here reaches a real
remote.

Run with `make test-assemble`.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
CANDIDATE = SCRIPTS / "candidate.py"
REPOSITORY = "lancedb/docs"
WORKFLOW = ".github/workflows/assemble.yml"

SITE = {
    "docs.json": b'{"name": "LanceDB"}\n',
    "index.mdx": b"# Home\n",
    # Published byte for byte: no line-ending conversion ...
    "guide/page.mdx": b"line one\r\nline two\r\n",
    # ... and no ignore rules, even the site's own.
    ".gitignore": b"*.png\n",
    "static/logo.png": bytes(range(256)),
    ".cursor/rules/.cursorrules": b"hidden, and published\n",
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
    `deploy-freeze`, and a clone of it to publish from."""
    seed = tmp_path / "seed"
    seed.mkdir()
    git(seed, "init", "-q", "-b", "main")
    docs_commit = commit(seed, {"README.md": b"docs\n"}, "main")
    git(seed, "checkout", "-q", "--orphan", "assembled")
    git(seed, "rm", "-q", "-rf", ".")
    old = commit(seed, {"docs.json": b"{}\n", "index.mdx": b"# Old\n"}, "assemble 1234567")
    git(seed, "checkout", "-q", "-b", "deploy-freeze", "main")
    freeze = commit(seed, {"docs/docs.json": b"{}\n"}, "production as it is")
    bare = tmp_path / "docs.git"
    git(tmp_path, "clone", "-q", "--bare", str(seed), str(bare))
    workspace = tmp_path / "checkout"
    git(tmp_path, "clone", "-q", str(bare), str(workspace))
    return {
        "bare": bare,
        "workspace": workspace,
        "docs_commit": docs_commit,
        "assembled": old,
        "deploy_freeze": freeze,
    }


SOURCES = {"lancedb": "1" * 40, "enterprise": "2" * 40}


def sources_line(docs_commit: str, **override: str) -> str:
    return " ".join(f"{k}={v}" for k, v in {**SOURCES, "build": docs_commit, **override}.items())


class Candidate:
    """One recorded candidate: its artifact directory and its run's evidence."""

    def __init__(
        self, base: Path, docs_commit: str, files=SITE, run_id="1001", sources=None, **run
    ):
        self.dir = base / f"candidate-{run_id}"
        self.site = self.dir / "site"
        self.run_json = base / f"run-{run_id}.json"
        self.run_id = run_id
        self.docs_commit = docs_commit
        self.sources = sources or sources_line(docs_commit)
        if isinstance(files, Path):
            shutil.copytree(files, self.site)
        else:
            write_files(self.site, files)
        self.run = {
            "id": int(run_id),
            "path": WORKFLOW,
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

    def record(self, refs=("lancedb=main", "enterprise=main"), sources=None, context=True):
        env = environment()
        if context:
            env.update(
                GITHUB_REPOSITORY=REPOSITORY,
                GITHUB_RUN_ID=self.run_id,
                GITHUB_RUN_ATTEMPT="1",
                GITHUB_EVENT_NAME=self.run["event"],
                GITHUB_REF=f"refs/heads/{self.run['head_branch']}",
                GITHUB_SHA=self.docs_commit,
                GITHUB_WORKFLOW_REF=f"{REPOSITORY}/{WORKFLOW}@refs/heads/main",
            )
        command = [sys.executable, str(CANDIDATE), "record", str(self.site)]
        command += ["--sources", sources or self.sources]
        for ref in refs:
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
    target: str = "production",
    run_id: str | None = None,
    site_sha256: str | None = None,
    content_dir: str = "/docs",
    artifact: Path | None | bool = True,
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    """Run the publish command as the Publish workflow does."""
    command = [sys.executable, str(CANDIDATE), "publish", "--target", target]
    command += ["--run-id", run_id or candidate.run_id]
    command += ["--site-sha256", site_sha256 or candidate.checksum]
    command += ["--run", str(candidate.run_json), "--repository", REPOSITORY]
    command += ["--content-dir", content_dir]
    if artifact is True:
        command += ["--candidate", str(candidate.dir)]
    elif artifact:
        command += ["--candidate", str(artifact)]
    return subprocess.run(
        command, cwd=remote["workspace"], env=env or environment(), capture_output=True, text=True
    )


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
    assert manifest["sources"] == {**SOURCES, "build": remote["docs_commit"]}
    assert manifest["refs"] == {"lancedb": "main", "enterprise": "main"}
    assert manifest["run"]["run_id"] == "1001"
    assert manifest["run"]["sha"] == remote["docs_commit"]


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
# publishing
# --------------------------------------------------------------------------- #


def test_production_publishes_exactly_the_candidate(tmp_path, remote):
    candidate = Candidate(tmp_path, remote["docs_commit"])
    before = refs(remote["bare"])

    result = publish(remote, candidate)

    assert result.returncode == 0, result.stderr
    after = refs(remote["bare"])
    head = after.pop("refs/heads/assembled")
    before.pop("refs/heads/assembled")
    assert after == before, "only assembled may change"
    assert git(remote["bare"], "rev-parse", f"{head}^") == remote["assembled"]
    assert published(remote["bare"], head) == SITE
    message = git(remote["bare"], "log", "-1", "--format=%B", head)
    assert f"Candidate-Site-SHA256: {candidate.checksum}" in message
    assert "Candidate-Run: 1001" in message
    assert f"Sources: {sources_line(remote['docs_commit'])}" in message
    assert "Refs: lancedb=main enterprise=main" in message
    assert json.loads(result.stdout)["commit"] == head


@pytest.mark.parametrize("content_dir", ["/", "."])
def test_root_content_directory(tmp_path, remote, content_dir):
    candidate = Candidate(tmp_path, remote["docs_commit"])

    result = publish(remote, candidate, content_dir=content_dir)

    assert result.returncode == 0, result.stderr
    assert published(remote["bare"], "assembled", prefix="") == SITE


def test_staging_writes_only_the_staging_branch(tmp_path, remote):
    # Staging also takes a manual run on another branch that read pull
    # request heads: a hosted preview before anything merges.
    candidate = Candidate(
        tmp_path, remote["docs_commit"], event="workflow_dispatch", head_branch="jack/a3-move"
    )
    candidate.record(refs=("lancedb=refs/pull/4122/head", "enterprise=refs/pull/7681/head"))
    candidate.checksum = candidate.manifest["site_sha256"]
    before = refs(remote["bare"])

    result = publish(remote, candidate, target="staging")

    assert result.returncode == 0, result.stderr
    after = refs(remote["bare"])
    staging = after.pop("refs/heads/staging")
    assert after == before, "only staging may change"
    assert published(remote["bare"], staging) == SITE


def refuse_unless_main(c):
    c.record(refs=("lancedb=refs/pull/4122/head", "enterprise=main"))


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
    "built on another branch": (
        lambda c: c.edit_run(head_branch="jack/a3-move"),
        {},
        "production takes candidates built on main",
    ),
    "producer read off main": (refuse_unless_main, {}, "production takes producers' main"),
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


@pytest.mark.parametrize("target", ["staging", "production"])
def test_pull_request_candidates_are_refused_for_both_targets(tmp_path, remote, target):
    candidate = Candidate(tmp_path, remote["docs_commit"], event="pull_request")
    before = refs(remote["bare"])

    result = publish(remote, candidate, target=target)

    assert result.returncode == 1
    assert "was a pull_request run" in result.stderr
    assert refs(remote["bare"]) == before


def assemble_sources(base: Path, index: bytes) -> dict:
    """lancedb, sophon and docs checkouts with the shape assemble.yaml reads."""
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
            "docs/web/index.mdx": index,
            "docs/web/enterprise/security.mdx": b"Open-source security\n",
        },
        "sophon": {"docs/web/enterprise/security.mdx": b"Enterprise security\n"},
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
    path: {repos["sophon"] / "docs/web"}
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


def test_promotion_publishes_the_checked_output_not_newer_sources(tmp_path, remote):
    base = tmp_path.resolve()
    repos = assemble_sources(base / "sources", b"# Release one\n")
    sources = assemble(base, repos, base / "built")
    docs_commit = sources.split("build=")[1]
    candidate = Candidate(base, docs_commit, files=base / "built", run_id="2001", sources=sources)
    checked = {
        p.relative_to(candidate.site).as_posix(): p.read_bytes()
        for p in candidate.site.rglob("*")
        if p.is_file()
    }
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
    assert published(remote["bare"], "assembled") == checked
    assert checked["index.mdx"] == b"# Release one\n"
    assert checked["enterprise/security.mdx"] == b"Enterprise security\n"

    # Nor can the run's ID carry the newer build's checksum.
    before = refs(remote["bare"])
    refused = publish(remote, candidate, site_sha256=checksum(newer_files))
    assert refused.returncode == 1
    assert "run 2001 recorded candidate" in refused.stderr
    assert refs(remote["bare"]) == before


def test_rollback_republishes_an_earlier_candidate_after_its_artifact_expired(tmp_path, remote):
    first = Candidate(tmp_path, remote["docs_commit"], run_id="1001")
    second = Candidate(
        tmp_path, remote["docs_commit"], files={**SITE, "index.mdx": b"# Broken\n"}, run_id="1002"
    )
    assert publish(remote, first).returncode == 0
    first_commit = git(remote["bare"], "rev-parse", "assembled")
    assert publish(remote, second).returncode == 0
    shutil.rmtree(first.dir)

    result = publish(remote, first)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["from"] == f"earlier publication {first_commit}"
    head = git(remote["bare"], "rev-parse", "assembled")
    assert published(remote["bare"], head) == SITE
    assert git(remote["bare"], "rev-parse", f"{head}^{{tree}}") == git(
        remote["bare"], "rev-parse", f"{first_commit}^{{tree}}"
    )
    history = git(remote["bare"], "log", "--format=%s", "assembled").splitlines()
    assert history[:3] == [
        f"Publish candidate {first.checksum[:12]} to production",
        f"Publish candidate {second.checksum[:12]} to production",
        f"Publish candidate {first.checksum[:12]} to production",
    ]


def test_an_earlier_publication_must_still_match_its_checksum(tmp_path, remote):
    candidate = Candidate(tmp_path, remote["docs_commit"], run_id="1003")
    # Someone pushes a commit claiming to publish this candidate, with other files.
    forger = tmp_path / "forger"
    git(tmp_path, "clone", "-q", "-b", "assembled", str(remote["bare"]), str(forger))
    git(forger, "rm", "-q", "-rf", ".")
    commit(
        forger,
        {"docs/index.mdx": b"# Not the candidate\n"},
        f"Publish candidate\n\nCandidate-Site-SHA256: {candidate.checksum}\n"
        f"Candidate-Run: 1003\nSources: {sources_line(remote['docs_commit'])}\n"
        "Refs: lancedb=main enterprise=main\n",
    )
    git(forger, "push", "-q", "origin", "assembled")
    shutil.rmtree(candidate.dir)
    before = refs(remote["bare"])

    result = publish(remote, candidate)

    assert result.returncode == 1
    assert "is not candidate" in result.stderr
    assert refs(remote["bare"]) == before


def test_a_branch_moved_by_someone_else_is_never_overwritten(tmp_path, remote):
    candidate = Candidate(tmp_path, remote["docs_commit"])
    # Another writer moves `assembled` just before this publication pushes.
    mover = tmp_path / "mover"
    git(tmp_path, "clone", "-q", "-b", "assembled", str(remote["bare"]), str(mover))
    moved = commit(mover, {"index.mdx": b"# Someone else\n"}, "someone else")
    shim = tmp_path / "bin"
    shim.mkdir()
    real_git = shutil.which("git")
    (shim / "git").write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = push ]; then "{real_git}" -C "{mover}" push -q origin assembled; fi\n'
        f'exec "{real_git}" "$@"\n'
    )
    (shim / "git").chmod(0o755)

    result = publish(remote, candidate, env=environment(PATH=f"{shim}:{os.environ['PATH']}"))

    assert result.returncode == 1
    assert "git push failed" in result.stderr
    assert git(remote["bare"], "rev-parse", "assembled") == moved
