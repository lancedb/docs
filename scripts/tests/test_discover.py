"""Tests for discovering releases and preparing their edition pull requests.

Disposable repositories stand in for lancedb/lancedb, lancedb/sophon and this
repository, and a fake stands in for GitHub's API. Their versions are
fixtures, not releases: OSS 0.97.0 to 0.99.2 and Enterprise 8.1.0 to 8.6.0.
Every edition is prepared by the real `edition.py prepare`, which builds it
with the assembler.

Run with `make test-assemble`.
"""

import copy
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl

import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import discover  # noqa: E402
import edition  # noqa: E402

WORKFLOWS = SCRIPTS.parent / ".github" / "workflows"
# Allow local stand-ins without adding a production URL-check bypass.
TEST_DOCS_URLS = re.compile(rf"{edition.DOCS_URLS.pattern}|/.+/lancedb-docs\.git", re.IGNORECASE)
NAVIGATION = {
    "navigation": {
        "tabs": [
            {
                "tab": "Documentation",
                "groups": [
                    {"group": "Start", "pages": ["index", "guide"]},
                    {"group": "Enterprise", "pages": ["enterprise/security"]},
                ],
            }
        ]
    }
}
GENEVA_NAV = {"tabs": [{"tab": "Geneva", "groups": [{"group": "Geneva", "pages": ["geneva/index"]}]}]}
INTERNAL = b"Internal engineering notes\n"
BOT = "github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>"
HOLD = "Production serves the earlier site."


@dataclass(frozen=True)
class Gitlink:
    commit: str


def environment() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        GIT_AUTHOR_NAME="Test",
        GIT_AUTHOR_EMAIL="test@example.com",
        GIT_COMMITTER_NAME="Test",
        GIT_COMMITTER_EMAIL="test@example.com",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_TERMINAL_PROMPT="0",
    )
    return env


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    """Discovery's own git calls see no user configuration either."""
    for key in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(key)
    for key, value in environment().items():
        if key.startswith("GIT_"):
            monkeypatch.setenv(key, value)
    monkeypatch.setattr(edition, "DOCS_URLS", TEST_DOCS_URLS)


def git(cwd: Path, *args: str, data: bytes | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, env=environment(), input=data, check=True, capture_output=True
    ).stdout.decode().strip()


def serving(path: Path, bare: bool = False) -> Path:
    """A repository that serves fetches as GitHub does: any commit by its
    hash, and without blobs on request."""
    path.mkdir(parents=True)
    git(path, "init", "-q", *(["--bare"] if bare else []), "-b", "main")
    git(path, "config", "uploadpack.allowAnySHA1InWant", "true")
    git(path, "config", "uploadpack.allowFilter", "true")
    return path


def commit(repo: Path, files: dict, message: str) -> str:
    """A commit of exactly `files`, on no branch."""
    lines = []
    for path, content in sorted(files.items()):
        if isinstance(content, Gitlink):
            lines.append(f"160000 commit {content.commit}\t{path}")
        else:
            lines.append(f"100644 blob {git(repo, 'hash-object', '-w', '--stdin', data=content)}\t{path}")
    index = repo / ".git" / "fixture-index"
    env = {**environment(), "GIT_INDEX_FILE": str(index)}
    subprocess.run(["git", "read-tree", "--empty"], cwd=repo, env=env, check=True)
    subprocess.run(["git", "update-index", "--index-info"], cwd=repo, env=env, check=True,
                   input="".join(f"{line}\n" for line in lines).encode())
    tree = subprocess.run(["git", "write-tree"], cwd=repo, env=env, check=True,
                          capture_output=True).stdout.decode().strip()
    index.unlink()
    return git(repo, "commit-tree", tree, "-m", message)


def annotated(repo: Path, name: str, target: str, force: bool = False) -> str:
    # Made again, a tag gets another message, so a new tag object.
    git(repo, "tag", *(["-f"] if force else []), "-a", "-m", f"{name} again" if force else name, name, target)
    return git(repo, "rev-parse", f"refs/tags/{name}")


def oss_pages(label: str, docs: bool = True) -> dict:
    """A lancedb/lancedb tree, with the Enterprise placeholder the assembler needs."""
    if not docs:
        return {"README.md": b"Before docs/web\n"}
    web = "docs/web/"
    return {
        "README.md": b"Not documentation\n",
        web + "docs.json": json.dumps(NAVIGATION).encode(),
        web + "index.mdx": f"# LanceDB {label}\n".encode(),
        web + "guide.mdx": f"---\ntitle: Guide\n---\n\nShared guide as of {label}.\n".encode(),
        web + "enterprise/security.mdx": b"Open-source security placeholder\n",
    }


def enterprise_pages(label: str, bundled: str | None, docs: bool = True) -> dict:
    files = {"README.md": b"Not documentation\n", "docs/internal/notes.mdx": INTERNAL}
    if docs:
        files["docs/web/enterprise/security.mdx"] = f"Enterprise security for {label}\n".encode()
    if bundled:
        files["src/lancedb"] = Gitlink(bundled)
    return files


def refs(repo: Path) -> dict[str, str]:
    listing = git(repo, "for-each-ref", "--format=%(refname) %(objectname)")
    return dict(line.split(" ") for line in listing.splitlines())


def preferred(config: dict, product: str) -> str | None:
    return next(p for p in config["products"] if p["product"] == product).get("preferred")


class FakeGitHub:
    """GitHub's API as discovery uses it: lancedb/lancedb releases, and this
    repository's pull requests, workflow dispatches and the runs they start."""

    def __init__(self, origin: Path):
        self.origin = origin
        self.releases = {}
        self.pull_requests = []
        self.posts = []
        self.runs = {}
        self.failures = {}

    def head(self, branch: str) -> str | None:
        return refs(self.origin).get(f"refs/heads/{branch}")

    def get(self, endpoint: str):
        path, _, query = endpoint.partition("?")
        params = dict(parse_qsl(query))
        if match := re.fullmatch(r"repos/lancedb/lancedb/releases/tags/(.+)", path):
            return copy.deepcopy(self.releases.get(match[1]))
        if path == "repos/lancedb/docs/pulls":
            assert params["state"] == "all"
            branch = params["head"].removeprefix("lancedb:")
            for pull in self.pull_requests:
                # An open pull request follows its branch.
                if pull["state"] == "open" and self.head(pull["head"]["ref"]):
                    pull["head"]["sha"] = self.head(pull["head"]["ref"])
            return [copy.deepcopy(p) for p in self.pull_requests if p["head"]["ref"] == branch]
        if match := re.fullmatch(r"repos/lancedb/docs/actions/workflows/([a-z]+\.yml)/runs", path):
            runs = self.runs.get((match[1], params["head_sha"]), [])
            return {"total_count": len(runs), "workflow_runs": copy.deepcopy(runs)}
        raise AssertionError(f"unexpected GET {endpoint}")

    def post(self, endpoint: str, payload: dict):
        self.posts.append((endpoint, payload))
        if self.failures.get(endpoint):
            self.failures[endpoint] -= 1
            raise discover.DiscoveryError(f"GitHub POST {endpoint}: Server Error (HTTP 502)")
        if endpoint == "repos/lancedb/docs/pulls":
            sha = self.head(payload["head"])
            if sha is None:
                raise discover.DiscoveryError(f"GitHub POST {endpoint}: Validation Failed (HTTP 422)")
            if any(p["state"] == "open" and p["head"]["ref"] == payload["head"] for p in self.pull_requests):
                raise discover.DiscoveryError(f"GitHub POST {endpoint}: A pull request already exists (HTTP 422)")
            return copy.deepcopy(self.add_pull(payload["head"], sha, payload["title"], payload["body"]))
        if match := re.fullmatch(r"repos/lancedb/docs/actions/workflows/(.+)/dispatches", endpoint):
            self.ran(match[1], self.head(payload["ref"]))
            return None
        raise AssertionError(f"unexpected POST {endpoint}")

    def ran(self, workflow: str, sha: str, conclusion: str | None = None) -> None:
        status = "completed" if conclusion else "queued"
        self.runs.setdefault((workflow, sha), []).append({"status": status, "conclusion": conclusion})

    def add_pull(self, branch: str, sha: str, title: str = "", body: str = "") -> dict:
        number = len(self.pull_requests) + 1
        pull = {
            "number": number, "state": "open", "merged_at": None, "title": title, "body": body,
            "head": {"ref": branch, "sha": sha}, "base": {"ref": "main"},
            "html_url": f"https://github.com/lancedb/docs/pull/{number}",
        }
        self.pull_requests.append(pull)
        return pull

    def dispatched(self) -> list[tuple[str, str]]:
        return [(endpoint.split("/")[-2], payload["ref"]) for endpoint, payload in self.posts
                if endpoint.endswith("/dispatches")]

    def opened(self) -> list[dict]:
        return [payload for endpoint, payload in self.posts if endpoint == "repos/lancedb/docs/pulls"]


@dataclass
class Result:
    findings: list
    commit: str
    report: str

    def __getitem__(self, name: str):
        found = [finding for finding in self.findings if finding.name == name]
        assert len(found) == 1, (name, [f.name for f in self.findings])
        return found[0]

    def states(self) -> dict[str, str]:
        return {finding.name: finding.state for finding in self.findings if finding.version}


class World:
    """Producer repositories with tagged releases, this repository's main and
    a checkout of it, GitHub, and the empty repositories the workflow makes."""

    def __init__(self, base: Path):
        self.lancedb = serving(base / "lancedb")
        self.sophon = serving(base / "sophon")
        self.docs = serving(base / "docs")
        for name in ("assemble.yaml", ".gitignore", "compose.yaml", "discover.yaml"):
            shutil.copy(SCRIPTS.parent / name, self.docs / name)
        for rel, content in {
            "docs/docs.nav.json": json.dumps(GENEVA_NAV),
            "docs/geneva/index.mdx": "# Geneva\n",
            "docs/api-reference/rest/openapi.yml": "openapi: 3.0.0\n",
        }.items():
            (self.docs / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.docs / rel).write_text(content)
        git(self.docs, "add", "-A")
        git(self.docs, "commit", "-q", "-m", "docs")
        self.origin = serving(base / "docs-upstream" / "lancedb-docs.git", bare=True)
        git(self.docs, "remote", "add", "origin", str(self.origin))
        self.publish()
        self.github = FakeGitHub(self.origin)
        self.clones = {}
        for name, upstream in (("lancedb", self.lancedb), ("sophon", self.sophon)):
            clone = base / "work" / f"{name}.git"
            clone.mkdir(parents=True)
            git(clone, "init", "-q", "--bare")
            git(clone, "remote", "add", "origin", f"file://{upstream}")
            self.clones[name] = clone

    # producers

    def lancedb_commit(self, label: str, docs: bool = True) -> str:
        """A commit no release tags, as an Enterprise release may bundle."""
        return commit(self.lancedb, oss_pages(label, docs), label)

    def oss(self, version: str, docs: bool = True, release: str | None = "published") -> str:
        target = commit(self.lancedb, oss_pages(version, docs), f"v{version}")
        annotated(self.lancedb, f"v{version}", target)
        if release:
            self.github.releases[f"v{version}"] = {
                "tag_name": f"v{version}", "draft": False, "prerelease": release == "prerelease",
            }
        return target

    def enterprise(self, version: str, bundled: str | None, docs: bool = True,
                   qualified: str | None = "annotated", tagged: bool = True) -> str:
        target = commit(self.sophon, enterprise_pages(version, bundled, docs), f"v{version}")
        if tagged:
            annotated(self.sophon, f"v{version}", target)
        name = f"release-qualified/v{version}"
        if qualified == "annotated":
            annotated(self.sophon, name, target)
        elif qualified == "lightweight":
            git(self.sophon, "tag", name, target)
        elif qualified:
            annotated(self.sophon, name, qualified)
        return target

    # lancedb/docs

    def publish(self) -> str:
        git(self.docs, "push", "-q", "--force", "origin", "HEAD:refs/heads/main")
        return git(self.docs, "rev-parse", "HEAD")

    def configure(self, enterprise: str | None = None, oss: str | None = None, pages: str | None = None,
                  skip: dict | None = None, holds: list | None = None) -> str:
        config = {
            "products": {
                "enterprise": {"from": enterprise, "skip": (skip or {}).get("enterprise", {})},
                "oss": {"from": oss, "skip": (skip or {}).get("oss", {}), "enterprise_pages": pages},
            },
            "holds": [HOLD] if holds is None else holds,
        }
        (self.docs / "discover.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        git(self.docs, "commit", "-q", "--allow-empty", "-am", "configure discovery")
        return self.publish()

    def sync(self) -> None:
        """Bring the checkout to main, as the workflow checks it out."""
        git(self.docs, "fetch", "-q", "origin", "main")
        git(self.docs, "reset", "-q", "--hard", "FETCH_HEAD")

    def merge(self, number: int) -> str:
        pull = self.github.pull_requests[number - 1]
        head = self.github.head(pull["head"]["ref"])
        git(self.origin, "update-ref", "refs/heads/main", head)
        pull.update(state="closed", merged_at="2026-01-02T03:04:05Z")
        self.sync()
        return head

    def show(self, rev: str, path: str) -> str:
        return git(self.origin, "show", f"{rev}:{path}")

    def run(self, open_prs: bool = False, config: Path | None = None) -> Result:
        findings, main = discover.discover(
            self.docs, self.clones["lancedb"], self.clones["sophon"], self.github, config, open_prs
        )
        return Result(findings, main, discover.report(findings, main, open_prs))

    def remotes(self) -> dict:
        return {name: refs(repo) for name, repo in
                (("lancedb", self.lancedb), ("sophon", self.sophon), ("docs", self.origin))}


@pytest.fixture
def world(tmp_path) -> World:
    return World(tmp_path.resolve())


@pytest.fixture
def released(world) -> World:
    """Enterprise 8.2.0, qualified and bundling a LanceDB commit no release
    tags, which is older than the newest OSS release."""
    world.bundled = world.lancedb_commit("the bundled revision")
    world.oss("0.98.0")
    world.enterprise("8.1.0", world.lancedb_commit("8.1's bundled revision"))
    world.enterprise("8.2.0", world.bundled)
    world.configure(enterprise="8.2.0")
    return world


def saved_enterprise(world: World) -> str:
    """Prepare Enterprise 8.2.0 and merge its pull request."""
    world.run(open_prs=True)
    return world.merge(1)


# discovery starts only where configured


def test_discovery_is_off_until_discover_yaml_names_where_it_starts(world):
    world.oss("0.98.0")
    world.enterprise("8.2.0", world.lancedb_commit("bundled"))

    result = world.run(open_prs=True)

    assert [(f.product, f.state) for f in result.findings] == [("enterprise", "skipped"), ("oss", "skipped")]
    assert "discovery is off" in result.report
    assert world.github.posts == []
    assert "refs/heads/editions/enterprise-8.2.0" not in refs(world.origin)


def test_releases_before_the_start_are_never_prepared(released):
    released.enterprise("8.0.5", released.lancedb_commit("a late maintenance release"))

    result = released.run(open_prs=True)

    assert result.states() == {"Enterprise 8.2.0": "open"}
    assert "2 releases before 8.2.0 (8.0.5 to 8.1.0): discovery starts at 8.2.0, with no backfill" in result.report
    assert [p["head"] for p in released.github.opened()] == ["editions/enterprise-8.2.0"]


# a dry run


def test_a_dry_run_reports_each_releases_exact_inputs_and_writes_nothing(released):
    remotes = released.remotes()
    main = refs(released.docs)["refs/heads/main"]

    result = released.run()

    found = result["Enterprise 8.2.0"]
    assert found.state == "prepare"
    sophon = released.sophon
    assert found.record["release"] == {
        "repository": "lancedb/sophon",
        "tag": "v8.2.0",
        "commit": git(sophon, "rev-parse", "v8.2.0^{commit}"),
        "qualification": {
            "tag": "release-qualified/v8.2.0",
            "object": git(sophon, "rev-parse", "refs/tags/release-qualified/v8.2.0"),
        },
        "bundled_lancedb": released.bundled,
    }
    assert found.record["sources"]["build"]["commit"] == main
    assert f"bundled LanceDB `{released.bundled}`" in result.report
    assert "Would prepare it on `editions/enterprise-8.2.0`" in result.report
    assert "sets the preferred Enterprise edition to 8.2.0 in compose.yaml" in result.report
    assert result.report.startswith("# Release discovery (dry run)")
    # Nothing written to GitHub or any remote, and no worktree left behind.
    assert released.github.posts == []
    assert released.remotes() == remotes
    assert len(git(released.docs, "worktree", "list").splitlines()) == 1
    assert not discover.failed(result.findings)


def test_the_dry_run_client_refuses_to_write(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("ran a command"))

    with pytest.raises(discover.DiscoveryError, match="a dry run writes nothing"):
        discover.GitHub(writes=False).post("repos/lancedb/docs/pulls", {"title": "x"})


def test_a_dry_run_can_try_a_configuration_that_is_not_committed(world, tmp_path):
    world.enterprise("8.2.0", world.lancedb_commit("bundled"))
    trial = tmp_path / "trial.yaml"
    trial.write_text(yaml.safe_dump({"products": {"enterprise": {"from": "8.2.0"}, "oss": {}}, "holds": []}))

    result = world.run(config=trial)

    assert result.states() == {"Enterprise 8.2.0": "prepare"}
    with pytest.raises(SystemExit):
        discover.parse(["--open-prs", "--config", str(trial)])


# eligible releases


def test_a_qualified_enterprise_release_is_prepared_with_the_lancedb_it_bundles(released):
    main = refs(released.origin)["refs/heads/main"]

    result = released.run(open_prs=True)

    found = result["Enterprise 8.2.0"]
    assert (found.state, found.done) == ("open", "opened pull request #1 and started its checks")
    branch = refs(released.origin)["refs/heads/editions/enterprise-8.2.0"]
    assert git(released.origin, "rev-parse", f"{branch}^") == main
    assert refs(released.origin)["refs/heads/main"] == main
    record = yaml.safe_load(released.show(branch, "releases/enterprise-8.2.0.yaml"))
    # The shared pages come from the commit the release bundles, which no OSS
    # release tags and no saved OSS edition holds.
    assert record["release"]["bundled_lancedb"] == record["sources"]["lancedb"]["commit"] == released.bundled
    assert "Shared guide as of the bundled revision." in released.show(branch, "editions/enterprise/8.2.0/guide.mdx")
    assert "Enterprise security for 8.2.0" in released.show(branch, "editions/enterprise/8.2.0/enterprise/security.mdx")
    assert "editions/oss" not in git(released.origin, "ls-tree", "-r", "--name-only", branch)
    changed = set(git(released.origin, "diff", "--name-only", main, branch).splitlines())
    assert {path for path in changed if not path.startswith("editions/enterprise/8.2.0/")} == {
        "compose.yaml", "releases/enterprise-8.2.0.yaml"
    }
    composed = yaml.safe_load(released.show(branch, "compose.yaml"))
    assert (preferred(composed, "enterprise"), preferred(composed, "oss")) == ("8.2.0", None)
    assert "# The version readers get by default." in released.show(branch, "compose.yaml")
    assert git(released.origin, "log", "-1", "--format=%an <%ae>", branch) == BOT
    assert edition.check(released.docs)[1] == []

    (pull,) = released.github.opened()
    assert (pull["title"], pull["head"], pull["base"]) == (
        "docs: save the Enterprise 8.2.0 edition", "editions/enterprise-8.2.0", "main"
    )
    body = pull["body"]
    assert "v8.2.0 and the annotated release-qualified/v8.2.0 tag name the same commit" in body
    assert f"bundled LanceDB `{released.bundled}`" in body
    assert f"tag object `{record['release']['qualification']['object']}`" in body
    assert f"{record['prepared']['files']} files, checksum `{record['prepared']['sha256']}`" in body
    assert "Sets the preferred Enterprise edition to 8.2.0" in body
    assert HOLD in body
    assert "does not merge or publish" in body
    assert INTERNAL.decode().strip() not in body
    assert released.github.dispatched() == [
        ("compose.yml", "editions/enterprise-8.2.0"), ("assemble.yml", "editions/enterprise-8.2.0")
    ]
    assert not discover.failed(result.findings)


def test_an_eligible_oss_release_takes_the_preferred_enterprise_editions_pages(released):
    released.oss("0.97.5")
    released.configure(enterprise="8.2.0", oss="0.98.0", pages="preferred")

    # An OSS edition waits until an Enterprise edition is preferred.
    waiting = released.run(open_prs=True)
    assert waiting["OSS 0.98.0"].state == "waiting"
    assert "compose.yaml names none yet" in waiting["OSS 0.98.0"].reason
    released.merge(1)
    enterprise = git(released.sophon, "rev-parse", "v8.2.0^{commit}")

    result = released.run(open_prs=True)

    assert result.states() == {"Enterprise 8.2.0": "saved", "OSS 0.98.0": "open"}
    assert "1 release before 0.98.0 (0.97.5): discovery starts at 0.98.0, with no backfill" in result.report
    branch = refs(released.origin)["refs/heads/editions/oss-0.98.0"]
    record = yaml.safe_load(released.show(branch, "releases/oss-0.98.0.yaml"))
    assert record["release"]["commit"] == git(released.lancedb, "rev-parse", "v0.98.0^{commit}")
    assert record["release"]["enterprise_pages"] == {"ref": "v8.2.0", "commit": enterprise}
    assert "Shared guide as of 0.98.0." in released.show(branch, "editions/oss/0.98.0/guide.mdx")
    composed = yaml.safe_load(released.show(branch, "compose.yaml"))
    assert (preferred(composed, "enterprise"), preferred(composed, "oss")) == ("8.2.0", "0.98.0")
    body = released.github.opened()[-1]["body"]
    assert "the GitHub release v0.98.0 of lancedb/lancedb is published and not a prerelease" in body
    assert f"Enterprise pages from `v8.2.0` at `{enterprise}`" in body


# releases that are not eligible


def test_unqualified_unpublished_and_prerelease_releases_are_not_prepared(released):
    released.enterprise("8.3.0", released.bundled, qualified=None)
    released.enterprise("8.4.0", released.bundled, tagged=False)
    rc = released.enterprise("8.5.0", released.bundled, qualified=None, tagged=False)
    annotated(released.sophon, "v8.5.0-rc.1", rc)
    released.oss("0.99.0", release=None)
    released.oss("0.99.1", release="prerelease")
    annotated(released.lancedb, "v0.99.2-beta.1", released.lancedb_commit("beta"))
    released.configure(enterprise="8.2.0", oss="0.98.0", pages="preferred")

    result = released.run(open_prs=True)

    assert result.states() == {
        "Enterprise 8.2.0": "open", "Enterprise 8.3.0": "blocked", "Enterprise 8.4.0": "waiting",
        "OSS 0.98.0": "waiting", "OSS 0.99.0": "waiting", "OSS 0.99.1": "skipped",
    }
    assert "lancedb/sophon v8.3.0 is not qualified" in result.report
    assert "the Enterprise release pipeline tags a release last, once it is published" in result.report
    assert "v0.99.0 has no published GitHub release yet" in result.report
    assert "the GitHub release v0.99.1 is marked as a prerelease" in result.report
    assert "1 prerelease from 8.2.0 on: only stable releases get editions" in result.report
    assert "1 prerelease from 0.98.0 on" in result.report
    # Counted, never named.
    assert "rc.1" not in result.report and "beta.1" not in result.report
    assert [p["head"] for p in released.github.opened()] == ["editions/enterprise-8.2.0"]
    assert discover.failed(result.findings)


@pytest.mark.parametrize(
    "case, error",
    [
        ("no docs", "has no docs/web: an edition is built only from a release's own documentation sources"),
        ("bundled lancedb without docs", "has no docs/web"),
        ("no bundled lancedb", "records no bundled LanceDB"),
        ("bundled lancedb missing", "which lancedb/lancedb does not serve"),
        ("qualifies another commit", "but release-qualified/v8.3.0 qualifies"),
        ("lightweight qualification", "release-qualified/v8.3.0 is not an annotated tag"),
    ],
)
def test_absent_docs_ambiguous_pins_and_conflicting_qualification_stop_for_review(released, case, error):
    other = released.enterprise("8.2.5", released.bundled)
    if case == "no docs":
        released.enterprise("8.3.0", released.bundled, docs=False)
    elif case == "bundled lancedb without docs":
        released.enterprise("8.3.0", released.lancedb_commit("no docs", docs=False))
    elif case == "no bundled lancedb":
        released.enterprise("8.3.0", None)
    elif case == "bundled lancedb missing":
        released.enterprise("8.3.0", "1" * 40)
    elif case == "qualifies another commit":
        released.enterprise("8.3.0", released.bundled, qualified=other)
    else:
        released.enterprise("8.3.0", released.bundled, qualified="lightweight")
    remotes = released.remotes()

    result = released.run(open_prs=True)

    found = result["Enterprise 8.3.0"]
    assert found.state == "blocked"
    assert error in found.reason
    assert discover.failed(result.findings)
    # The other releases go ahead, and nothing is pushed for this one.
    assert result["Enterprise 8.2.0"].state == "open"
    assert "refs/heads/editions/enterprise-8.3.0" not in refs(released.origin)
    assert released.remotes()["docs"]["refs/heads/main"] == remotes["docs"]["refs/heads/main"]


def test_oss_stops_when_the_preferred_enterprise_release_tag_has_moved(released):
    saved_enterprise(released)
    annotated(released.sophon, "v8.2.0", released.enterprise("8.9.9", released.bundled), force=True)
    released.configure(oss="0.98.0", pages="preferred")

    result = released.run(open_prs=True)

    assert result["OSS 0.98.0"].state == "blocked"
    assert "the preferred Enterprise edition 8.2.0 was saved from" in result["OSS 0.98.0"].reason
    assert released.github.opened()[1:] == []


def test_an_oss_release_without_docs_stops_for_review(world):
    world.oss("0.97.0", docs=False)
    world.enterprise("8.2.0", world.lancedb_commit("bundled"))
    world.configure(enterprise="8.2.0", oss="0.97.0", pages="preferred")
    world.run(open_prs=True)
    world.merge(1)

    result = world.run(open_prs=True)

    assert result["OSS 0.97.0"].state == "blocked"
    assert "has no docs/web" in result["OSS 0.97.0"].reason


def test_a_skipped_release_is_left_alone(released):
    released.enterprise("8.3.0", released.bundled, docs=False)
    released.configure(enterprise="8.2.0", skip={"enterprise": {"8.3.0": "Published without documentation."}})

    result = released.run(open_prs=True)

    assert result["Enterprise 8.3.0"].state == "skipped"
    assert "discover.yaml skips it: Published without documentation." in result.report
    assert not discover.failed(result.findings)


# repeated discovery, saved editions and existing pull requests


def test_repeated_discovery_reuses_the_open_pull_request_and_keeps_reviewers_commits(released):
    released.run(open_prs=True)
    posts = len(released.github.posts)
    branch = "refs/heads/editions/enterprise-8.2.0"

    again = released.run(open_prs=True)

    assert again["Enterprise 8.2.0"].state == "open" and again["Enterprise 8.2.0"].done is None
    assert len(released.github.posts) == posts
    # A reviewer pushes to the bot's branch.
    review = serving(released.docs.parent / "review")
    git(review, "fetch", "-q", str(released.origin), branch)
    git(review, "checkout", "-q", "FETCH_HEAD")
    (review / "NOTES.md").write_text("Checked the About page.\n")
    git(review, "add", "NOTES.md")
    git(review, "commit", "-q", "-m", "review")
    git(review, "push", "-q", str(released.origin), f"HEAD:{branch}")
    reviewed = refs(released.origin)[branch]
    for workflow in discover.CHECKS:
        released.github.ran(workflow, reviewed)

    third = released.run(open_prs=True)

    assert third["Enterprise 8.2.0"].state == "open"
    assert refs(released.origin)[branch] == reviewed
    assert len(released.github.posts) == posts


def test_a_saved_edition_is_left_alone_after_corrections(released):
    saved_enterprise(released)
    page = released.docs / "editions/enterprise/8.2.0/guide.mdx"
    page.write_text(page.read_text() + "\nA reviewed correction.\n")
    _lines, problems = edition.check(released.docs)
    sha256 = re.search(r"`sha256: ([0-9a-f]{64})`", problems[0])[1]
    path = released.docs / "releases/enterprise-8.2.0.yaml"
    path.write_text(path.read_text().replace(
        "corrections: []\n", f"corrections:\n- summary: Fix the guide.\n  files: [guide.mdx]\n  sha256: {sha256}\n"))
    git(released.docs, "commit", "-q", "-am", "correct Enterprise 8.2.0")
    released.publish()
    corrected = page.read_bytes()
    posts = len(released.github.posts)

    result = released.run(open_prs=True)

    assert result["Enterprise 8.2.0"].state == "saved"
    assert "saved from these release inputs, with 1 correction" in result.report
    assert len(released.github.posts) == posts
    assert page.read_bytes() == corrected
    assert not discover.failed(result.findings)


def test_a_moved_tag_or_recreated_qualification_stops_for_review(released):
    saved_enterprise(released)
    annotated(released.sophon, "release-qualified/v8.2.0", git(released.sophon, "rev-parse", "v8.2.0^{commit}"),
              force=True)

    result = released.run(open_prs=True)

    found = result["Enterprise 8.2.0"]
    assert found.state == "blocked"
    assert "saved from other release inputs: release-qualified/v8.2.0 was tag object" in found.reason
    assert "An edition is never replaced" in found.reason

    moved = released.enterprise("8.2.1", released.bundled, qualified=None, tagged=False)
    annotated(released.sophon, "v8.2.0", moved, force=True)
    annotated(released.sophon, "release-qualified/v8.2.0", moved, force=True)
    assert "v8.2.0 was" in released.run()["Enterprise 8.2.0"].reason


def test_a_saved_release_that_is_no_longer_eligible_stops_for_review(released):
    released.configure(enterprise="8.2.0", oss="0.98.0", pages="preferred")
    saved_enterprise(released)
    released.run(open_prs=True)
    released.merge(2)
    # The release owner marks the OSS release as a prerelease after its edition is saved.
    released.github.releases["v0.98.0"]["prerelease"] = True

    result = released.run(open_prs=True)

    assert result["OSS 0.98.0"].state == "blocked"
    assert "saved, but now the GitHub release v0.98.0 is marked as a prerelease" in result.report


def test_enterprise_pages_that_move_while_they_are_fetched_are_never_used(released, monkeypatch):
    saved_enterprise(released)
    released.configure(oss="0.98.0", pages="preferred")
    approved = git(released.sophon, "rev-parse", "v8.2.0^{commit}")
    replacement = released.enterprise("8.3.0", released.bundled)
    fetch = discover.fetch

    def move_then_fetch(repo, *refspecs):
        # The tag moves after discovery checked it, before the fetch.
        if repo == released.clones["sophon"] and "+refs/tags/v8.2.0:refs/tags/v8.2.0" in refspecs:
            annotated(released.sophon, "v8.2.0", replacement, force=True)
        return fetch(repo, *refspecs)

    monkeypatch.setattr(discover, "fetch", move_then_fetch)
    branches, posts = refs(released.origin), len(released.github.posts)

    result = released.run(open_prs=True)

    found = result["OSS 0.98.0"]
    assert found.state == "blocked"
    assert (f"preparation would read v8.2.0 at {replacement} rather than {approved}: a tag moved while discovery "
            "ran, so nothing is prepared from it") in found.reason
    assert refs(released.origin) == branches
    assert released.github.posts[posts:] == []

    # Next time the tag no longer names what the preferred edition records, which stops OSS as before.
    monkeypatch.setattr(discover, "fetch", fetch)
    assert "the preferred Enterprise edition 8.2.0 was saved from" in released.run(open_prs=True)["OSS 0.98.0"].reason
    assert refs(released.origin) == branches
    # Put back, it is prepared from the recorded commit.
    annotated(released.sophon, "v8.2.0", approved, force=True)
    assert released.run(open_prs=True)["OSS 0.98.0"].state == "open"
    record = yaml.safe_load(released.show(refs(released.origin)["refs/heads/editions/oss-0.98.0"],
                                          "releases/oss-0.98.0.yaml"))
    assert record["release"]["enterprise_pages"] == {"ref": "v8.2.0", "commit": approved}


def test_a_tag_that_moves_before_preparing_is_never_pushed(released, monkeypatch):
    saved_enterprise(released)
    released.configure(oss="0.98.0", pages="preferred")
    approved = git(released.sophon, "rev-parse", "v8.2.0^{commit}")
    replacement = released.enterprise("8.3.0", released.bundled)
    prepare = edition.prepare

    def move_then_prepare(args, repo=edition.REPO_ROOT):
        annotated(released.sophon, "v8.2.0", replacement, force=True)
        discover.fetch(released.clones["sophon"], *discover.tag_refspecs("v8.2.0"))
        return prepare(args, repo)

    monkeypatch.setattr(edition, "prepare", move_then_prepare)
    branches, posts = refs(released.origin), len(released.github.posts)

    result = released.run(open_prs=True)

    found = result["OSS 0.98.0"]
    assert found.failed
    assert f"failed: preparation read v8.2.0 at {replacement} rather than {approved}" in found.done
    assert refs(released.origin) == branches
    assert released.github.posts[posts:] == []
    assert len(git(released.docs, "worktree", "list").splitlines()) == 1


@pytest.mark.parametrize("tag", ["v8.2.0", "release-qualified/v8.2.0"])
def test_enterprise_tags_that_move_while_they_are_fetched_stop_for_review(released, monkeypatch, tag):
    listed = git(released.sophon, "rev-parse", "v8.2.0^{commit}")
    qualified = git(released.sophon, "rev-parse", "refs/tags/release-qualified/v8.2.0")
    other = released.enterprise("8.2.0 again", released.bundled, qualified=None, tagged=False)
    fetch = discover.fetch

    def move_then_fetch(repo, *refspecs):
        if repo == released.clones["sophon"] and "+refs/tags/v8.2.0:refs/tags/v8.2.0" in refspecs:
            if tag == "v8.2.0":
                # Released again from another commit, qualified there too.
                annotated(released.sophon, "v8.2.0", other, force=True)
                annotated(released.sophon, "release-qualified/v8.2.0", other, force=True)
            else:
                annotated(released.sophon, "release-qualified/v8.2.0", listed, force=True)
        return fetch(repo, *refspecs)

    monkeypatch.setattr(discover, "fetch", move_then_fetch)
    branches = refs(released.origin)

    result = released.run(open_prs=True)

    found = result["Enterprise 8.2.0"]
    assert found.state == "blocked"
    assert "a tag moved while discovery ran, so nothing is prepared from it" in found.reason
    if tag == "v8.2.0":
        assert f"v8.2.0 at {other} rather than {listed}" in found.reason
    else:
        now = git(released.sophon, "rev-parse", "refs/tags/release-qualified/v8.2.0")
        assert f"release-qualified/v8.2.0 at {now} rather than {qualified}" in found.reason
    assert refs(released.origin) == branches
    assert released.github.opened() == []


def test_changed_qualification_stops_an_open_pull_request(released):
    released.run(open_prs=True)
    annotated(released.sophon, "release-qualified/v8.2.0", git(released.sophon, "rev-parse", "v8.2.0^{commit}"),
              force=True)

    result = released.run(open_prs=True)

    assert result["Enterprise 8.2.0"].state == "blocked"
    assert "pull request #1 does not hold the edition of these release inputs" in result.report
    assert len(released.github.opened()) == 1


def test_a_closed_pull_request_is_not_reopened(released):
    released.run(open_prs=True)
    released.github.pull_requests[0]["state"] = "closed"
    posts = len(released.github.posts)

    result = released.run(open_prs=True)

    assert result["Enterprise 8.2.0"].state == "declined"
    assert "pull request #1 was closed without merging; reopen it to prepare this release again" in result.report
    assert len(released.github.posts) == posts
    assert not discover.failed(result.findings)
    # Reopened, it is reviewed again.
    released.github.pull_requests[0]["state"] = "open"
    assert released.run(open_prs=True)["Enterprise 8.2.0"].state == "open"


def test_a_branch_without_a_pull_request_that_holds_other_inputs_stops_for_review(released):
    # A branch of that name, pushed by hand.
    main = refs(released.origin)["refs/heads/main"]
    git(released.origin, "update-ref", "refs/heads/editions/enterprise-8.2.0", main)

    result = released.run(open_prs=True)

    assert result["Enterprise 8.2.0"].state == "blocked"
    assert "exists without a pull request, and does not hold the edition of these release inputs" in result.report
    assert released.github.posts == []
    assert refs(released.origin)["refs/heads/editions/enterprise-8.2.0"] == main


def test_a_merged_pull_request_whose_record_main_lacks_stops_for_review(released):
    released.run(open_prs=True)
    released.github.pull_requests[0].update(state="closed", merged_at="2026-01-02T03:04:05Z")

    result = released.run(open_prs=True)

    assert result["Enterprise 8.2.0"].state == "blocked"
    assert "was merged, but main has no record of this release" in result.report


# interrupted and concurrent runs


def test_a_run_interrupted_after_pushing_opens_the_pull_request_next_time(released):
    released.github.failures["repos/lancedb/docs/pulls"] = 1

    first = released.run(open_prs=True)

    assert first["Enterprise 8.2.0"].failed
    assert "Failed: GitHub POST repos/lancedb/docs/pulls" in first.report
    assert discover.failed(first.findings)
    pushed = refs(released.origin)["refs/heads/editions/enterprise-8.2.0"]
    assert released.github.pull_requests == []
    # Not yet: discovery reports what it would do.
    assert released.run()["Enterprise 8.2.0"].state == "resume"

    second = released.run(open_prs=True)

    assert second["Enterprise 8.2.0"].done == "opened pull request #1 and started its checks"
    assert released.github.pull_requests[0]["head"]["sha"] == pushed
    assert refs(released.origin)["refs/heads/editions/enterprise-8.2.0"] == pushed
    assert "An earlier discovery run prepared it" in released.github.pull_requests[0]["body"]
    assert "Sets the preferred Enterprise edition to 8.2.0 in compose.yaml" in released.github.pull_requests[0]["body"]
    assert released.github.dispatched()[-2:] == [
        ("compose.yml", "editions/enterprise-8.2.0"), ("assemble.yml", "editions/enterprise-8.2.0")
    ]


@pytest.mark.parametrize("workflow", ["compose.yml", "assemble.yml"])
def test_a_check_that_did_not_start_is_started_next_time_on_its_own(released, workflow):
    branch = "editions/enterprise-8.2.0"
    dispatch = f"repos/lancedb/docs/actions/workflows/{workflow}/dispatches"
    released.github.failures[dispatch] = 1

    first = released.run(open_prs=True)

    found = first["Enterprise 8.2.0"]
    assert found.state == "open" and found.failed
    assert found.done.startswith(f"opened pull request #1, then failed: could not start {workflow} (GitHub POST")
    assert found.done.endswith("the next run starts what has not run")
    # The other check started anyway.
    assert released.github.dispatched() == [("compose.yml", branch), ("assemble.yml", branch)]
    head, posts = released.github.head(branch), len(released.github.posts)

    second = released.run(open_prs=True)

    found = second["Enterprise 8.2.0"]
    assert (found.state, found.failed) == ("open", False)
    assert found.done == f"started {workflow}, which had not run on its head"
    assert released.github.posts[posts:] == [(dispatch, {"ref": branch})]
    assert len(released.github.pull_requests) == 1 and released.github.head(branch) == head
    assert not discover.failed(second.findings)

    # A check that has run is never started again, whatever its result.
    for runs in released.github.runs.values():
        runs[:] = [{"status": "completed", "conclusion": "failure"}]
    third = released.run(open_prs=True)
    assert third["Enterprise 8.2.0"].done is None
    assert len(released.github.posts) == posts + 1


def test_a_check_that_still_cannot_start_fails_the_run_with_its_name(released):
    dispatch = "repos/lancedb/docs/actions/workflows/assemble.yml/dispatches"
    released.github.failures[dispatch] = 2

    released.run(open_prs=True)
    second = released.run(open_prs=True)

    found = second["Enterprise 8.2.0"]
    assert found.failed and discover.failed(second.findings)
    assert found.done.startswith("failed: could not start assemble.yml (GitHub POST")
    assert "Failed: could not start assemble.yml" in second.report
    assert len(released.github.pull_requests) == 1


def test_concurrent_runs_open_one_pull_request(released, monkeypatch):
    prepare_branch = discover.prepare_branch
    other = []

    def meanwhile(*args, **kwargs):
        made = prepare_branch(*args, **kwargs)
        if not other:
            # Another run prepares the same release, pushes and opens its pull request first.
            monkeypatch.setattr(discover, "prepare_branch", prepare_branch)
            other.append(released.run(open_prs=True))
        return made

    monkeypatch.setattr(discover, "prepare_branch", meanwhile)

    result = released.run(open_prs=True)

    assert other[0]["Enterprise 8.2.0"].done == "opened pull request #1 and started its checks"
    found = result["Enterprise 8.2.0"]
    assert found.state == "open" and not found.failed
    assert "appeared while this run prepared it, so its commits are kept" in result.report
    assert len(released.github.pull_requests) == 1
    assert released.github.head("editions/enterprise-8.2.0") == released.github.pull_requests[0]["head"]["sha"]


def test_a_pull_request_opened_meanwhile_is_reused(released, monkeypatch):
    released.github.failures["repos/lancedb/docs/pulls"] = 1
    released.run(open_prs=True)
    post = released.github.post

    def opened_first(endpoint, payload):
        if endpoint == "repos/lancedb/docs/pulls" and not released.github.pull_requests:
            released.github.add_pull(payload["head"], released.github.head(payload["head"]))
        return post(endpoint, payload)

    monkeypatch.setattr(released.github, "post", opened_first)

    result = released.run(open_prs=True)

    assert result["Enterprise 8.2.0"].done == "opened pull request #1 and started its checks"
    assert len(released.github.pull_requests) == 1


# out-of-order and maintenance releases


def test_out_of_order_and_maintenance_releases_never_change_the_preferred_edition(released):
    released.enterprise("8.4.0", released.bundled)
    released.configure(enterprise="8.4.0")
    released.run(open_prs=True)
    released.merge(1)
    released.enterprise("8.2.9", released.bundled)
    released.enterprise("8.3.1", released.lancedb_commit("8.3.1's bundled revision"))
    released.enterprise("8.5.0", released.bundled)
    released.configure(enterprise="8.2.0")

    result = released.run(open_prs=True)

    assert result.states() == {
        "Enterprise 8.2.0": "open", "Enterprise 8.2.9": "open", "Enterprise 8.3.1": "open",
        "Enterprise 8.4.0": "saved", "Enterprise 8.5.0": "open",
    }
    main = refs(released.origin)["refs/heads/main"]
    for version in ("8.2.0", "8.2.9", "8.3.1", "8.5.0"):
        branch = refs(released.origin)[f"refs/heads/editions/enterprise-{version}"]
        assert released.show(branch, "compose.yaml") == released.show(main, "compose.yaml")
        record = "releases/enterprise-8.4.0.yaml"
        assert released.show(branch, record) == released.show(main, record)
    bodies = {p["head"]: p["body"] for p in released.github.opened()}
    assert "Leaves the preferred Enterprise edition at 8.4.0, a newer release" in bodies["editions/enterprise-8.3.1"]
    assert "This release is newer: to make it the default, change `preferred`" in bodies["editions/enterprise-8.5.0"]


# failures


def test_a_failed_preparation_leaves_main_and_saved_editions_unchanged(released, monkeypatch):
    saved_enterprise(released)
    released.enterprise("8.3.0", released.bundled)
    released.enterprise("8.4.0", released.bundled)
    saved = {p.relative_to(released.docs).as_posix(): p.read_bytes()
             for p in (released.docs / "editions").rglob("*") if p.is_file()}
    prepare = edition.prepare

    def failing(args, repo=edition.REPO_ROOT):
        if args.version == "8.3.0":
            (repo / "editions/enterprise/8.2.0/guide.mdx").write_text("Clobbered\n")
            raise edition.EditionError("the assembler failed")
        return prepare(args, repo)

    monkeypatch.setattr(edition, "prepare", failing)
    main = refs(released.origin)["refs/heads/main"]

    result = released.run(open_prs=True)

    assert result["Enterprise 8.3.0"].failed and "Failed: the assembler failed." in result.report
    assert result["Enterprise 8.4.0"].state == "open"
    assert "refs/heads/editions/enterprise-8.3.0" not in refs(released.origin)
    assert refs(released.origin)["refs/heads/main"] == main
    assert {p.relative_to(released.docs).as_posix(): p.read_bytes()
            for p in (released.docs / "editions").rglob("*") if p.is_file()} == saved
    assert len(git(released.docs, "worktree", "list").splitlines()) == 1
    assert discover.failed(result.findings)


def test_no_pull_request_is_opened_while_main_fails_edition_check(released):
    saved_enterprise(released)
    # A change to a saved edition that no correction records reaches main.
    page = released.docs / "editions/enterprise/8.2.0/guide.mdx"
    page.write_text("Changed without a correction.\n")
    git(released.docs, "commit", "-q", "-am", "unrecorded change")
    released.publish()
    released.enterprise("8.3.0", released.bundled)

    result = released.run(open_prs=True)

    assert result["Enterprise 8.3.0"].failed
    assert "`edition.py check` fails on the new branch: editions/enterprise/8.2.0 has changed" in result.report
    assert "refs/heads/editions/enterprise-8.3.0" not in refs(released.origin)
    assert len(released.github.opened()) == 1


def test_opening_pull_requests_needs_a_checkout_of_mains_newest_commit(released):
    git(released.origin, "update-ref", "refs/heads/main",
        git(released.origin, "commit-tree", "-p", "main", "-m", "moved on", "main^{tree}"))
    remotes = released.remotes()

    with pytest.raises(discover.DiscoveryError, match="nothing was prepared, so run discovery again"):
        released.run(open_prs=True)

    assert released.remotes() == remotes and released.github.posts == []


# configuration


@pytest.mark.parametrize(
    "config, error",
    [
        ({"products": {"enterprise": {}, "oss": {}}}, "sets `products` and `holds`"),
        ({"products": {"enterprise": {}}, "holds": []}, "products sets oss and enterprise, once each"),
        ({"products": {"enterprise": {"from": "8.2"}, "oss": {}}, "holds": []}, "not a version such as 1.2.3"),
        ({"products": {"enterprise": {"from": "8.2.0-rc.1"}, "oss": {}}, "holds": []}, "not a version"),
        ({"products": {"enterprise": {"start": "8.2.0"}, "oss": {}}, "holds": []}, "enterprise sets only from, skip"),
        ({"products": {"enterprise": {"enterprise_pages": "preferred"}, "oss": {}}, "holds": []}, "sets only"),
        ({"products": {"enterprise": {}, "oss": {"from": "0.98.0"}}, "holds": []}, "set oss.enterprise_pages too"),
        ({"products": {"enterprise": {}, "oss": {"enterprise_pages": "latest"}}, "holds": []}, "not `preferred`"),
        ({"products": {"enterprise": {"skip": {"8.2.0": ""}}, "oss": {}}, "holds": []}, "skip maps versions"),
        ({"products": {"enterprise": {"skip": ["8.2.0"]}, "oss": {}}, "holds": []}, "skip maps versions"),
        ({"products": {"enterprise": {}, "oss": {}}, "holds": [""]}, "holds lists one line"),
    ],
)
def test_discover_yaml_is_checked(tmp_path, config, error):
    path = tmp_path / "discover.yaml"
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(discover.DiscoveryError, match=re.escape(error)):
        discover.load_config(path)


def test_the_committed_configuration_prepares_nothing():
    config = discover.load_config(SCRIPTS.parent / "discover.yaml")

    assert {name: options["from"] for name, options in config["products"].items()} == {"oss": None, "enterprise": None}
    assert config["products"]["oss"]["enterprise_pages"] is None


# the workflows


def workflow(name: str) -> dict:
    loaded = yaml.safe_load((WORKFLOWS / name).read_text())
    # YAML 1.1 reads the `on:` key as true.
    loaded["on"] = loaded.pop(True)
    return loaded


def test_discovery_runs_only_by_hand_and_writes_only_when_enabled_on_main():
    discover_yml = workflow("discover.yml")
    jobs = discover_yml["jobs"]

    # No schedule: starting discovery automatically is a later, separate change.
    assert set(discover_yml["on"]) == {"workflow_dispatch"}
    assert discover_yml["on"]["workflow_dispatch"]["inputs"]["mode"]["default"] == "dry-run"
    assert discover_yml["permissions"] == {"contents": "read"}
    assert discover_yml["concurrency"] == {"group": "discover", "queue": "max", "cancel-in-progress": False}
    assert jobs["dry-run"]["if"] == "inputs.mode == 'dry-run'"
    assert jobs["dry-run"]["permissions"] == {"contents": "read", "pull-requests": "read"}
    assert "--open-prs" not in yaml.safe_dump(jobs["dry-run"])
    assert jobs["open-prs"]["if"] == "inputs.mode == 'open-prs'"
    assert jobs["open-prs"]["permissions"] == {
        "contents": "write", "pull-requests": "write", "actions": "write"
    }
    gate = jobs["open-prs"]["steps"][0]
    assert gate["if"] == "github.ref != 'refs/heads/main' || vars.RELEASE_DISCOVERY != 'open-prs'"
    assert "exit 1" in gate["run"]
    text = (WORKFLOWS / "discover.yml").read_text()
    # The only secret is the read-only Enterprise token's key.
    assert set(re.findall(r"secrets\.([A-Z_]+)", text)) == {"SOPHON_DOCS_APP_PRIVATE_KEY"}
    assert text.count("permission-contents: read") == 2
    for job in jobs.values():
        assert any(step.get("with", {}).get("repositories") == "sophon" for step in job["steps"])


def test_a_dispatched_compose_run_checks_a_branch_against_the_main_it_was_made_from():
    step = next(s for s in workflow("compose.yml")["jobs"]["compose"]["steps"]
                if s.get("name") == "Check that saved editions and their records are kept")

    assert 'if [ "$EVENT" = workflow_dispatch ]; then' in step["run"]
    assert "BASE_SHA=$(git merge-base HEAD refs/remotes/origin/main)" in step["run"]
    assert set(discover.CHECKS) == {"compose.yml", "assemble.yml"}
    assert workflow("compose.yml")["jobs"]["compose"]["name"] == discover.CHECKS["compose.yml"]
    assert workflow("assemble.yml")["jobs"]["test"]["name"] == discover.CHECKS["assemble.yml"]
    for name in discover.CHECKS:
        assert "workflow_dispatch" in workflow(name)["on"]
