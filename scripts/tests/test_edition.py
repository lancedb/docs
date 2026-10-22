"""Tests for preparing saved editions from release records.

Disposable repositories stand in for LanceDB, Enterprise and this
repository. Their versions are fixtures, not releases: OSS 0.94.0 to 0.99.0 and
Enterprise 8.1.0 to 8.10.0, each built to exercise one case. The docs fixture
uses this repository's own assemble.yaml and .gitignore, and every edition is
built by the real assembler.

Run with `make test-assemble`.
"""

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import edition  # noqa: E402

# Allow local stand-ins without adding a production URL-check bypass.
LANCEDB_DOCS_URLS = edition.DOCS_URLS
TEST_DOCS_URLS = re.compile(rf"{LANCEDB_DOCS_URLS.pattern}|/.+/lancedb-docs\.git", re.IGNORECASE)

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
MARKER = bytes(range(256))
INTERNAL = b"Internal engineering notes\n"


@dataclass(frozen=True)
class Gitlink:
    commit: str


@dataclass(frozen=True)
class Symlink:
    target: str


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
    """The builder's own git calls see no user configuration either."""
    for key in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(key)
    for key, value in environment().items():
        if key.startswith("GIT_"):
            monkeypatch.setenv(key, value)


@pytest.fixture(autouse=True)
def stand_in_for_lancedb_docs(monkeypatch):
    monkeypatch.setattr(edition, "DOCS_URLS", TEST_DOCS_URLS)


def git(cwd: Path, *args: str, data: bytes | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, env=environment(), input=data, check=True, capture_output=True
    ).stdout.decode().strip()


def snapshot(repo: Path, files: dict, message: str) -> str:
    """Commit exactly `files` on main, whatever the working tree holds."""
    lines = []
    for path, content in sorted(files.items()):
        if isinstance(content, Gitlink):
            lines.append(f"160000 commit {content.commit}\t{path}")
            continue
        data = content.target.encode() if isinstance(content, Symlink) else content
        blob = git(repo, "hash-object", "-w", "--stdin", data=data)
        lines.append(f"{'120000' if isinstance(content, Symlink) else '100644'} blob {blob}\t{path}")
    index = repo / ".git" / "fixture-index"
    env = {**environment(), "GIT_INDEX_FILE": str(index)}
    subprocess.run(["git", "read-tree", "--empty"], cwd=repo, env=env, check=True)
    subprocess.run(["git", "update-index", "--index-info"], cwd=repo, env=env, check=True,
                   input="".join(f"{line}\n" for line in lines).encode())
    tree = subprocess.run(["git", "write-tree"], cwd=repo, env=env, check=True,
                          capture_output=True).stdout.decode().strip()
    index.unlink()
    parent = subprocess.run(["git", "rev-parse", "--verify", "--quiet", "HEAD"], cwd=repo,
                            env=environment(), capture_output=True).stdout.decode().strip()
    commit = git(repo, "commit-tree", tree, *(["-p", parent] if parent else []), "-m", message)
    git(repo, "update-ref", "refs/heads/main", commit)
    return commit


def annotated(repo: Path, name: str, commit: str) -> str:
    git(repo, "tag", "-a", "-m", name, name, commit)
    return git(repo, "rev-parse", f"refs/tags/{name}")


def new_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    git(path, "init", "-q", "-b", "main")
    return path


def oss_pages(label: str) -> dict:
    """Minimal docs/web fixture, including the required Enterprise placeholder."""
    web = "docs/web/"
    guide = (
        "---\ntitle: Guide\n---\n\n"
        'import { PyGuide } from "/snippets/guide.mdx";\n\n'
        f"Shared guide as of {label}.\n\n"
        '<CodeBlock filename="Python" language="python" icon="python">\n{PyGuide}\n</CodeBlock>\n\n'
        "{/*\nDraft for the next release.\n\n## Coming soon {#coming-soon}\n*/}\n"
    )
    return {
        "README.md": b"Not documentation\n",
        "python/lancedb.py": b"print('library code')\n",
        web + "docs.json": json.dumps(NAVIGATION).encode(),
        web + "index.mdx": f"# LanceDB {label}\n".encode(),
        web + "guide.mdx": guide.encode(),
        web + "snippets/guide.mdx": f"export const PyGuide = {json.dumps(f'print({label!r})' + chr(10))};\n".encode(),
        web + "static/assets/marker.svg": MARKER,
        web + "enterprise/security.mdx": b"Open-source security placeholder\n",
        web + ".mintignore": b"draft.mdx\n",
        web + "draft.mdx": b"# Not released yet\n",
    }


def enterprise_pages(label: str, bundled: str | None, **extra) -> dict:
    files = {
        "README.md": b"Not documentation\n",
        "docs/ci.md": INTERNAL,
        "docs/internal/notes.mdx": INTERNAL,
        "docs/web/enterprise/security.mdx": f"Enterprise security for {label}\n".encode(),
        **extra,
    }
    if bundled:
        files["src/lancedb"] = Gitlink(bundled)
    return files


@pytest.fixture(scope="module")
def sources(tmp_path_factory) -> dict:
    """LanceDB and Enterprise repositories with tagged fixture releases. Shared, so
    tests only read them."""
    base = tmp_path_factory.mktemp("sources").resolve()
    lancedb = new_repo(base / "lancedb")
    c = {}
    c["no-docs"] = snapshot(lancedb, {"README.md": b"Before docs/web\n"}, "no docs yet")
    git(lancedb, "tag", "v0.97.0", c["no-docs"])
    c["bundled"] = snapshot(lancedb, oss_pages("the bundled revision"), "bundled by 8.1")
    c["ignored"] = snapshot(lancedb, {**oss_pages("0.95.0"), "docs/web/scratch/notes.mdx": b"x\n"}, "0.95")
    git(lancedb, "tag", "v0.95.0", c["ignored"])
    control = {"docs/web/.gitattributes": b"* -text\n", "docs/web/static/.gitignore": b"*.svg\n"}
    c["control"] = snapshot(lancedb, {**oss_pages("0.94.0"), **control}, "0.94")
    git(lancedb, "tag", "v0.94.0", c["control"])
    c["0.98.0"] = snapshot(lancedb, oss_pages("0.98.0"), "0.98")
    annotated(lancedb, "v0.98.0", c["0.98.0"])
    c["0.99.0"] = snapshot(lancedb, oss_pages("0.99.0"), "0.99")
    annotated(lancedb, "v0.99.0", c["0.99.0"])
    c["main"] = snapshot(lancedb, oss_pages("unreleased main"), "main moves on")
    # A working tree that differs from every commit.
    (lancedb / "docs/web").mkdir(parents=True)
    (lancedb / "docs/web/guide.mdx").write_text("Uncommitted text\n")

    enterprise_repo = new_repo(base / "enterprise_repo")
    e = {}

    def release(version: str, files: dict, qualified: str | None = "self") -> str:
        commit = snapshot(enterprise_repo, files, version)
        annotated(enterprise_repo, f"v{version}", commit)
        if qualified == "lightweight":
            git(enterprise_repo, "tag", f"release-qualified/v{version}", commit)
        elif qualified:
            annotated(enterprise_repo, f"release-qualified/v{version}", commit if qualified == "self" else qualified)
        e[version] = commit
        return commit

    release("8.1.0", enterprise_pages("8.1", c["bundled"]))
    release("8.2.0", enterprise_pages("8.2", c["0.99.0"]))
    release("8.3.0", enterprise_pages("8.3", c["0.99.0"]), qualified=None)
    release("8.4.0", enterprise_pages("8.4", c["0.99.0"]), qualified=e["8.3.0"])
    release("8.5.0", enterprise_pages("8.5", c["no-docs"]))
    release("8.6.0", enterprise_pages("8.6", None))
    release("8.7.0", enterprise_pages("8.7", c["0.99.0"], **{"docs/web/.notes.mdx": INTERNAL}))
    release("8.8.0", enterprise_pages("8.8", c["0.99.0"], **{"docs/web/enterprise/ci.md": Symlink("../../ci.md")}))
    release("8.9.0", enterprise_pages("8.9", "1" * 40))
    release("8.10.0", enterprise_pages("8.10", c["0.99.0"]), qualified="lightweight")
    e["main"] = snapshot(enterprise_repo, enterprise_pages("unreleased main", c["main"]), "main moves on")
    return {"lancedb": lancedb, "enterprise_repo": enterprise_repo, "oss": c, "enterprise": e}


@pytest.fixture
def docs(tmp_path) -> Path:
    return docs_checkout(tmp_path.resolve() / "docs")


def docs_checkout(path: Path) -> Path:
    """Docs checkout with a local bare origin standing in for canonical main."""
    repo = new_repo(path)
    for name in ("assemble.yaml", ".gitignore"):
        shutil.copy(SCRIPTS.parent / name, repo / name)
    files = {
        "docs/docs.nav.json": json.dumps(GENEVA_NAV),
        "docs/geneva/index.mdx": "# Geneva\n",
        "docs/api-reference/rest/openapi.yml": "openapi: 3.0.0\n",
    }
    for rel, content in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(content)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "docs")
    origin = path.parent / f"{path.name}-upstream" / "lancedb-docs.git"
    git(path.parent, "init", "-q", "--bare", "-b", "main", str(origin))
    git(repo, "remote", "add", "origin", str(origin))
    on_main(repo)
    return repo


def on_main(docs: Path) -> str:
    """Make HEAD available through the canonical-main stand-in."""
    git(docs, "push", "-q", "--force", "origin", "HEAD:refs/heads/main")
    return git(docs, "rev-parse", "HEAD")


def prepare(docs: Path, sources: dict, *argv: str, lancedb: Path | None = None) -> str:
    args = edition.parse([
        "prepare", *argv,
        "--lancedb", str(lancedb or sources["lancedb"]), "--enterprise-repo", str(sources["enterprise_repo"]),
    ])
    return edition.prepare(args, repo=docs)


def saved(root: Path) -> dict[str, bytes]:
    """Every file under `root`, hidden ones included."""
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def checksum(files: dict[str, bytes]) -> str:
    listing = "".join(f"{hashlib.sha256(files[p]).hexdigest()}  {p}\n" for p in sorted(files))
    return hashlib.sha256(listing.encode()).hexdigest()


def record(docs: Path, name: str) -> dict:
    return yaml.safe_load((docs / "releases" / f"{name}.yaml").read_text())


def state(docs: Path) -> dict[str, bytes | None]:
    """What preparation may change: editions, records and leftovers in build/,
    directories included."""
    found = {}
    for top in ("editions", "releases", "build"):
        root = docs / top
        if root.exists():
            found[top] = None
            for path in sorted(root.rglob("*")):
                found[f"{top}/{path.relative_to(root).as_posix()}"] = (
                    path.read_bytes() if path.is_file() else None
                )
    return found


def correct(docs: Path, name: str, page: str, text: str) -> str:
    """Apply the README's edit-and-record correction workflow."""
    product, version = name.split("-")
    (docs / "editions" / product / version / page).write_text(text)
    return record_correction(docs, name, page)


def record_correction(docs: Path, name: str, page: str) -> str:
    """Add a correction entry with the checksum `check` reports for the edition."""
    product, version = name.split("-")
    _lines, problems = edition.check(docs)
    reported = [p for p in problems if p.startswith(f"editions/{product}/{version} has changed")]
    sha256 = re.search(r"`sha256: ([0-9a-f]{64})`", reported[0])[1]
    add_correction(docs, name, page, sha256)
    return sha256


def add_correction(docs: Path, name: str, page: str, sha256: str) -> None:
    path = docs / "releases" / f"{name}.yaml"
    entry = f"- summary: Fix {page}.\n  files: [{page}]\n  sha256: {sha256}\n"
    path.write_text(path.read_text().replace("corrections: []\n", "corrections:\n") + entry)


def commit_all(docs: Path, message: str) -> str:
    git(docs, "add", "-A")
    git(docs, "commit", "-q", "-m", message)
    return git(docs, "rev-parse", "HEAD")


# preparing editions


def test_oss_edition_is_built_from_its_tag_and_named_enterprise_pages(docs, sources):
    out = prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")

    files = saved(docs / "editions/oss/0.98.0")
    assert sorted(files) == [
        ".mintignore", "api-reference/rest/openapi.yml", "docs.json", "draft.mdx",
        "enterprise/security.mdx", "geneva/index.mdx", "guide.mdx", "index.mdx",
        "static/assets/marker.svg",
    ]
    guide = files["guide.mdx"].decode()
    assert "Shared guide as of 0.98.0." in guide
    assert "unreleased main" not in guide and "Uncommitted" not in guide
    assert "```python Python icon=\"python\"\nprint('0.98.0')\n```" in guide
    assert "/snippets/" not in guide
    assert "{/*\nDraft for the next release." in guide
    assert files[".mintignore"] == b"draft.mdx\n"
    assert files["static/assets/marker.svg"] == MARKER
    assert files["enterprise/security.mdx"] == b"Enterprise security for 8.1\n"
    assert json.loads(files["docs.json"])["navigation"]["tabs"][1]["tab"] == "Geneva"

    oss, ent = sources["oss"], sources["enterprise"]
    rec = record(docs, "oss-0.98.0")
    assert rec["release"] == {
        "repository": "lancedb/lancedb",
        "tag": "v0.98.0",
        "commit": oss["0.98.0"],
        "enterprise_pages": {"ref": "v8.1.0", "commit": ent["8.1.0"]},
    }
    assert rec["sources"] == {
        "lancedb": {"repository": "lancedb/lancedb", "path": "docs/web", "commit": oss["0.98.0"]},
        "enterprise": {"repository": "lancedb/sophon", "path": "docs/web", "commit": ent["8.1.0"]},
        "build": {"repository": "lancedb/docs", "path": "docs", "commit": git(docs, "rev-parse", "origin/main")},
        "openapi": {
            "repository": "lance-format/lance-namespace",
            "release": yaml.safe_load((docs / "assemble.yaml").read_text())["openapi"]["release"],
            "path": "docs/src/spec.yaml",
            "sha256": hashlib.sha256(b"openapi: 3.0.0\n").hexdigest(),
        },
    }
    assert rec["prepared"] == {"files": len(files), "sha256": checksum(files)}
    assert rec["corrections"] == []
    assert "prepared oss 0.98.0: 9 files" in out
    assert not (docs / "build").exists()


def test_enterprise_edition_reads_the_lancedb_it_bundles(docs, sources):
    bundled = sources["oss"]["bundled"]
    prepare(docs, sources, "enterprise", "8.1.0", "--bundled-lancedb", bundled)

    files = saved(docs / "editions/enterprise/8.1.0")
    # The bundled revision has no OSS release tag or saved edition.
    assert "Shared guide as of the bundled revision." in files["guide.mdx"].decode()
    assert files["enterprise/security.mdx"] == b"Enterprise security for 8.1\n"
    assert not (docs / "editions/oss").exists()
    assert not any(INTERNAL in data for data in files.values())

    rec = record(docs, "enterprise-8.1.0")
    enterprise_repo = sources["enterprise_repo"]
    assert rec["release"] == {
        "repository": "lancedb/sophon",
        "tag": "v8.1.0",
        "commit": sources["enterprise"]["8.1.0"],
        "qualification": {
            "tag": "release-qualified/v8.1.0",
            "object": git(enterprise_repo, "rev-parse", "refs/tags/release-qualified/v8.1.0"),
        },
        "bundled_lancedb": bundled,
    }
    assert rec["sources"]["lancedb"]["commit"] == bundled
    assert rec["sources"]["enterprise"]["commit"] == sources["enterprise"]["8.1.0"]


@pytest.mark.parametrize(
    "argv, error",
    [
        (["oss", "0.97.0", "--enterprise", "v8.1.0"], "has no docs/web"),
        (["oss", "0.96.0", "--enterprise", "v8.1.0"], "has no tag v0.96.0"),
        (["oss", "1.0.0-beta.1", "--enterprise", "v8.1.0"], "not a stable release version"),
        (["oss", "0.98.0"], "name the Enterprise tag or commit"),
        (["oss", "0.98.0", "--enterprise", "main"], "not a branch"),
        (["oss", "0.98.0", "--enterprise", "v8.1.0", "--bundled-lancedb", "1" * 40], "Enterprise releases only"),
        (["oss", "0.95.0", "--enterprise", "v8.1.0"], ".gitignore would leave out"),
        (["oss", "0.94.0", "--enterprise", "v8.1.0"],
         r"would hold \['.gitattributes', 'static/.gitignore'\], which change how Git stores it"),
        (["enterprise", "8.1.0", "--enterprise", "v8.2.0"], "applies to OSS releases"),
        (["enterprise", "8.3.0"], "is not qualified"),
        (["enterprise", "8.4.0"], "but release-qualified/v8.4.0 qualifies"),
        (["enterprise", "8.10.0"], "release-qualified/v8.10.0 is not an annotated tag"),
        (["enterprise", "8.5.0"], "has no docs/web"),
        (["enterprise", "8.6.0"], "records no bundled LanceDB"),
        (["enterprise", "8.1.0", "--bundled-lancedb", "1" * 40], "disagrees with Enterprise v8.1.0"),
        (["enterprise", "8.9.0"], "fetch it first"),
        (["enterprise", "8.7.0"], r"\.notes\.mdx is hidden"),
        (["enterprise", "8.8.0"], "ci.md is a symlink"),
    ],
)
def test_preparation_stops_and_saves_nothing(docs, sources, argv, error):
    with pytest.raises(edition.EditionError, match=error):
        prepare(docs, sources, *argv)

    assert state(docs) == {}


def test_bundled_lancedb_is_named_in_full(docs, sources):
    short = sources["oss"]["bundled"][:8]
    with pytest.raises(edition.EditionError, match=f"takes the full 40-character commit hash .*, not '{short}'"):
        prepare(docs, sources, "enterprise", "8.1.0", "--bundled-lancedb", short)
    assert state(docs) == {}


def test_a_root_without_recorded_provenance_stops_preparation(docs, sources):
    config = yaml.safe_load((docs / "assemble.yaml").read_text())
    config["roots"].append({"name": "notes", "path": "notes", "role": "reference"})
    (docs / "assemble.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    commit_all(docs, "another root")
    on_main(docs)

    with pytest.raises(edition.EditionError, match=r"assemble.yaml has roots \['lancedb', 'enterprise', "
                       r"'build', 'notes'\], but edition.py records the sources of"):
        prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    assert state(docs) == {}


# the docs input commit


def test_docs_inputs_come_from_main_not_from_the_branch(docs, sources):
    main = git(docs, "rev-parse", "origin/main")
    spec = yaml.safe_load((docs / "assemble.yaml").read_text())["openapi"]["release"]
    # A pull request's branch changes a docs-owned page and assemble.yaml in a
    # commit main does not have, and the page again without committing it.
    git(docs, "switch", "-q", "-c", "editions")
    (docs / "docs/geneva/index.mdx").write_text("# Geneva, on a branch\n")
    config = (docs / "assemble.yaml").read_text()
    (docs / "assemble.yaml").write_text(config.replace(f"release: {spec}\n", "release: v9.9.9\n"))
    commit_all(docs, "unmerged changes")
    (docs / "docs/geneva/index.mdx").write_text("# Geneva, uncommitted\n")

    prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    git(docs, "add", "editions", "releases")
    git(docs, "commit", "-q", "-m", "oss 0.98.0")
    prepare(docs, sources, "enterprise", "8.1.0")

    for name in ("oss-0.98.0", "enterprise-8.1.0"):
        rec = record(docs, name)
        assert rec["sources"]["build"]["commit"] == main
        assert rec["sources"]["openapi"]["release"] == spec
        assert (docs / rec["edition"] / "geneva/index.mdx").read_text() == "# Geneva\n"


def test_an_earlier_commit_of_main_can_be_named(docs, sources):
    earlier = git(docs, "rev-parse", "HEAD")
    (docs / "docs/geneva/index.mdx").write_text("# Geneva, later\n")
    commit_all(docs, "a later page")
    later = on_main(docs)

    prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0", "--docs", earlier)
    prepare(docs, sources, "oss", "0.99.0", "--enterprise", "v8.2.0")

    for name, commit, page in (("oss-0.98.0", earlier, "# Geneva\n"), ("oss-0.99.0", later, "# Geneva, later\n")):
        rec = record(docs, name)
        assert rec["sources"]["build"]["commit"] == commit
        assert (docs / rec["edition"] / "geneva/index.mdx").read_text() == page


def test_docs_inputs_off_main_or_unlike_the_running_scripts_are_refused(docs, sources):
    git(docs, "switch", "-q", "-c", "editions")
    (docs / "docs/geneva/index.mdx").write_text("# Geneva, on a branch\n")
    head = commit_all(docs, "an unmerged page")
    oss = ["oss", "0.98.0", "--enterprise", "v8.1.0"]

    main = git(docs, "rev-parse", "origin/main")
    with pytest.raises(edition.EditionError,
                       match=rf"--docs: HEAD is {head}, which lancedb/docs main \({main}\) does not contain"):
        prepare(docs, sources, *oss, "--docs", "HEAD")

    (docs / "scripts").mkdir()
    (docs / "scripts/assemble.py").write_text("# a changed assembler\n")
    commit_all(docs, "an unmerged assembler")
    with pytest.raises(edition.EditionError, match="this checkout's scripts/ are not those of"):
        prepare(docs, sources, *oss)
    (docs / "scripts/assemble.py").write_text("# changed again\n")
    with pytest.raises(edition.EditionError, match="commit or stash the changes to scripts/"):
        prepare(docs, sources, *oss)

    git(docs, "remote", "set-url", "origin", str(docs.parent / "missing/lancedb-docs.git"))
    with pytest.raises(edition.EditionError, match="could not fetch lancedb/docs main; check network access and Git authentication"):
        prepare(docs, sources, *oss)
    git(docs, "remote", "remove", "origin")
    with pytest.raises(edition.EditionError, match="this checkout's origin is not set, not lancedb/docs"):
        prepare(docs, sources, *oss)
    assert state(docs) == {}


def test_only_the_lancedb_docs_urls_are_accepted_as_origin(docs, monkeypatch):
    monkeypatch.setattr(edition, "DOCS_URLS", LANCEDB_DOCS_URLS)
    # Nothing here reaches the network: an accepted URL's fetch stops at once.
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    for url in ("https://github.com/lancedb/docs", "https://github.com/lancedb/docs.git",
                "https://github.com/LanceDB/docs.git", "git@github.com:lancedb/docs.git",
                "ssh://git@github.com/lancedb/docs.git"):
        git(docs, "remote", "set-url", "origin", url)
        with pytest.raises(edition.EditionError, match="could not fetch lancedb/docs main; check network access and Git authentication"):
            edition.docs_main(docs)
    for url in ("https://github.com/someone/docs.git", "git@github.com:someone/docs.git",
                "https://github.com/lancedb/docs-fork.git", "https://github.com/lancedb/docs.git.old",
                "http://github.com/lancedb/docs.git", "https://github.com.example.com/lancedb/docs.git",
                "https://example.com/lancedb/docs.git", "file:///srv/lancedb/docs.git",
                "/srv/git/lancedb-docs.git"):
        git(docs, "remote", "set-url", "origin", url)
        with pytest.raises(edition.EditionError, match="this checkout's origin is not supported, not"):
            edition.docs_main(docs)



@pytest.mark.parametrize("rewritten", [False, True])
def test_rejected_remote_does_not_print_credentials(docs, monkeypatch, rewritten):
    secret = "fixture-only-remote-password"
    remote = f"https://user:{secret}@github.com/lancedb/docs.git"
    monkeypatch.setattr(edition, "DOCS_URLS", LANCEDB_DOCS_URLS)
    if rewritten:
        git(docs, "remote", "set-url", "origin", "https://github.com/lancedb/docs.git")
        git(docs, "config", f"url.{remote}.insteadOf", "https://github.com/lancedb/docs.git")
    else:
        git(docs, "remote", "set-url", "origin", remote)
    with pytest.raises(edition.EditionError, match="origin is not supported") as caught:
        edition.docs_main(docs)
    assert secret not in str(caught.value)
    assert remote not in str(caught.value)


def test_fetch_failure_does_not_print_remote_diagnostics(docs, monkeypatch):
    secret = "fixture-only-fetch-password"
    run = subprocess.run

    def fail_fetch(argv, **kwargs):
        if argv[:4] == ["git", "-C", str(docs), "fetch"]:
            return subprocess.CompletedProcess(argv, 128, b"", (
                f"fatal: unable to access 'https://user:{secret}@example.com/docs.git': denied"
            ).encode())
        return run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", fail_fetch)
    with pytest.raises(edition.EditionError, match="check network access and Git authentication") as caught:
        edition.docs_main(docs)
    assert secret not in str(caught.value)
    assert "example.com" not in str(caught.value)


def test_a_fork_origin_is_refused_before_anything_is_fetched_or_saved(docs, sources):
    """The usual fork layout: origin is a fork, upstream is lancedb/docs, and
    the fork's main has a docs-owned change lancedb/docs does not have."""
    fork = docs.parent / "fork.git"
    git(docs.parent, "clone", "-q", "--bare", git(docs, "remote", "get-url", "origin"), str(fork))
    git(docs, "remote", "rename", "origin", "upstream")
    git(docs, "remote", "add", "origin", str(fork))
    (docs / "docs/geneva/index.mdx").write_text("# Only in the fork\n")
    fork_main = commit_all(docs, "a change only the fork has")
    git(docs, "push", "-q", "origin", "HEAD:refs/heads/main")
    assert not edition.is_ancestor(docs, fork_main, "upstream/main")
    refs = git(docs, "for-each-ref")

    refused = r"this checkout's origin is not supported, not lancedb/docs; a fork's main"
    with pytest.raises(edition.EditionError, match=refused):
        prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    with pytest.raises(edition.EditionError, match=refused):
        prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0", "--docs", "upstream/main")
    git(docs, "remote", "set-url", "origin", git(docs, "remote", "get-url", "upstream"))
    git(docs, "remote", "set-url", "--add", "origin", str(fork))
    with pytest.raises(edition.EditionError, match=refused):
        prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    # The URL is the one fetching would use, after any rewriting.
    git(docs, "config", "--unset-all", "remote.origin.url")
    git(docs, "remote", "set-url", "origin", "https://github.com/lancedb/docs.git")
    git(docs, "config", f"url.{fork}.insteadOf", "https://github.com/lancedb/docs.git")
    with pytest.raises(edition.EditionError, match=refused):
        prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    assert state(docs) == {}
    assert git(docs, "for-each-ref") == refs


def test_a_main_not_fetched_from_lancedb_docs_is_not_trusted(docs, sources):
    main = git(docs, "rev-parse", "origin/main")
    git(docs, "switch", "-q", "-c", "editions")
    (docs / "docs/geneva/index.mdx").write_text("# Geneva, on a branch\n")
    branch = commit_all(docs, "an unmerged page")
    # As if origin/main had been fetched from elsewhere, or set by hand.
    git(docs, "update-ref", "refs/remotes/origin/main", branch)

    with pytest.raises(edition.EditionError, match=rf"which lancedb/docs main \({main}\) does not contain"):
        prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0", "--docs", branch)
    assert state(docs) == {}
    git(docs, "update-ref", "refs/remotes/origin/main", branch)
    prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    assert record(docs, "oss-0.98.0")["sources"]["build"]["commit"] == main
    assert (docs / "editions/oss/0.98.0/geneva/index.mdx").read_text() == "# Geneva\n"
    assert git(docs, "rev-parse", "origin/main") == main


def shadowed_main(docs: Path, kind: str) -> tuple[str, str]:
    """A local branch or tag named origin/main on an unmerged commit, which the
    shorthand origin/main resolves to before the remote-tracking ref."""
    main = git(docs, "rev-parse", "refs/remotes/origin/main")
    git(docs, "switch", "-q", "-c", "editions")
    (docs / "docs/geneva/index.mdx").write_text("# Geneva, unmerged\n")
    unmerged = commit_all(docs, "an unmerged page")
    git(docs, kind, "origin/main", unmerged)
    assert git(docs, "rev-parse", "origin/main") == unmerged
    return main, unmerged


@pytest.mark.parametrize("kind", ["branch", "tag"])
def test_a_local_ref_named_origin_main_is_not_canonical_main(docs, sources, kind):
    main, _unmerged = shadowed_main(docs, kind)

    prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")

    rec = record(docs, "oss-0.98.0")
    assert rec["sources"]["build"]["commit"] == main
    assert (docs / rec["edition"] / "geneva/index.mdx").read_text() == "# Geneva\n"


def test_main_is_fetched_where_it_is_read_when_origin_tracks_another_branch(docs, sources):
    """Git updates no tracking ref from the configured refspec here, so the
    fetch itself must write refs/remotes/origin/main."""
    main = git(docs, "rev-parse", "refs/remotes/origin/main")
    git(docs, "config", "remote.origin.fetch", "+refs/heads/other:refs/remotes/origin/other")
    git(docs, "switch", "-q", "-c", "editions")
    (docs / "docs/geneva/index.mdx").write_text("# Geneva, unmerged\n")
    git(docs, "update-ref", "refs/remotes/origin/main", commit_all(docs, "an unmerged page"))

    prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")

    rec = record(docs, "oss-0.98.0")
    assert rec["sources"]["build"]["commit"] == main
    assert (docs / rec["edition"] / "geneva/index.mdx").read_text() == "# Geneva\n"


@pytest.mark.parametrize("kind", ["branch", "tag"])
def test_naming_an_unmerged_commit_is_refused_despite_a_local_origin_main(docs, sources, kind):
    main, unmerged = shadowed_main(docs, kind)

    for ref in (unmerged, "origin/main"):
        with pytest.raises(edition.EditionError,
                           match=rf"--docs: {ref} is {unmerged}, which lancedb/docs main \({main}\) does not contain"):
            prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0", "--docs", ref)
    assert state(docs) == {}


# protecting saved editions


def test_conflicting_inputs_for_a_saved_release_stop(docs, sources, tmp_path):
    prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    before = state(docs)

    with pytest.raises(edition.EditionError, match="enterprise_pages recorded .*, requested"):
        prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.2.0")

    retagged = tmp_path / "retagged"
    git(tmp_path, "clone", "-q", str(sources["lancedb"]), str(retagged))
    git(retagged, "tag", "-f", "-a", "-m", "moved", "v0.98.0", sources["oss"]["0.99.0"])
    with pytest.raises(edition.EditionError, match="already saved from different inputs .*release recorded"):
        prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0", lancedb=retagged)

    assert state(docs) == before



@pytest.mark.parametrize("scripts_state", ["uncommitted", "branch", "main"])
def test_saved_edition_does_not_need_the_current_assembler(docs, sources, monkeypatch, scripts_state):
    prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    correct(docs, "oss-0.98.0", "guide.mdx", "Corrected guide\n")
    before = state(docs)
    (docs / "scripts").mkdir()
    (docs / "scripts/assemble.py").write_text("# a later assembler\n")
    if scripts_state != "uncommitted":
        commit_all(docs, "change assembler")
        if scripts_state == "main":
            on_main(docs)

    def must_not_fetch(_repo):
        pytest.fail("an existing edition must not fetch docs main")

    monkeypatch.setattr(edition, "docs_main", must_not_fetch)
    out = prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    assert "is corrected (1 correction); nothing changed" in out
    assert state(docs) == before
    (docs / "editions/oss/0.98.0/guide.mdx").write_text("Unrecorded edit\n")
    with pytest.raises(edition.EditionError, match="has changed since its latest recorded correction"):
        prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    with pytest.raises(edition.EditionError, match="enterprise_pages recorded"):
        prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.2.0")


def test_duplicate_preparation_changes_nothing_even_after_a_correction(docs, sources):
    prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    out = prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    assert "editions/oss/0.98.0 is as prepared; nothing changed" in out

    correct(docs, "oss-0.98.0", "guide.mdx", "Corrected guide\n")
    # A commit reference and a tag naming it are the same release input.
    (docs / "docs/geneva/index.mdx").write_text("# Geneva, later\n")
    git(docs, "commit", "-q", "-am", "later")
    on_main(docs)
    before = state(docs)
    out = prepare(docs, sources, "oss", "0.98.0", "--enterprise", sources["enterprise"]["8.1.0"])

    assert "is corrected (1 correction); nothing changed" in out
    assert state(docs) == before
    assert (docs / "editions/oss/0.98.0/guide.mdx").read_text() == "Corrected guide\n"


def test_a_newer_edition_leaves_a_corrected_older_one_alone(docs, sources):
    prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    correct(docs, "oss-0.98.0", "guide.mdx", "Corrected guide\n")
    before = state(docs)

    prepare(docs, sources, "oss", "0.99.0", "--enterprise", "v8.2.0")
    prepare(docs, sources, "enterprise", "8.2.0")

    after = state(docs)
    assert {k: v for k, v in after.items() if k in before} == before
    assert "Shared guide as of 0.99.0." in (docs / "editions/oss/0.99.0/guide.mdx").read_text()
    assert "Shared guide as of 0.99.0." in (docs / "editions/enterprise/8.2.0/guide.mdx").read_text()


def test_a_failed_preparation_leaves_saved_editions_intact(docs, sources, monkeypatch):
    prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    before = state(docs)

    with pytest.raises(edition.EditionError, match="has no docs/web"):
        prepare(docs, sources, "enterprise", "8.5.0")
    assert state(docs) == before

    def fail(*_args):
        raise OSError("disk full")

    # The edition cannot be moved into place, after its folders were made.
    with monkeypatch.context() as patch:
        patch.setattr(edition.os, "rename", fail)
        with pytest.raises(OSError, match="disk full"):
            prepare(docs, sources, "enterprise", "8.1.0")
    assert state(docs) == before

    # The last step fails: the record cannot be put in place after the edition
    # has been moved there.
    with monkeypatch.context() as patch:
        patch.setattr(edition.os, "replace", fail)
        with pytest.raises(OSError, match="disk full"):
            prepare(docs, sources, "enterprise", "8.1.0")
    assert state(docs) == before


def test_an_edition_or_a_record_without_its_pair_is_refused(docs, sources):
    orphan = docs / "editions/oss/0.99.0"
    orphan.mkdir(parents=True)
    (orphan / "index.mdx").write_text("half made\n")
    with pytest.raises(edition.EditionError, match="exists without releases/oss-0.99.0.yaml"):
        prepare(docs, sources, "oss", "0.99.0", "--enterprise", "v8.2.0")

    prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    shutil.rmtree(docs / "editions/oss/0.98.0")
    with pytest.raises(edition.EditionError, match="restore it from Git history"):
        prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    assert not (docs / "editions/oss/0.98.0").exists()


# checking saved editions


def test_check_pairs_records_and_requires_corrections_to_be_listed(docs, sources):
    assert edition.check(docs) == ([], [])

    prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    # Finder files are not part of what Git saves.
    for folder in ("editions", "editions/oss", "editions/oss/0.98.0"):
        (docs / folder / ".DS_Store").write_bytes(b"\0")
    assert edition.check(docs) == (["oss 0.98.0: as prepared"], [])

    (docs / "editions/oss/0.98.0/guide.mdx").write_text("Edited without a record\n")
    lines, problems = edition.check(docs)
    assert lines == [] and "has changed since it was prepared" in problems[0]

    correct(docs, "oss-0.98.0", "guide.mdx", "Corrected guide\n")
    (docs / "editions/enterprise/8.1.0").mkdir(parents=True)
    (docs / "editions/cloud").mkdir()
    lines, problems = edition.check(docs)
    assert lines == ["oss 0.98.0: corrected (1 correction)"]
    assert problems == [
        "editions/cloud is not a product folder",
        "editions/enterprise/8.1.0 has no release record",
    ]


def test_command_prepares_from_main_and_checks_against_the_base(docs, sources):
    """End to end, as a person runs it: the scripts are part of the checkout."""
    (docs / "scripts").mkdir()
    for name in ("assemble.py", "candidate.py", "edition.py"):
        shutil.copy(SCRIPTS / name, docs / "scripts" / name)
    git(docs, "add", "-A")
    git(docs, "commit", "-q", "-m", "scripts")
    main = on_main(docs)

    # Run the checkout's CLI with the local canonical-main stand-in.
    runner = (
        "import re, sys; sys.path.insert(0, 'scripts'); import edition; "
        "edition.DOCS_URLS = re.compile(sys.argv[1], re.IGNORECASE); raise SystemExit(edition.main(sys.argv[2:]))"
    )

    def run(*argv: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-B", "-c", runner, TEST_DOCS_URLS.pattern, *argv],
            cwd=docs, env=environment(), capture_output=True, text=True,
        )

    repos = ["--lancedb", str(sources["lancedb"]), "--enterprise-repo", str(sources["enterprise_repo"])]
    result = run("prepare", "oss", "0.98.0", "--enterprise", "v8.1.0", *repos)
    assert result.returncode == 0, result.stderr
    assert record(docs, "oss-0.98.0")["sources"]["build"]["commit"] == main

    with (docs / "scripts/assemble.py").open("a") as script:
        script.write("# a local change\n")
    result = run("prepare", "oss", "0.99.0", "--enterprise", "v8.2.0", *repos)
    assert result.returncode == 1
    assert "commit or stash the changes to scripts/" in result.stderr
    assert not (docs / "editions/oss/0.99.0").exists()
    git(docs, "checkout", "--", "scripts/assemble.py")

    result = run("check")
    assert (result.returncode, result.stdout) == (0, "oss 0.98.0: as prepared\n")
    base = commit_all(docs, "oss 0.98.0")
    assert run("check", "--base", base).returncode == 0

    (docs / "editions/oss/0.98.0/index.mdx").write_text("A correction\n")
    result = run("check")
    assert (result.returncode, result.stdout) == (1, "")
    reported = re.fullmatch(
        r"edition: editions/oss/0\.98\.0 has changed since it was prepared; its files now have "
        r"checksum ([0-9a-f]{64})\. For a reviewed correction, add an entry under `corrections` in "
        r"releases/oss-0\.98\.0\.yaml with `sha256: \1`\n",
        result.stderr,
    )[1]

    # The prepared checksum rewritten to match: only the base shows the change.
    rewrite(docs, "oss-0.98.0", change("prepared.sha256", reported))
    assert run("check").returncode == 0
    result = run("check", "--base", base)
    assert (result.returncode, result.stdout) == (1, "oss 0.98.0: as prepared\n")
    assert result.stderr == f"edition: {kept(base, 'releases/oss-0.98.0.yaml', 'prepared.sha256 changed')}\n"

    result = run("check", "--base", "1" * 40)
    assert (result.returncode, result.stdout) == (1, "")
    assert result.stderr.startswith(f"edition: --base: {'1' * 40} is not a commit in ")


# corrections are bound to the edition's checksum


def test_a_later_unrecorded_change_fails_after_a_recorded_correction(docs, sources):
    """A recorded correction to guide.mdx, then an unrelated page deleted
    without updating the record."""
    prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    folder = docs / "editions/oss/0.98.0"
    (folder / "guide.mdx").write_text("Reviewed correction\n")
    add_correction(docs, "oss-0.98.0", "guide.mdx", checksum(saved(folder)))
    assert edition.check(docs) == (["oss 0.98.0: corrected (1 correction)"], [])
    (folder / "enterprise/security.mdx").unlink()

    lines, problems = edition.check(docs)

    assert lines == []
    assert len(problems) == 1
    assert problems[0].startswith(
        "editions/oss/0.98.0 has changed since its latest recorded correction; its files now "
        "have checksum "
    )


def test_each_correction_records_the_snapshot_it_accepts(docs, sources):
    prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    prepared = record(docs, "oss-0.98.0")["prepared"]
    first = correct(docs, "oss-0.98.0", "guide.mdx", "Corrected guide\n")
    assert edition.check(docs) == (["oss 0.98.0: corrected (1 correction)"], [])

    page = docs / "editions/oss/0.98.0/index.mdx"
    page.write_text("Unreviewed edit\n")
    assert "has changed since its latest recorded correction" in edition.check(docs)[1][0]
    page.unlink()
    assert "has changed since its latest recorded correction" in edition.check(docs)[1][0]

    second = record_correction(docs, "oss-0.98.0", "index.mdx")
    assert edition.check(docs) == (["oss 0.98.0: corrected (2 corrections)"], [])
    rec = record(docs, "oss-0.98.0")
    assert rec["prepared"] == prepared
    assert [c["sha256"] for c in rec["corrections"]] == [first, second]
    assert second == checksum(saved(docs / "editions/oss/0.98.0"))

    before = state(docs)
    out = prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    assert out.endswith("editions/oss/0.98.0 is corrected (2 corrections); nothing changed")
    assert state(docs) == before

    (docs / "editions/oss/0.98.0/guide.mdx").write_text("Another unreviewed edit\n")
    before = state(docs)
    with pytest.raises(edition.EditionError, match="already saved; nothing changed, but .* has changed"):
        prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    assert state(docs) == before


# merged editions and records are kept


@pytest.fixture
def base(docs, sources) -> str:
    """Main after OSS 0.98.0 and Enterprise 8.1.0 were merged, and then a
    correction to OSS 0.98.0: the base a pull request is checked against."""
    prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    prepare(docs, sources, "enterprise", "8.1.0")
    commit_all(docs, "editions")
    correct(docs, "oss-0.98.0", "guide.mdx", "Corrected guide\n")
    return commit_all(docs, "a correction")


KEPT = ["enterprise 8.1.0: as prepared", "oss 0.98.0: corrected (1 correction)"]


def kept(base: str, rel: str, *changes: str) -> str:
    """The problem `check --base` reports for a record changed since `base`."""
    return (
        f"{rel} differs from the base {base}: {', '.join(changes)}; a saved release's "
        "record only gains corrections, at the end"
    )


def test_unchanged_editions_and_a_new_one_pass_against_the_base(docs, sources, base):
    assert edition.check(docs, base=base) == (KEPT, [])

    # A new edition, and a record written again with other formatting and no
    # header comment.
    prepare(docs, sources, "oss", "0.99.0", "--enterprise", "v8.2.0")
    rewrite(docs, "enterprise-8.1.0", lambda rec: None)
    assert edition.check(docs, base=base) == ([*KEPT, "oss 0.99.0: as prepared"], [])

    (docs / "editions/oss/0.99.0/index.mdx").write_text("Edited without a record\n")
    lines, problems = edition.check(docs, base=base)
    assert lines == KEPT
    assert len(problems) == 1
    assert problems[0].startswith("editions/oss/0.99.0 has changed since it was prepared")


def test_prepared_rewritten_with_the_snapshot_fails_against_the_base(docs, base):
    folder = docs / "editions/enterprise/8.1.0"
    (folder / "index.mdx").write_text("Edited\n")
    (folder / "guide.mdx").unlink()
    files = saved(folder)
    rewrite(docs, "enterprise-8.1.0", change("prepared", {"files": len(files), "sha256": checksum(files)}))

    # The edition matches its record, so on its own the change passes.
    assert edition.check(docs) == (KEPT, [])
    assert edition.check(docs, base=base) == (KEPT, [
        kept(base, "releases/enterprise-8.1.0.yaml", "prepared.files changed", "prepared.sha256 changed"),
    ])


def test_editing_or_removing_a_recorded_correction_fails_against_the_base(docs, base):
    folder = docs / "editions/oss/0.98.0"
    # A page deleted, and the existing correction's checksum rewritten to match.
    (folder / "enterprise/security.mdx").unlink()

    def rewritten(rec):
        rec["corrections"][0]["sha256"] = checksum(saved(folder))

    rewrite(docs, "oss-0.98.0", rewritten)
    assert edition.check(docs) == (KEPT, [])
    assert edition.check(docs, base=base) == (KEPT, [
        kept(base, "releases/oss-0.98.0.yaml", "corrections[0].sha256 changed"),
    ])

    # The correction undone and its entry removed: the edition is as prepared.
    git(docs, "reset", "-q", "--hard", base)
    git(docs, "checkout", "HEAD~1", "--", "editions/oss/0.98.0/guide.mdx")
    rewrite(docs, "oss-0.98.0", change("corrections", []))
    lines = [KEPT[0], "oss 0.98.0: as prepared"]
    assert edition.check(docs) == (lines, [])
    assert edition.check(docs, base=base) == (lines, [
        kept(base, "releases/oss-0.98.0.yaml", "corrections[0] was removed"),
    ])


def test_removing_or_replacing_a_saved_release_fails_against_the_base(docs, sources, base):
    shutil.rmtree(docs / "editions/enterprise/8.1.0")
    (docs / "releases/enterprise-8.1.0.yaml").unlink()
    assert edition.check(docs) == ([KEPT[1]], [])
    assert edition.check(docs, base=base) == ([KEPT[1]], [
        f"releases/enterprise-8.1.0.yaml is on the base {base} but has been removed; a saved "
        "release keeps its record",
        f"editions/enterprise/8.1.0 is on the base {base} but has been removed; a saved edition "
        "is never removed",
    ])

    git(docs, "reset", "-q", "--hard", base)
    shutil.rmtree(docs / "editions/enterprise/8.1.0")
    assert edition.check(docs, base=base) == ([KEPT[1]], [
        "releases/enterprise-8.1.0.yaml: editions/enterprise/8.1.0 is missing",
    ])

    # Removed with its record, then prepared again from other inputs.
    git(docs, "reset", "-q", "--hard", base)
    shutil.rmtree(docs / "editions/oss/0.98.0")
    (docs / "releases/oss-0.98.0.yaml").unlink()
    prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.2.0")
    assert edition.check(docs) == ([KEPT[0], "oss 0.98.0: as prepared"], [])
    assert edition.check(docs, base=base)[1] == [kept(
        base, "releases/oss-0.98.0.yaml", "prepared.sha256 changed", "release.enterprise_pages.commit changed",
        "release.enterprise_pages.ref changed", "sources.enterprise.commit changed", "corrections[0] was removed",
    )]


def test_an_appended_correction_passes_against_the_base(docs, base):
    rec = record(docs, "oss-0.98.0")
    correct(docs, "oss-0.98.0", "index.mdx", "Corrected index\n")

    assert edition.check(docs, base=base) == ([KEPT[0], "oss 0.98.0: corrected (2 corrections)"], [])
    assert record(docs, "oss-0.98.0")["corrections"][0] == rec["corrections"][0]


# release records keep their provenance


def rewrite(docs: Path, name: str, change) -> None:
    path = docs / "releases" / f"{name}.yaml"
    rec = yaml.safe_load(path.read_text())
    change(rec)
    path.write_text(yaml.safe_dump(rec))


def test_records_without_their_pinned_sources_fail(docs, sources):
    """An OSS record without its docs source and with a malformed commit, and
    an Enterprise record without its qualification and bundled LanceDB."""
    prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    prepare(docs, sources, "enterprise", "8.1.0")

    def oss(rec):
        del rec["sources"]["build"]
        rec["sources"]["lancedb"]["commit"] = "not-a-commit"

    def enterprise(rec):
        del rec["release"]["qualification"]
        del rec["release"]["bundled_lancedb"]

    rewrite(docs, "oss-0.98.0", oss)
    rewrite(docs, "enterprise-8.1.0", enterprise)

    assert edition.check(docs) == ([], [
        "releases/enterprise-8.1.0.yaml: release.qualification is missing; "
        "release.bundled_lancedb is missing",
        "releases/oss-0.98.0.yaml: sources.lancedb.commit is 'not-a-commit', not a commit "
        "hash; sources.build is missing",
    ])


@pytest.fixture(scope="module")
def records(tmp_path_factory, sources) -> dict[str, dict]:
    """The records the builder writes, which are the contract."""
    docs = docs_checkout(tmp_path_factory.mktemp("records").resolve() / "docs")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(edition, "DOCS_URLS", TEST_DOCS_URLS)
        prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
        prepare(docs, sources, "enterprise", "8.1.0")
    return {name: record(docs, name) for name in ("oss-0.98.0", "enterprise-8.1.0")}


def change(path: str, value=KeyError):
    def apply(rec):
        *parents, key = path.split(".")
        for parent in parents:
            rec = rec[parent]
        if value is KeyError:
            del rec[key]
        else:
            rec[key] = value
    return apply


A, B, C = "a" * 64, "b" * 64, "2" * 40

# One change each, and the one problem it must cause. {release}, {pages},
# {bundled} and {prepared} stand for the record's own values.
MALFORMED = [
    ("oss-0.98.0", change("format", 2), "format is 2, not 1"),
    ("oss-0.98.0", change("product", "cloud"), "product is 'cloud', not oss or enterprise"),
    ("oss-0.98.0", change("version", "0.98"), "version is '0.98', not a version such as 1.2.3"),
    ("oss-0.98.0", change("edition", "editions/oss/0.99.0"),
     "edition is 'editions/oss/0.99.0', not editions/oss/0.98.0"),
    ("oss-0.98.0", change("notes", "x"), "notes is not a field here"),
    ("oss-0.98.0", change("release.repository", "lancedb/sophon"),
     "release.repository is 'lancedb/sophon', not lancedb/lancedb"),
    ("oss-0.98.0", change("release.tag", "v0.99.0"), "release.tag is 'v0.99.0', not v0.98.0"),
    ("oss-0.98.0", change("release.commit", "abc123"), "release.commit is 'abc123', not a commit hash"),
    ("oss-0.98.0", change("release.commit", C),
     "sources.lancedb.commit is {release}, but release.commit is " + C),
    ("oss-0.98.0", change("release.enterprise_pages"), "release.enterprise_pages is missing"),
    ("oss-0.98.0", change("release.enterprise_pages.ref", ""),
     "release.enterprise_pages.ref is '', not the tag or commit named with --enterprise"),
    ("oss-0.98.0", change("release.enterprise_pages.commit", C),
     "sources.enterprise.commit is {pages}, but release.enterprise_pages.commit is " + C),
    ("oss-0.98.0", change("release.bundled_lancedb", C), "release.bundled_lancedb is not a field here"),
    ("oss-0.98.0", change("sources.build"), "sources.build is missing"),
    ("oss-0.98.0", change("sources.build.commit", "HEAD"), "sources.build.commit is 'HEAD', not a commit hash"),
    ("oss-0.98.0", change("sources.enterprise.repository", "lancedb/lancedb"),
     "sources.enterprise.repository is 'lancedb/lancedb', not lancedb/sophon"),
    ("oss-0.98.0", change("sources.lancedb.path", "docs"), "sources.lancedb.path is 'docs', not docs/web"),
    ("oss-0.98.0", change("sources.geneva", {}), "sources.geneva is not a field here"),
    ("oss-0.98.0", change("sources.openapi.sha256", "x"),
     "sources.openapi.sha256 is 'x', not a SHA-256 checksum"),
    ("oss-0.98.0", change("prepared"), "prepared is missing"),
    ("oss-0.98.0", change("prepared.files", "9"), "prepared.files is '9', not a number of files"),
    ("oss-0.98.0", change("prepared.sha256", "0" * 63),
     f"prepared.sha256 is {'0' * 63!r}, not a SHA-256 checksum"),
    ("oss-0.98.0", change("corrections", {}), "corrections is {{}}, not a list"),
    ("oss-0.98.0", change("corrections", [{"summary": "Fix.", "files": ["guide.mdx"]}]),
     "corrections[0].sha256 is missing, not the SHA-256 checksum of the corrected edition that "
     "`check` reports"),
    ("oss-0.98.0", change("corrections", [{"summary": "Fix.", "sha256": "{prepared}"}]),
     "corrections[0].sha256 is the checksum before it; a correction changes the edition"),
    ("oss-0.98.0", change("corrections", [{"summary": "Fix.", "sha256": A}, {"summary": "Again.", "sha256": A}]),
     "corrections[1].sha256 is the checksum before it; a correction changes the edition"),
    ("oss-0.98.0", change("corrections", [{"sha256": A}]), "corrections[0].summary must be one line of text"),
    ("oss-0.98.0", change("corrections", [{"summary": "One\nTwo", "sha256": A}]),
     "corrections[0].summary must be one line of text"),
    ("oss-0.98.0", change("corrections", [{"summary": "Fix.", "files": "guide.mdx", "sha256": A}]),
     "corrections[0].files must be a list of the edition's paths"),
    ("oss-0.98.0", change("corrections", [{"summary": "Fix.", "sha256": A, "approved": True}]),
     "corrections[0].approved is not a field here"),
    ("enterprise-8.1.0", change("release.qualification"), "release.qualification is missing"),
    ("enterprise-8.1.0", change("release.qualification.tag", "release-qualified/v8.2.0"),
     "release.qualification.tag is 'release-qualified/v8.2.0', not release-qualified/v8.1.0"),
    ("enterprise-8.1.0", change("release.qualification.object", "annotated"),
     "release.qualification.object is 'annotated', not a tag object hash"),
    ("enterprise-8.1.0", change("release.bundled_lancedb"), "release.bundled_lancedb is missing"),
    ("enterprise-8.1.0", change("release.bundled_lancedb", C),
     "sources.lancedb.commit is {bundled}, but release.bundled_lancedb is " + C),
    ("enterprise-8.1.0", change("release.commit", C),
     "sources.enterprise.commit is {release}, but release.commit is " + C),
    ("enterprise-8.1.0", change("release.repository", "lancedb/lancedb"),
     "release.repository is 'lancedb/lancedb', not lancedb/sophon"),
    ("enterprise-8.1.0", change("release.enterprise_pages", {"ref": "v8.1.0", "commit": C}),
     "release.enterprise_pages is not a field here"),
]


@pytest.mark.parametrize(
    "name, mutate, problem", MALFORMED,
    ids=[f"{n}:{p.replace('{{}}', '{}')[:60]}" for n, _m, p in MALFORMED],
)
def test_a_malformed_record_names_the_field(records, tmp_path, name, mutate, problem):
    rec = copy.deepcopy(records[name])
    values = {
        "release": rec["release"]["commit"],
        "pages": (rec["release"].get("enterprise_pages") or {}).get("commit"),
        "bundled": rec["release"].get("bundled_lancedb"),
        "prepared": rec["prepared"]["sha256"],
    }
    mutate(rec)
    for entry in rec.get("corrections") if isinstance(rec.get("corrections"), list) else []:
        if entry.get("sha256") == "{prepared}":
            entry["sha256"] = values["prepared"]
    path = tmp_path / "releases" / f"{name}.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(rec))

    with pytest.raises(edition.EditionError) as raised:
        edition.load_record(path)
    assert str(raised.value) == f"releases/{name}.yaml: " + problem.format(**values)


def test_records_the_builder_writes_are_valid(records, tmp_path):
    """Including an Enterprise record whose bundled LanceDB no release tags,
    and a record with two corrections."""
    corrected = copy.deepcopy(records["oss-0.98.0"])
    corrected["corrections"] = [
        {"summary": "Fix the guide.", "files": ["guide.mdx"], "sha256": A},
        {"summary": "Remove a page.", "sha256": B},
    ]
    for name, rec in [*records.items(), ("oss-0.98.0", corrected)]:
        assert edition.record_problems(rec, f"{name}.yaml") == []
    assert edition.record_problems(records["enterprise-8.1.0"], "enterprise-8.2.0.yaml") == [
        "product and version name enterprise-8.1.0.yaml, not enterprise-8.2.0.yaml"
    ]
