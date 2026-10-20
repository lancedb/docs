"""Tests for preparing saved editions from release records.

Disposable repositories stand in for lancedb/lancedb, lancedb/sophon and this
repository. Their versions are fixtures, not releases: OSS 0.95.0 to 0.99.0 and
Enterprise 8.1.0 to 8.9.0, each built to exercise one case. The docs fixture
uses this repository's own assemble.yaml and .gitignore, and every edition is
built by the real assembler.

Run with `make test-assemble`.
"""

import hashlib
import json
import os
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
    """lancedb's docs/web as of `label`: navigation, a page with an example and a
    draft section, a draft page, an asset, and the Enterprise placeholder the
    current assembler needs."""
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
    """lancedb and sophon repositories with tagged fixture releases. Shared, so
    tests only read them."""
    base = tmp_path_factory.mktemp("sources").resolve()
    lancedb = new_repo(base / "lancedb")
    c = {}
    c["no-docs"] = snapshot(lancedb, {"README.md": b"Before docs/web\n"}, "no docs yet")
    git(lancedb, "tag", "v0.97.0", c["no-docs"])
    c["bundled"] = snapshot(lancedb, oss_pages("the bundled revision"), "bundled by 8.1")
    c["ignored"] = snapshot(lancedb, {**oss_pages("0.95.0"), "docs/web/scratch/notes.mdx": b"x\n"}, "0.95")
    git(lancedb, "tag", "v0.95.0", c["ignored"])
    c["0.98.0"] = snapshot(lancedb, oss_pages("0.98.0"), "0.98")
    annotated(lancedb, "v0.98.0", c["0.98.0"])
    c["0.99.0"] = snapshot(lancedb, oss_pages("0.99.0"), "0.99")
    annotated(lancedb, "v0.99.0", c["0.99.0"])
    c["main"] = snapshot(lancedb, oss_pages("unreleased main"), "main moves on")
    # A working tree that differs from every commit.
    (lancedb / "docs/web").mkdir(parents=True)
    (lancedb / "docs/web/guide.mdx").write_text("Uncommitted text\n")

    sophon = new_repo(base / "sophon")
    e = {}

    def release(version: str, files: dict, qualified: str | None = "self") -> str:
        commit = snapshot(sophon, files, version)
        annotated(sophon, f"v{version}", commit)
        if qualified:
            annotated(sophon, f"release-qualified/v{version}", commit if qualified == "self" else qualified)
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
    git(sophon, "tag", "v8.1.0-lightweight", e["8.1.0"])
    e["main"] = snapshot(sophon, enterprise_pages("unreleased main", c["main"]), "main moves on")
    return {"lancedb": lancedb, "sophon": sophon, "oss": c, "enterprise": e}


@pytest.fixture
def docs(tmp_path) -> Path:
    """A lancedb/docs checkout: this repository's assemble.yaml and .gitignore,
    and docs-owned pages, a navigation fragment and the OpenAPI spec."""
    repo = new_repo(tmp_path.resolve() / "docs")
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
    return repo


def prepare(docs: Path, sources: dict, *argv: str, lancedb: Path | None = None) -> str:
    args = edition.parse([
        "prepare", *argv,
        "--lancedb", str(lancedb or sources["lancedb"]), "--sophon", str(sources["sophon"]),
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


def correct(docs: Path, name: str, page: str, text: str) -> None:
    """A reviewed correction: edit the saved page and say so in the record."""
    product, version = name.split("-")
    (docs / "editions" / product / version / page).write_text(text)
    path = docs / "releases" / f"{name}.yaml"
    path.write_text(path.read_text().replace(
        "corrections: []", f"corrections:\n- summary: Fix {page}.\n  files: [{page}]"
    ))


# --------------------------------------------------------------------------- #
# preparing editions
# --------------------------------------------------------------------------- #


def test_oss_edition_is_built_from_its_tag_and_named_enterprise_pages(docs, sources):
    out = prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")

    files = saved(docs / "editions/oss/0.98.0")
    assert sorted(files) == [
        ".mintignore", "api-reference/rest/openapi.yml", "docs.json", "draft.mdx",
        "enterprise/security.mdx", "geneva/index.mdx", "guide.mdx", "index.mdx",
        "static/assets/marker.svg",
    ]
    guide = files["guide.mdx"].decode()
    # The tag's own page, with its example written in, its draft section and
    # draft page kept for Mint, and its asset byte for byte.
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
        "build": {"repository": "lancedb/docs", "path": "docs", "commit": git(docs, "rev-parse", "HEAD")},
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
    # The shared pages are the bundled revision's, which no OSS release tags,
    # and no OSS edition is needed for them.
    assert "Shared guide as of the bundled revision." in files["guide.mdx"].decode()
    assert files["enterprise/security.mdx"] == b"Enterprise security for 8.1\n"
    assert not (docs / "editions/oss").exists()
    assert not any(INTERNAL in data for data in files.values())

    rec = record(docs, "enterprise-8.1.0")
    sophon = sources["sophon"]
    assert rec["release"] == {
        "repository": "lancedb/sophon",
        "tag": "v8.1.0",
        "commit": sources["enterprise"]["8.1.0"],
        "qualification": {
            "tag": "release-qualified/v8.1.0",
            "object": git(sophon, "rev-parse", "refs/tags/release-qualified/v8.1.0"),
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
        (["oss", "0.98.0"], "name the lancedb/sophon tag or commit"),
        (["oss", "0.98.0", "--enterprise", "main"], "not a branch"),
        (["oss", "0.98.0", "--enterprise", "v8.1.0", "--bundled-lancedb", "1" * 40], "Enterprise releases only"),
        (["oss", "0.95.0", "--enterprise", "v8.1.0"], ".gitignore would leave out"),
        (["enterprise", "8.1.0", "--enterprise", "v8.2.0"], "applies to OSS releases"),
        (["enterprise", "8.3.0"], "is not qualified"),
        (["enterprise", "8.4.0"], "but release-qualified/v8.4.0 qualifies"),
        (["enterprise", "8.5.0"], "has no docs/web"),
        (["enterprise", "8.6.0"], "records no bundled LanceDB"),
        (["enterprise", "8.1.0", "--bundled-lancedb", "1" * 40], "disagrees with lancedb/sophon v8.1.0"),
        (["enterprise", "8.9.0"], "fetch it first"),
        (["enterprise", "8.7.0"], r"\.notes\.mdx is hidden"),
        (["enterprise", "8.8.0"], "ci.md is a symlink"),
    ],
)
def test_preparation_stops_and_saves_nothing(docs, sources, argv, error):
    with pytest.raises(edition.EditionError, match=error):
        prepare(docs, sources, *argv)

    assert state(docs) == {}


# --------------------------------------------------------------------------- #
# protecting saved editions
# --------------------------------------------------------------------------- #


def test_conflicting_inputs_for_a_saved_release_stop(docs, sources, tmp_path):
    prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    before = state(docs)

    with pytest.raises(edition.EditionError, match="enterprise_pages recorded .*, requested"):
        prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.2.0")

    # The release tag moved after the edition was saved.
    retagged = tmp_path / "retagged"
    git(tmp_path, "clone", "-q", str(sources["lancedb"]), str(retagged))
    git(retagged, "tag", "-f", "-a", "-m", "moved", "v0.98.0", sources["oss"]["0.99.0"])
    with pytest.raises(edition.EditionError, match="already saved from different inputs .*release recorded"):
        prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0", lancedb=retagged)

    assert state(docs) == before


def test_duplicate_preparation_changes_nothing_even_after_a_correction(docs, sources):
    prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    out = prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    assert "editions/oss/0.98.0 is as prepared; nothing changed" in out

    correct(docs, "oss-0.98.0", "guide.mdx", "Corrected guide\n")
    # This repository moves on, and the same release is prepared again, its
    # Enterprise pages named by commit this time.
    (docs / "docs/geneva/index.mdx").write_text("# Geneva, later\n")
    git(docs, "commit", "-q", "-am", "later")
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


# --------------------------------------------------------------------------- #
# checking saved editions
# --------------------------------------------------------------------------- #


def test_check_pairs_records_and_requires_corrections_to_be_listed(docs, sources):
    assert edition.check(docs) == ([], [])

    prepare(docs, sources, "oss", "0.98.0", "--enterprise", "v8.1.0")
    # Finder files are not part of what Git saves.
    for folder in ("editions", "editions/oss", "editions/oss/0.98.0"):
        (docs / folder / ".DS_Store").write_bytes(b"\0")
    assert edition.check(docs) == (["oss 0.98.0: as prepared"], [])

    (docs / "editions/oss/0.98.0/guide.mdx").write_text("Edited without a record\n")
    lines, problems = edition.check(docs)
    assert lines == [] and "lists no correction" in problems[0]

    correct(docs, "oss-0.98.0", "guide.mdx", "Corrected guide\n")
    (docs / "editions/enterprise/8.1.0").mkdir(parents=True)
    lines, problems = edition.check(docs)
    assert lines == ["oss 0.98.0: corrected (1 correction)"]
    assert problems == ["editions/enterprise/8.1.0 has no release record"]


def test_command_records_only_committed_build_inputs(docs, sources):
    """End to end, as a person runs it: the scripts are part of the checkout."""
    (docs / "scripts").mkdir()
    for name in ("assemble.py", "candidate.py", "edition.py"):
        shutil.copy(SCRIPTS / name, docs / "scripts" / name)
    git(docs, "add", "-A")
    git(docs, "commit", "-q", "-m", "scripts")

    def run(*argv: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-B", "scripts/edition.py", *argv],
            cwd=docs, env=environment(), capture_output=True, text=True,
        )

    repos = ["--lancedb", str(sources["lancedb"]), "--sophon", str(sources["sophon"])]
    result = run("prepare", "oss", "0.98.0", "--enterprise", "v8.1.0", *repos)
    assert result.returncode == 0, result.stderr
    assert record(docs, "oss-0.98.0")["sources"]["build"]["commit"] == git(docs, "rev-parse", "HEAD")

    with (docs / "scripts/assemble.py").open("a") as script:
        script.write("# a local change\n")
    result = run("prepare", "oss", "0.99.0", "--enterprise", "v8.2.0", *repos)
    assert result.returncode == 1
    assert "commit or stash the changes" in result.stderr
    assert not (docs / "editions/oss/0.99.0").exists()

    result = run("check")
    assert (result.returncode, result.stdout) == (0, "oss 0.98.0: as prepared\n")
