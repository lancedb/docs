"""Tests for composing saved editions into one versioned site.

Each test saves small editions, laid out as the assembler builds them and with
records that `edition.py check` accepts, in a disposable docs repository, then
composes them. Their versions are fixtures, not releases: OSS 0.98.0 and
0.99.0, Enterprise 8.1.0 to 8.10.0. The records' commits are made up, since
composing reads only the saved editions and their records.

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
from pathlib import Path

import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import candidate  # noqa: E402
import compose  # noqa: E402
import edition  # noqa: E402

# The MDX compiler Mint uses: `<mint>/node_modules/@mdx-js/mdx/index.js`.
MDX_COMPILER = os.environ.get("MDX_COMPILER")
CONFIG = {
    "host": "docs.lancedb.com",
    "products": [
        {"product": "enterprise", "label": "Enterprise", "name": "LanceDB Enterprise", "icon": "server"},
        {"product": "oss", "label": "OSS", "name": "LanceDB OSS", "icon": "database"},
    ],
}
NAVIGATION = {
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
REDIRECTS = [
    {"source": "/start", "destination": "guide"},
    {"source": "/old/:slug*", "destination": "/guide#setup"},
]


def environment() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1", GIT_TERMINAL_PROMPT="0")
    return env


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    """Composing runs `git check-ignore`; it sees no user configuration."""
    for key in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(key)
    for key, value in environment().items():
        if key.startswith("GIT_"):
            monkeypatch.setenv(key, value)


@pytest.fixture
def docs(tmp_path) -> Path:
    """A docs checkout with this repository's .gitignore."""
    repo = tmp_path / "docs"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, env=environment(), check=True)
    shutil.copy(SCRIPTS.parent / ".gitignore", repo / ".gitignore")
    configure(repo)
    shared = repo / "site"
    (shared / "static").mkdir(parents=True)
    (shared / "settings.json").write_text(json.dumps({
        "name": "LanceDB", "favicon": "/static/favicon.svg",
        "banner": {"content": "Shared announcement"},
        "integrations": {"ga4": {"measurementId": "G-SHARED"}},
    }))
    (shared / "static/favicon.svg").write_text("<svg>shared</svg>\n")
    (shared / "static/style.css").write_text("/* shared style */\n")
    return repo


def configure(repo: Path, enterprise: str | None = None, oss: str | None = None) -> None:
    config = copy.deepcopy(CONFIG)
    config["products"][0]["preferred"] = enterprise
    config["products"][1]["preferred"] = oss
    (repo / "compose.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


def edition_files(label: str, redirects: list | None = None, extra: dict | None = None) -> dict:
    """A small edition as the assembler saves one."""
    docs_json = {
        "name": "LanceDB",
        "favicon": "/static/favicon.svg",
        "navigation": NAVIGATION,
        "redirects": REDIRECTS if redirects is None else redirects,
    }
    return {
        "docs.json": json.dumps(docs_json, indent=2) + "\n",
        ".mintignore": "draft.mdx\n",
        "index.mdx": f'---\ntitle: "Home"\n---\n\nLanceDB {label}. Read the [guide](/guide#setup).\n',
        "guide.mdx": f'---\ntitle: "Guide"\n---\n\n## Setup {{#setup}}\n\nThe guide as of {label}.\n',
        "enterprise/security.mdx": f'---\ntitle: "Security"\n---\n\nSecurity as of {label}.\n',
        "draft.mdx": '---\ntitle: "Draft"\n---\n\nNot released yet; see [the plan](/plans/next).\n',
        "static/favicon.svg": "<svg/>\n",
        "static/logo.svg": f"<svg>{label}</svg>\n",
        "static/style.css": f"/* {label} */\n",
        **(extra or {}),
    }


def made_up(*parts: str) -> str:
    return hashlib.sha1(" ".join(parts).encode()).hexdigest()


def save(repo: Path, product: str, version: str, files: dict | None = None) -> None:
    """Save an edition and the record `prepare` would write for it."""
    rel = f"editions/{product}/{version}"
    for path, content in (files or edition_files(f"{product} {version}")).items():
        target = repo / rel / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    hashes = edition.saved_hashes(repo, rel)
    lancedb, enterprise = made_up("lancedb", product, version), made_up("enterprise", product, version)
    if product == "enterprise":
        release = {"repository": "lancedb/sophon", "tag": f"v{version}", "commit": enterprise,
                   "qualification": {"tag": f"release-qualified/v{version}", "object": made_up("tag", version)},
                   "bundled_lancedb": lancedb}
    else:
        release = {"repository": "lancedb/lancedb", "tag": f"v{version}", "commit": lancedb,
                   "enterprise_pages": {"ref": "v8.2.0", "commit": enterprise}}
    record = {
        "format": edition.FORMAT, "product": product, "version": version, "edition": rel,
        "release": release,
        "sources": {
            "lancedb": edition.source("lancedb", lancedb),
            "enterprise": edition.source("enterprise", enterprise),
            "build": edition.source("build", made_up("docs")),
        },
        "prepared": {"files": len(hashes), "sha256": candidate.site_checksum(hashes)},
        "corrections": [],
    }
    path = edition.record_path(repo, product, version)
    path.parent.mkdir(exist_ok=True)
    path.write_text(edition.HEADER + yaml.safe_dump(record, sort_keys=False))


def correct(repo: Path, product: str, version: str, page: str, text: str, summary: str) -> None:
    """A reviewed correction: the edited page, recorded with the edition's new checksum."""
    rel = f"editions/{product}/{version}"
    (repo / rel / page).write_text(text)
    path = edition.record_path(repo, product, version)
    record = yaml.safe_load(path.read_text())
    checksum = candidate.site_checksum(edition.saved_hashes(repo, rel))
    record["corrections"].append({"summary": summary, "files": [page], "sha256": checksum})
    path.write_text(edition.HEADER + yaml.safe_dump(record, sort_keys=False))


def composed(repo: Path, tmp_path: Path, name: str = "site") -> Path:
    compose.compose(repo, tmp_path / name)
    return tmp_path / name


def site_json(site: Path) -> dict:
    return json.loads((site / "docs.json").read_text())


def tree(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def versions(site: Path) -> dict[str, dict]:
    """Each version of the site by its path, as the selector lists it."""
    return {
        version["tabs"][0]["groups"][0]["pages"][0].rsplit("/", 1)[0]: version
        for product in site_json(site)["navigation"]["products"]
        for version in product["versions"]
    }


def unversioned(site: Path) -> list[dict]:
    return [r for r in site_json(site)["redirects"] if not compose.VERSIONED_RE.match(r["source"])]


def standard(docs: Path) -> None:
    """Enterprise 8.1.0 and 8.2.0 and OSS 0.99.0, with 8.2.0 and 0.99.0 preferred."""
    for product, version in (("enterprise", "8.1.0"), ("enterprise", "8.2.0"), ("oss", "0.99.0")):
        save(docs, product, version)
    configure(docs, enterprise="8.2.0", oss="0.99.0")


# selectors and layout


def test_the_selectors_list_enterprise_first_and_each_products_versions_newest_first(docs, tmp_path):
    for version in ("8.9.0", "8.10.0", "8.2.0"):
        save(docs, "enterprise", version)
    save(docs, "oss", "0.99.0")
    configure(docs, enterprise="8.9.0", oss="0.99.0")
    products = site_json(composed(docs, tmp_path))["navigation"]["products"]

    assert [(p["product"], p["icon"], [v["version"] for v in p["versions"]]) for p in products] == [
        ("Enterprise", "server", ["8.10.0", "8.9.0", "8.2.0"]),
        ("OSS", "database", ["0.99.0"]),
    ]
    # The newest edition is not the default unless compose.yaml prefers it.
    assert [(v.get("default"), v.get("tag")) for v in products[0]["versions"]] == [
        (None, None), (True, "Latest"), (None, None)
    ]


def test_each_edition_keeps_its_pages_assets_navigation_and_about_page_under_its_version(docs, tmp_path):
    standard(docs)
    site = composed(docs, tmp_path)
    saved = docs / "editions/enterprise/8.1.0"

    assert (site / "enterprise/v8-1-0/guide.mdx").read_bytes() == (saved / "guide.mdx").read_bytes()
    assert (site / "enterprise/v8-1-0/static/logo.svg").read_text() == "<svg>enterprise 8.1.0</svg>\n"
    assert not (site / "enterprise/v8-1-0/docs.json").exists()
    assert versions(site)["enterprise/v8-1-0"]["tabs"] == [
        {
            "tab": "Documentation",
            "groups": [
                {"group": "Start", "pages": ["enterprise/v8-1-0/index", "enterprise/v8-1-0/guide",
                                             "enterprise/v8-1-0/about-this-version"]},
                {"group": "Enterprise", "pages": ["enterprise/v8-1-0/enterprise/security"]},
            ],
        }
    ]
    about = (site / "enterprise/v8-1-0/about-this-version.mdx").read_text()
    assert "LanceDB Enterprise 8.1.0, a qualified release" in about
    assert f"`lancedb/lancedb` commit `{made_up('lancedb', 'enterprise', '8.1.0')}`" in about
    # Public prose names the Enterprise documentation, not its repository or commit.
    assert "sophon" not in about and made_up("enterprise", "enterprise", "8.1.0") not in about
    oss = (site / "oss/v0-99-0/about-this-version.mdx").read_text()
    assert "the Enterprise documentation of LanceDB Enterprise 8.2.0" in oss


def test_switching_preferred_editions_does_not_change_shared_settings_or_assets(docs, tmp_path):
    standard(docs)
    before = tree(docs / "editions")
    site = composed(docs, tmp_path)

    # Mintlify loads every stylesheet on every page, so editions cannot each have one.
    assert sorted(p.relative_to(site).as_posix() for p in site.rglob("*.css")) == ["static/style.css"]
    assert (site / "static/style.css").read_text() == "/* shared style */\n"
    assert (site / "static/favicon.svg").read_text() == "<svg>shared</svg>\n"
    config = site_json(site)
    assert (config["name"], config["favicon"]) == ("LanceDB", "/static/favicon.svg")
    configure(docs, enterprise="8.1.0", oss="0.99.0")
    second = composed(docs, tmp_path, "second")
    shared = lambda config: {k: v for k, v in config.items() if k not in ("navigation", "redirects")}
    assert shared(site_json(second)) == shared(config)
    assert tree(second / "static") == tree(site / "static")
    assert tree(docs / "editions") == before
    assert next(r for r in site_json(second)["redirects"] if r["source"] == "/guide")["destination"] == "/enterprise/v8-1-0/guide"


def test_shared_settings_can_change_without_a_release_or_correction(docs, tmp_path):
    standard(docs)
    before = tree(docs / "editions"), tree(docs / "releases")
    first = composed(docs, tmp_path)
    path = docs / "site/settings.json"
    settings = json.loads(path.read_text())
    settings["banner"]["content"] = "New announcement"
    path.write_text(json.dumps(settings))
    (docs / "site/static/style.css").write_text("/* new shared style */\n")
    second = composed(docs, tmp_path, "second")
    assert site_json(second)["banner"]["content"] == "New announcement"
    assert (second / "static/style.css").read_text() == "/* new shared style */\n"
    assert tree(first / "enterprise") == tree(second / "enterprise")
    save(docs, "enterprise", "8.3.0")
    configure(docs, enterprise="8.3.0", oss="0.99.0")
    third = composed(docs, tmp_path, "third")
    assert site_json(third)["banner"] == site_json(second)["banner"]
    assert tree(third / "static") == tree(second / "static")
    for root, saved in zip((docs / "editions", docs / "releases"), before):
        assert {p: tree(root)[p] for p in saved} == saved


def test_shared_stylesheet_dependencies_keep_their_root_and_relative_paths(docs, tmp_path):
    standard(docs)
    css = '.hero { background: url("/static/hero.svg"); }\n@font-face { src: url("font.woff2"); }\n'
    (docs / "site/static/style.css").write_text(css)
    (docs / "site/static/hero.svg").write_text("<svg>hero</svg>")
    (docs / "site/static/font.woff2").write_bytes(b"font fixture")
    site = composed(docs, tmp_path)
    assert tree(site / "static") == tree(docs / "site/static")
    assert (site / "static/style.css").read_text() == css


@pytest.mark.parametrize("key", ["navigation", "redirects"])
def test_shared_settings_cannot_override_generated_routing(docs, tmp_path, key):
    standard(docs)
    path = docs / "site/settings.json"
    path.write_text(json.dumps({"name": "LanceDB", key: []}))
    with pytest.raises(compose.ComposeError, match="cannot set navigation or redirects"):
        composed(docs, tmp_path)
    assert not (tmp_path / "site").exists()


@pytest.mark.parametrize("path", ["private.txt", "static/page.mdx"])
def test_shared_files_cannot_add_unversioned_pages_or_other_root_files(docs, tmp_path, path):
    standard(docs)
    (docs / "site" / path).write_text("not an asset")
    with pytest.raises(compose.ComposeError, match="shared assets belong in static/"):
        composed(docs, tmp_path)


def test_shared_assets_cannot_replace_a_different_example_download(docs, tmp_path):
    files = edition_files("8.1.0")
    files["guide.mdx"] += '\n```sh\ncurl https://docs.lancedb.com/static/favicon.svg\n```\n'
    save(docs, "enterprise", "8.1.0", files)
    configure(docs, enterprise="8.1.0")
    with pytest.raises(compose.ComposeError, match="unversioned path must be the same"):
        composed(docs, tmp_path)
    (docs / "site/static/favicon.svg").write_text(files["static/favicon.svg"])
    assert (composed(docs, tmp_path) / "static/favicon.svg").read_text() == files["static/favicon.svg"]


def test_shared_assets_refuse_symlinks(docs, tmp_path):
    standard(docs)
    (docs / "site/static/link").symlink_to(docs / "editions")
    with pytest.raises(candidate.CandidateError, match="symlink"):
        composed(docs, tmp_path)


# links


PAGE = """---
title: "Links"
icon: "/static/logo.svg"
---

[Root](/guide#setup) [Home](/) [Folder](/enterprise/) [Page](/enterprise/security)
[Host](https://docs.lancedb.com/start?tab=py) [Moved](https://docs.lancedb.com/old/x) [Own anchor](https://docs.lancedb.com/old/x#own)
[https://docs.lancedb.com/guide](https://docs.lancedb.com/guide)
[Relative](guide) [External](https://example.com/guide) [Anchor](#setup) [Mail](mailto:a@example.com)
![Logo](/static/logo.svg) <img src="/static/logo.svg" /> <Card title="Guide" icon="book" href="/guide" />
<Card icon="/static/logo.svg" href={"/guide"} />

[ref]: /guide

`/guide` stays as written in code, and so does `https://docs.lancedb.com/guide`.

{/* A draft section links to [the plan](/plans/next) and https://docs.lancedb.com/plans. */}

```bash
curl -O https://docs.lancedb.com/static/data/rows.csv
cd /guide
```
"""
# A link to the production host lands where its redirects lead, keeping a
# redirect's anchor unless the link has its own.
POINTED = """---
title: "Links"
icon: "/enterprise/v8-2-0/static/logo.svg"
---

[Root](/enterprise/v8-2-0/guide#setup) [Home](/enterprise/v8-2-0) [Folder](/enterprise/v8-2-0/enterprise/) [Page](/enterprise/v8-2-0/enterprise/security)
[Host](/enterprise/v8-2-0/guide?tab=py) [Moved](/enterprise/v8-2-0/guide#setup) [Own anchor](/enterprise/v8-2-0/guide#own)
[https://docs.lancedb.com/guide](/enterprise/v8-2-0/guide)
[Relative](guide) [External](https://example.com/guide) [Anchor](#setup) [Mail](mailto:a@example.com)
![Logo](/enterprise/v8-2-0/static/logo.svg) <img src="/enterprise/v8-2-0/static/logo.svg" /> <Card title="Guide" icon="book" href="/enterprise/v8-2-0/guide" />
<Card icon="/enterprise/v8-2-0/static/logo.svg" href={"/enterprise/v8-2-0/guide"} />

[ref]: /enterprise/v8-2-0/guide

`/guide` stays as written in code, and so does `https://docs.lancedb.com/guide`.

{/* A draft section links to [the plan](/plans/next) and https://docs.lancedb.com/plans. */}

```bash
curl -O https://docs.lancedb.com/static/data/rows.csv
cd /guide
```
"""


def test_links_point_into_their_own_edition_and_code_and_comments_stay_as_written(docs, tmp_path):
    save(docs, "enterprise", "8.2.0", edition_files("8.2.0", extra={
        "links.mdx": PAGE, "static/data/rows.csv": "a,b\n"}))
    configure(docs, enterprise="8.2.0")
    site = composed(docs, tmp_path)

    assert (site / "enterprise/v8-2-0/links.mdx").read_text() == POINTED


@pytest.mark.parametrize(
    "text, problem",
    [
        ("See [OSS](/oss/v0-99-0/guide).", r"/oss/v0-99-0/guide points into another edition"),
        ("See [up](../../outside).", r"\.\./\.\./outside leaves the edition"),
        ("See https://docs.lancedb.com/guide for more.", r"https://docs.lancedb.com/guide is a bare link"),
    ],
)
def test_a_link_that_would_leave_its_edition_stops_composition(docs, tmp_path, text, problem):
    save(docs, "enterprise", "8.2.0", edition_files("8.2.0", extra={
        "enterprise/security.mdx": f'---\ntitle: "Security"\n---\n\n{text}\n'}))
    configure(docs, enterprise="8.2.0")

    with pytest.raises(compose.ComposeError, match=rf"editions/enterprise/8.2.0/enterprise/security.mdx:5: {problem}"):
        composed(docs, tmp_path)
    assert not (tmp_path / "site").exists()


def test_a_draft_page_is_composed_hidden_and_its_links_are_not_checked(docs, tmp_path):
    save(docs, "enterprise", "8.2.0", edition_files("8.2.0", extra={
        ".mintignore": "# Drafts\ndraft.mdx\nnotes/\n!keep.mdx\n",
        "draft.mdx": "Draft for the next release: see [OSS](/oss/v0-99-0/guide) and https://docs.lancedb.com/x.\n",
        "notes/plan.mdx": "A plan.\n",
    }))
    save(docs, "oss", "0.99.0")
    configure(docs, enterprise="8.2.0", oss="0.99.0")
    site = composed(docs, tmp_path)

    assert (site / ".mintignore").read_text() == (
        "# Composed by scripts/compose.py from each edition's .mintignore.\n"
        "\n# enterprise 8.2.0\n# Drafts\n/enterprise/v8-2-0/**/draft.mdx\n/enterprise/v8-2-0/**/notes/\n"
        "!/enterprise/v8-2-0/**/keep.mdx\n"
        "\n# oss 0.99.0\n/oss/v0-99-0/**/draft.mdx\n"
    )
    assert (site / "enterprise/v8-2-0/notes/plan.mdx").is_file()
    sources = {r["source"] for r in unversioned(site)}
    assert "/guide" in sources and not sources & {"/draft", "/notes/plan"}


def test_the_composed_mintignore_hides_what_each_editions_own_did_and_nothing_else(tmp_path):
    lines = ["draft.mdx", "/top.mdx", "plans/", "!plans/keep.mdx", "notes/*.mdx", "!notes/keep.mdx",
             "**/deep/x.mdx", "sub/**", "!sub/again.mdx", "*.tmp", "[ab]c.mdx"]
    paths = ["draft.mdx", "a/draft.mdx", "top.mdx", "a/top.mdx", "plans/x.mdx", "plans/keep.mdx",
             "a/plans/y.mdx", "notes/x.mdx", "notes/keep.mdx", "notes/sub/x.mdx", "deep/x.mdx",
             "a/b/deep/x.mdx", "sub/x.mdx", "sub/again.mdx", "sub/d/e.mdx", "x.tmp", "ac.mdx", "cc.mdx"]
    # Mint reads .mintignore with gitignore rules, as git reads a .gitignore.
    reference = tmp_path / "reference"
    reference.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=reference, env=environment(), check=True)
    (reference / ".gitignore").write_text("\n".join(lines) + "\n")
    hidden = set(subprocess.run(["git", "check-ignore", "--no-index", "--stdin"], cwd=reference,
                                env=environment(), input="\n".join(paths), capture_output=True,
                                text=True).stdout.split())
    assert "plans/keep.mdx" in hidden and "sub/again.mdx" not in hidden

    rules = compose.ignore_rules(compose.pointed_ignore(line, "oss/v0-99-0") for line in lines)
    assert {p for p in paths if compose.ignored(rules, f"oss/v0-99-0/{p}")} == hidden
    assert not any(compose.ignored(rules, f"enterprise/v8-2-0/{p}") for p in paths)


# downloads


CURL = "```bash\ncurl -O https://docs.lancedb.com/static/data/rows.csv\n```\n"


def download_edition(label: str, rows: str) -> dict:
    files = edition_files(label, extra={"static/data/rows.csv": rows})
    files["guide.mdx"] += CURL
    return files


def test_a_file_that_code_downloads_is_also_served_where_the_code_fetches_it(docs, tmp_path):
    save(docs, "enterprise", "8.1.0", download_edition("8.1.0", "a,b\n"))
    save(docs, "enterprise", "8.2.0", download_edition("8.2.0", "a,b\n"))
    configure(docs, enterprise="8.2.0")
    summary = compose.compose(docs, tmp_path / "site")
    site = tmp_path / "site"

    assert (site / "static/data/rows.csv").read_text() == "a,b\n"
    assert (site / "enterprise/v8-1-0/static/data/rows.csv").read_text() == "a,b\n"
    assert (site / "enterprise/v8-1-0/guide.mdx").read_text().endswith(CURL)
    assert summary["downloads"] == {"/static/data/rows.csv": ["enterprise 8.2.0", "enterprise 8.1.0"]}


def test_examples_that_download_different_files_from_one_address_stop_composition(docs, tmp_path):
    save(docs, "enterprise", "8.1.0", download_edition("8.1.0", "a,b\n"))
    save(docs, "enterprise", "8.2.0", download_edition("8.2.0", "a,b,c\n"))
    configure(docs, enterprise="8.2.0")

    with pytest.raises(compose.ComposeError, match=r"/static/data/rows.csv is the download of enterprise "
                       r"8.2.0 and the download of enterprise 8.1.0, and they differ"):
        composed(docs, tmp_path)


# redirects


def test_redirects_stay_in_their_edition_and_keep_their_anchors(docs, tmp_path):
    standard(docs)
    redirects = {r["source"]: r["destination"] for r in site_json(composed(docs, tmp_path))["redirects"]}

    assert redirects["/enterprise/v8-1-0/start"] == "/enterprise/v8-1-0/guide"
    assert redirects["/enterprise/v8-1-0/old/:slug*"] == "/enterprise/v8-1-0/guide#setup"
    assert redirects["/oss/v0-99-0/start"] == "/oss/v0-99-0/guide"


def test_unversioned_urls_redirect_to_the_preferred_enterprise_edition_or_are_reported(docs, tmp_path):
    for version in ("8.1.0", "8.2.0"):
        save(docs, "enterprise", version)
    oss = edition_files("0.99.0", extra={"new.mdx": '---\ntitle: "New"\n---\n\nOnly in OSS 0.99.0.\n'})
    save(docs, "oss", "0.99.0", oss)
    configure(docs, enterprise="8.1.0", oss="0.99.0")
    summary = compose.compose(docs, tmp_path / "site")

    assert unversioned(tmp_path / "site") == [
        {"source": "/enterprise/security", "destination": "/enterprise/v8-1-0/enterprise/security", "permanent": False},
        {"source": "/guide", "destination": "/enterprise/v8-1-0/guide", "permanent": False},
        {"source": "/old/:slug*", "destination": "/enterprise/v8-1-0/guide#setup", "permanent": False},
        {"source": "/start", "destination": "/enterprise/v8-1-0/guide", "permanent": False},
    ]
    # No catch-all: a page the preferred Enterprise edition lacks is left for review.
    assert summary["unresolved"] == [("/new", ["oss 0.99.0"])]


@pytest.mark.parametrize(
    "redirects, problem",
    [
        ([{"source": "/a", "destination": "/b"}, {"source": "/b", "destination": "/a"}],
         r"the redirect from /enterprise/v8-2-0/a loops"),
        ([{"source": "/enterprise/:slug*", "destination": "/guide"}],
         r"the unversioned redirect from /enterprise/:slug\* would capture /enterprise/v8-2-0"),
        ([{"source": "/gone", "destination": "/nowhere"}],
         r"the redirect from /gone leads to /nowhere, which enterprise 8.2.0 does not publish"),
    ],
)
def test_redirects_that_loop_capture_pages_or_lead_nowhere_stop_composition(docs, tmp_path, redirects, problem):
    save(docs, "enterprise", "8.2.0", edition_files("8.2.0", redirects=redirects))
    configure(docs, enterprise="8.2.0")

    with pytest.raises(compose.ComposeError, match=problem):
        composed(docs, tmp_path)


# adding editions and the preferred edition


def test_adding_editions_leaves_older_editions_and_unversioned_redirects_unchanged(docs, tmp_path):
    save(docs, "enterprise", "8.1.0")
    save(docs, "oss", "0.98.0")
    configure(docs, enterprise="8.1.0", oss="0.98.0")
    before = composed(docs, tmp_path, "before")
    # A newer release, then a maintenance release prepared after it.
    for product, version in (("enterprise", "8.2.0"), ("enterprise", "8.1.1"), ("oss", "0.99.0")):
        save(docs, product, version)
    after = composed(docs, tmp_path, "after")

    for prefix in ("enterprise/v8-1-0", "oss/v0-98-0"):
        assert tree(before / prefix) == tree(after / prefix)
        assert versions(before)[prefix] == versions(after)[prefix]
    assert list(versions(after)) == ["enterprise/v8-2-0", "enterprise/v8-1-1", "enterprise/v8-1-0",
                                     "oss/v0-99-0", "oss/v0-98-0"]
    assert unversioned(before) == unversioned(after)


def test_the_preferred_edition_changes_only_with_compose_yaml(docs, tmp_path):
    save(docs, "enterprise", "8.2.0")
    save(docs, "enterprise", "8.1.1")
    configure(docs, enterprise="8.2.0")
    first = compose.compose(docs, tmp_path / "first")
    # Composing again later, with every saved file touched since, builds the same site.
    for path in (docs / "editions").rglob("*"):
        os.utime(path, (1, 1))
    again = compose.compose(docs, tmp_path / "again")
    assert (again["sha256"], again["default"]) == (first["sha256"], "enterprise 8.2.0")
    assert tree(tmp_path / "first") == tree(tmp_path / "again")

    configure(docs, enterprise="8.1.1")
    changed = compose.compose(docs, tmp_path / "changed")
    assert changed["default"] == "enterprise 8.1.1"
    assert {r["destination"] for r in unversioned(tmp_path / "changed")} >= {"/enterprise/v8-1-1/guide"}


@pytest.mark.parametrize(
    "saved, preferred, problem",
    [
        ([], {}, r"there are no saved editions"),
        ([("enterprise", "8.2.0")], {}, r"compose.yaml names no preferred enterprise edition; it has 8.2.0"),
        ([("enterprise", "8.2.0")], {"enterprise": "8.3.0"}, r"prefers enterprise 8.3.0, which is not a saved edition"),
        ([("enterprise", "8.2.0")], {"enterprise": "8.2.0", "oss": "0.99.0"}, r"prefers oss 0.99.0"),
        ([("oss", "0.99.0")], {"oss": "0.99.0"}, r"lists enterprise first, so readers land on its preferred edition"),
    ],
)
def test_compose_yaml_must_name_a_saved_preferred_edition_for_each_product(docs, tmp_path, saved, preferred, problem):
    for product, version in saved:
        save(docs, product, version)
    configure(docs, **preferred)

    with pytest.raises(compose.ComposeError, match=problem):
        composed(docs, tmp_path)


# corrections


def test_a_recorded_correction_is_composed_and_listed_on_the_about_page(docs, tmp_path):
    save(docs, "enterprise", "8.1.0")
    configure(docs, enterprise="8.1.0")
    corrected = '---\ntitle: "Guide"\n---\n\n## Setup {#setup}\n\nThe corrected guide.\n'
    correct(docs, "enterprise", "8.1.0", "guide.mdx", corrected, "Fix the {retry} <count> in *guide*.")
    site = composed(docs, tmp_path)

    assert (site / "enterprise/v8-1-0/guide.mdx").read_text() == corrected
    about = (site / "enterprise/v8-1-0/about-this-version.mdx").read_text()
    assert about.endswith("## Corrections {#corrections}\n\n- Fix the \\{retry\\} \\<count\\> in \\*guide\\*. (`guide.mdx`)\n")


def test_an_edition_changed_without_a_recorded_correction_is_not_composed(docs, tmp_path):
    save(docs, "enterprise", "8.1.0")
    configure(docs, enterprise="8.1.0")
    (docs / "editions/enterprise/8.1.0/guide.mdx").write_text("An unreviewed change.\n")

    with pytest.raises(compose.ComposeError, match=r"do not match their records"):
        composed(docs, tmp_path)


# the About page and the command


def compile_mdx(text: str) -> subprocess.CompletedProcess:
    script = (
        f"import {{compile}} from {json.dumps(MDX_COMPILER)};\n"
        "let source = '';\n"
        "for await (const chunk of process.stdin) source += chunk;\n"
        "await compile(source);\n"
    )
    # Mintlify reads a heading's `{#id}` itself; the bare compiler cannot.
    body = re.sub(r"^(#{1,6} .*?) \{#[A-Za-z0-9][A-Za-z0-9._-]*\}$", r"\1", text.split("\n---\n", 1)[1], flags=re.M)
    return subprocess.run(["node", "--input-type=module", "-e", script], input=body,
                          capture_output=True, text=True)


@pytest.mark.skipif(
    not (MDX_COMPILER and shutil.which("node")),
    reason="set MDX_COMPILER to @mdx-js/mdx's index.js and put node on PATH",
)
def test_about_pages_compile_as_mdx_whatever_a_correction_says(docs, tmp_path):
    standard(docs)
    correct(docs, "enterprise", "8.1.0", "guide.mdx", "Corrected.\n", "Escape {braces}, <tags>, `ticks` and |pipes|.")
    site = composed(docs, tmp_path)

    for path in sorted(site.rglob("about-this-version.mdx")):
        result = compile_mdx(path.read_text())
        assert result.returncode == 0, f"{path}: {result.stderr}"


def test_the_command_reports_unversioned_paths_it_leaves_without_a_redirect(docs, tmp_path, monkeypatch, capsys):
    save(docs, "enterprise", "8.2.0")
    save(docs, "oss", "0.99.0", edition_files("0.99.0", extra={"new.mdx": "Only in OSS.\n"}))
    configure(docs, enterprise="8.2.0", oss="0.99.0")
    monkeypatch.setattr(compose, "REPO_ROOT", docs)

    assert compose.main(["--output", str(tmp_path / "site")]) == 0
    out = capsys.readouterr().out
    assert "  enterprise 8.2.0 at /enterprise/v8-2-0 (preferred)\n" in out
    assert (
        "unversioned paths without a page in enterprise 8.2.0, left without a redirect for review (1):\n"
        "  /new (oss 0.99.0)\n"
    ) in out


@pytest.mark.parametrize("path", ["editions", "site", "site/static"])
def test_composing_never_writes_over_the_saved_editions_or_shared_inputs(docs, tmp_path, path):
    standard(docs)

    with pytest.raises(compose.ComposeError, match=r"overlaps"):
        compose.compose(docs, docs / path)
    assert (docs / "editions/enterprise/8.1.0/guide.mdx").is_file()
    assert (docs / "site/settings.json").is_file()


@pytest.mark.parametrize("value", ['"/enterprise/security"', "{'/enterprise/security'}", '{`/enterprise/security`}'])
def test_static_jsx_links_stay_with_the_selected_edition(docs, tmp_path, value):
    markup = f'<Card title="Next" href={value} />'
    files = edition_files("8.1.0")
    files["guide.mdx"] += markup + '\n'
    save(docs, "enterprise", "8.1.0", files)
    save(docs, "enterprise", "8.2.0")
    configure(docs, enterprise="8.2.0")
    site = composed(docs, tmp_path)
    text = (site / "enterprise/v8-1-0/guide.mdx").read_text()
    assert markup.replace('/enterprise/security', '/enterprise/v8-1-0/enterprise/security') in text
    if MDX_COMPILER and shutil.which("node"):
        result = compile_mdx(text)
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("value", ['{target}', '{`/guide/${section}`}', '{"/guide" + suffix}', '{\n"/guide"\n}'])
def test_unresolved_jsx_links_stop_before_output(docs, tmp_path, value):
    files = edition_files("8.1.0")
    files["guide.mdx"] += f'<Card href={value} />\n'
    save(docs, "enterprise", "8.1.0", files)
    configure(docs, enterprise="8.1.0")
    with pytest.raises(compose.ComposeError, match=r'guide.mdx:\d+: href must be'):
        composed(docs, tmp_path)
    assert not (tmp_path / "site").exists()


def test_jsx_in_examples_and_comments_is_not_rewritten(docs, tmp_path):
    examples = '\n``<Card href={`/guide`} />``\n\n```jsx\n<Card href={target} />\n```\n{/* <Card href={target} /> */}\n'
    files = edition_files("8.1.0")
    files["guide.mdx"] += examples
    save(docs, "enterprise", "8.1.0", files)
    configure(docs, enterprise="8.1.0")
    assert (composed(docs, tmp_path) / "enterprise/v8-1-0/guide.mdx").read_text().endswith(examples)


@pytest.mark.parametrize("host", ['https://docs.lancedb.com', '//docs.lancedb.com'])
def test_same_host_redirects_stay_in_the_edition(docs, tmp_path, host):
    redirects = [
        {"source": "/start", "destination": host + "/middle?mode=old#setup"},
        {"source": "/middle", "destination": "/guide"},
        {"source": "/external", "destination": "https://example.com/guide?q=1#x"},
    ]
    files = edition_files("8.1.0", redirects=redirects)
    files["guide.mdx"] += '\n[Old](https://docs.lancedb.com/start)\n'
    save(docs, "enterprise", "8.1.0", files)
    save(docs, "enterprise", "8.2.0")
    configure(docs, enterprise="8.2.0")
    site = composed(docs, tmp_path)
    targets = {r['source']: r['destination'] for r in site_json(site)['redirects']}
    assert targets['/enterprise/v8-1-0/start'] == '/enterprise/v8-1-0/middle?mode=old#setup'
    assert targets['/enterprise/v8-1-0/external'] == 'https://example.com/guide?q=1#x'
    assert '[Old](/enterprise/v8-1-0/guide?mode=old#setup)' in (site / 'enterprise/v8-1-0/guide.mdx').read_text()


@pytest.mark.parametrize("redirects", [
    [{"source": "/guide", "destination": "/guide"}],
    [{"source": "/guide", "destination": "/enterprise/security"},
     {"source": "/enterprise/security", "destination": "/guide"}],
    [{"source": "/old/:slug*", "destination": "/old/:slug*"}],
    [{"source": "/old/:slug*", "destination": "/other/:slug*"},
     {"source": "/other/:slug*", "destination": "/old/:slug*"}],
    [{"source": "/old/:slug*", "destination": "/old/more/:slug*"}],
])
def test_redirects_cannot_loop_even_with_pages_or_wildcards(docs, tmp_path, redirects):
    save(docs, "enterprise", "8.1.0", edition_files("8.1.0", redirects=redirects))
    configure(docs, enterprise="8.1.0")
    with pytest.raises(compose.ComposeError, match=r'loops|does not terminate'):
        composed(docs, tmp_path)


def test_redirect_takes_precedence_over_an_existing_page(docs, tmp_path):
    files = edition_files("8.1.0", redirects=[{"source": "/guide", "destination": "/enterprise/security#detail"}])
    files["index.mdx"] += '\n[Moved](https://docs.lancedb.com/guide)\n'
    save(docs, "enterprise", "8.1.0", files)
    configure(docs, enterprise="8.1.0")
    assert '[Moved](/enterprise/v8-1-0/enterprise/security#detail)' in (composed(docs, tmp_path) / 'enterprise/v8-1-0/index.mdx').read_text()


@pytest.mark.parametrize("target", ['/../../enterprise/security', '/%2e%2e/guide', '/enterprise/v8-1-0/../guide'])
def test_root_paths_cannot_traverse_out_of_the_edition(docs, tmp_path, target):
    files = edition_files("8.1.0")
    files["guide.mdx"] += f'\n[Escape]({target})\n'
    save(docs, "enterprise", "8.1.0", files)
    configure(docs, enterprise="8.1.0")
    with pytest.raises(compose.ComposeError, match='leaves the edition'):
        composed(docs, tmp_path)


def test_safe_dot_segments_are_normalized(docs, tmp_path):
    files = edition_files("8.1.0")
    files["guide.mdx"] += '\n[Security](/guide/../enterprise/./security?q=1#detail)\n'
    save(docs, "enterprise", "8.1.0", files)
    configure(docs, enterprise="8.1.0")
    assert '[Security](/enterprise/v8-1-0/enterprise/security?q=1#detail)' in (composed(docs, tmp_path) / 'enterprise/v8-1-0/guide.mdx').read_text()


def test_exact_redirect_wins_over_an_earlier_wildcard(docs, tmp_path):
    files = edition_files('8.1.0', redirects=[
        {'source': '/old/:slug*', 'destination': '/guide'},
        {'source': '/old/guide', 'destination': '/enterprise/security'},
    ])
    files['index.mdx'] += '\n[Exact](https://docs.lancedb.com/old/guide)\n'
    save(docs, 'enterprise', '8.1.0', files)
    configure(docs, enterprise='8.1.0')
    assert '[Exact](/enterprise/v8-1-0/enterprise/security)' in (composed(docs, tmp_path) / 'enterprise/v8-1-0/index.mdx').read_text()


def test_wildcard_to_parent_keeps_the_parent_page(docs, tmp_path):
    files = edition_files('8.1.0', redirects=[{'source': '/guide/:slug*', 'destination': '/guide'}])
    files['index.mdx'] += '\n[Guide](https://docs.lancedb.com/guide/old)\n'
    save(docs, 'enterprise', '8.1.0', files)
    configure(docs, enterprise='8.1.0')
    site = composed(docs, tmp_path)
    assert '[Guide](/enterprise/v8-1-0/guide)' in (site / 'enterprise/v8-1-0/index.mdx').read_text()


@pytest.mark.parametrize("target", ["&#47;enterprise/security", "&#x2f;enterprise/security", "&sol;enterprise/security", "/guide?a=1&amp;b=2"])
def test_html_entities_in_link_attributes_are_refused(docs, tmp_path, target):
    files = edition_files("8.1.0")
    files["guide.mdx"] += f'\n<a href="{target}">Guide</a>\n'
    save(docs, "enterprise", "8.1.0", files)
    configure(docs, enterprise="8.1.0")
    with pytest.raises(compose.ComposeError, match=r"guide.mdx:\d+: href contains HTML entities"):
        composed(docs, tmp_path)
    assert not (tmp_path / "site").exists()


def test_literal_query_in_link_expression_is_preserved(docs, tmp_path):
    files = edition_files("8.1.0")
    files["guide.mdx"] += '\n<a href={"/guide?a=1&b=2"}>Guide</a>\n'
    save(docs, "enterprise", "8.1.0", files)
    configure(docs, enterprise="8.1.0")
    page = (composed(docs, tmp_path) / "enterprise/v8-1-0/guide.mdx").read_text()
    assert 'href={"/enterprise/v8-1-0/guide?a=1&b=2"}' in page


def test_attribute_names_in_urls_and_prose_are_not_jsx(docs, tmp_path):
    text = '\n[Query](/guide?href=/enterprise/security)\nThe parameter href=target is prose.\n<Card title="href=example > text" href={`/guide`} />\n'
    files = edition_files('8.1.0'); files['guide.mdx'] += text
    save(docs, 'enterprise', '8.1.0', files)
    configure(docs, enterprise='8.1.0')
    page = (composed(docs, tmp_path) / 'enterprise/v8-1-0/guide.mdx').read_text()
    assert '[Query](/enterprise/v8-1-0/guide?href=/enterprise/security)' in page
    assert 'The parameter href=target is prose.' in page
    assert '<Card title="href=example > text" href={`/enterprise/v8-1-0/guide`} />' in page


@pytest.mark.parametrize('attribute', [
    'onClick={() => {}}',
    'data={{message: "> }", nested: {enabled: true}}}',
    'onClick={() => { /* } > */ return "done"; }}',
    'title={`literal > }`}',
    'disabled={count > 0}',
])
def test_other_jsx_expressions_cannot_hide_a_link(docs, tmp_path, attribute):
    markup = f'<a {attribute}\n  href={{`/enterprise/security`}}>Next</a>'
    files = edition_files('8.1.0')
    files['guide.mdx'] += '\n' + markup + '\n'
    save(docs, 'enterprise', '8.1.0', files)
    save(docs, 'enterprise', '8.2.0')
    configure(docs, enterprise='8.2.0')
    page = (composed(docs, tmp_path) / 'enterprise/v8-1-0/guide.mdx').read_text()
    assert markup.replace('/enterprise/security', '/enterprise/v8-1-0/enterprise/security') in page
    if MDX_COMPILER and shutil.which('node'):
        result = compile_mdx(page)
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('attributes', [
    'onClick={() => /}/.test(value)} href={`/guide`}',
    'title={`value ${suffix}`} href={`/guide`}',
    '{...props} href={`/guide`}',
    'title={<a href="/guide">Guide</a>} href={`/guide`}',
    'onClick={() => {\n}} href={`/guide`}',
    'href\n = {`/guide`}',
    'title="long\n title" href={`/guide`}',
])
def test_unsupported_jsx_attributes_fail_instead_of_hiding_links(docs, tmp_path, attributes):
    files = edition_files('8.1.0')
    files['guide.mdx'] += f'\n<a {attributes}>Next</a>\n'
    save(docs, 'enterprise', '8.1.0', files)
    configure(docs, enterprise='8.1.0')
    with pytest.raises(compose.ComposeError, match=r'guide.mdx:\d+:'):
        composed(docs, tmp_path)
    assert not (tmp_path / 'site').exists()


@pytest.mark.parametrize('source', ['/legacy/article-*', '/legacy/*', '/:first*/:second*', '/legacy/:slug+', '/legacy/:slug?'])
def test_unhandled_redirect_patterns_are_refused(docs, tmp_path, source):
    files = edition_files('8.1.0', redirects=[
        {'source': source, 'destination': '/guide'},
        {'source': '/guide', 'destination': '/legacy/article-one'},
    ], extra={'legacy/article-one.mdx': '---\ntitle: Legacy\n---\nExisting page.\n'})
    save(docs, 'enterprise', '8.1.0', files)
    configure(docs, enterprise='8.1.0')
    with pytest.raises(compose.ComposeError, match='unsupported redirect source'):
        composed(docs, tmp_path)
    assert not (tmp_path / 'site').exists()
