from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sync_hf_datasets as hf

SOURCE = Path(__file__).resolve().parents[2]
CARD = "---\ntitle: Upstream\n---\n# Upstream\n\nA **sample** dataset with `values`.\n\n## Details\n\nLance<>HF\n"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    datasets = tmp_path / "docs" / "datasets"
    shutil.copytree(SOURCE / "docs" / "datasets", datasets)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copyfile(SOURCE / "scripts" / "hf_datasets.yaml", scripts / "hf_datasets.yaml")
    fragment = tmp_path / "docs" / "docs.nav.json"
    shutil.copyfile(SOURCE / "docs" / "docs.nav.json", fragment)
    for name, value in {
        "REPO_ROOT": tmp_path,
        "CONFIG_PATH": scripts / "hf_datasets.yaml",
        "DATASETS_DIR": datasets,
        "INDEX_PATH": datasets / "index.mdx",
        "NAV_FRAGMENT_PATH": fragment,
    }.items():
        monkeypatch.setattr(hf, name, value)
    monkeypatch.setattr(hf, "fetch_card", lambda ds: CARD)
    return tmp_path


def dataset(slug, category="Robotics"):
    return hf.Dataset(slug, slug, slug, slug.title(), category)


def tab(fragment):
    return next(item["entry"] for item in fragment["insert"] if item["entry"].get("tab") == "Datasets")


def snapshot(repo):
    return {path.relative_to(repo): path.read_bytes() for path in repo.rglob("*") if path.is_file()}


def test_current_fragment_is_byte_identical_with_unchanged_config(repo):
    original = hf.NAV_FRAGMENT_PATH.read_text()
    assert hf.render_nav_fragment(hf.load_config()) == original
    fragment = json.loads(original)
    assert tab(fragment)["pages"][1]["group"] == "Robotics"
    assert tab(fragment)["pages"][1]["pages"][0] == "integrations/lerobotdataset"


def test_add_remove_and_preserve_fragment_content(repo):
    original = json.loads(hf.NAV_FRAGMENT_PATH.read_text())
    original["set"] = [{"into": ["Documentation"], "key": "icon", "value": "book"}]
    robotics = tab(original)["pages"][1]
    robotics["icon"] = "robot"
    robotics["pages"].append("integrations/robot-guide")
    tab(original)["pages"].insert(2, {"group": "Editorial", "pages": [], "icon": "book"})
    hf.NAV_FRAGMENT_PATH.write_text(json.dumps(original))
    categories = [hf.Category("Robotics", (dataset("new-robot"),)), hf.Category("New category", (dataset("new-card", "New category"),))]
    result = json.loads(hf.render_nav_fragment(categories))
    expected = copy.deepcopy(original)
    expected_tab = tab(expected)
    expected_tab["pages"] = [
        "datasets/index",
        {"group": "Robotics", "icon": "robot", "pages": ["integrations/lerobotdataset", "datasets/new-robot", "integrations/robot-guide"]},
        {"group": "Editorial", "pages": [], "icon": "book"},
        {"group": "New category", "pages": ["datasets/new-card"]},
    ]
    assert result == expected
    assert "groups" not in tab(result)
    assert list(result) == list(original)
    assert list(result["insert"][1]) == list(original["insert"][1])


def test_removed_category_keeps_editorial_link(repo):
    result = json.loads(hf.render_nav_fragment([]))
    assert tab(result)["pages"] == ["datasets/index", {"group": "Robotics", "pages": ["integrations/lerobotdataset"]}]


def test_offline_sync_pages_index_and_navigation(repo, monkeypatch):
    categories = [hf.Category("Robotics", (dataset("new-robot"),))]
    monkeypatch.setattr(hf, "load_config", lambda: categories)
    before_index = hf.INDEX_PATH.read_text()
    before_nav = json.loads(hf.NAV_FRAGMENT_PATH.read_text())
    hf.sync()
    assert {path.name for path in hf.DATASETS_DIR.glob("*.mdx")} == {"index.mdx", "new-robot.mdx"}
    page = (hf.DATASETS_DIR / "new-robot.mdx").read_text()
    assert 'title: "New-Robot"' in page
    assert 'description: "A sample dataset with values."' in page
    assert "https://huggingface.co/datasets/lance-format/new-robot" in page
    assert "Lance&lt;&gt;HF" in page
    index = hf.INDEX_PATH.read_text()
    assert 'href="/datasets/new-robot"' in index
    assert index.split(hf.SYNC_START)[0] == before_index.split(hf.SYNC_START)[0]
    assert index.split(hf.SYNC_END)[1] == before_index.split(hf.SYNC_END)[1]
    nav = json.loads(hf.NAV_FRAGMENT_PATH.read_text())
    assert nav["insert"][0] == before_nav["insert"][0]
    assert nav["redirects"] == before_nav["redirects"]
    assert tab(nav)["pages"][1]["pages"] == ["integrations/lerobotdataset", "datasets/new-robot"]


@pytest.mark.parametrize("failure", ["fetch", "markers", "missing_tab", "duplicate_tab", "wrong_shape", "invalid_json"])
def test_sync_failure_leaves_no_partial_output(repo, monkeypatch, failure):
    categories = [hf.Category("Robotics", (dataset("first"), dataset("second")))]
    monkeypatch.setattr(hf, "load_config", lambda: categories)
    if failure == "fetch":
        def fetch(ds):
            if ds.slug == "second":
                raise RuntimeError("mock fetch error")
            return CARD
        monkeypatch.setattr(hf, "fetch_card", fetch)
    elif failure == "markers":
        hf.INDEX_PATH.write_text("missing markers")
    else:
        nav = json.loads(hf.NAV_FRAGMENT_PATH.read_text())
        if failure == "missing_tab":
            nav["insert"].pop()
        elif failure == "duplicate_tab":
            nav["insert"].append(copy.deepcopy(nav["insert"][1]))
        elif failure == "wrong_shape":
            tab(nav)["groups"] = tab(nav).pop("pages")
        hf.NAV_FRAGMENT_PATH.write_text("invalid json" if failure == "invalid_json" else json.dumps(nav))
    before = snapshot(repo)
    with pytest.raises((RuntimeError, json.JSONDecodeError)):
        hf.sync()
    assert snapshot(repo) == before


def test_dry_run_fetches_and_validates_without_writes(repo, monkeypatch):
    fetched = []
    monkeypatch.setattr(hf, "fetch_card", lambda ds: fetched.append(ds.slug) or CARD)
    before = snapshot(repo)
    hf.sync(dry_run=True)
    assert fetched == [ds.slug for cat in hf.load_config() for ds in cat.datasets]
    assert snapshot(repo) == before
