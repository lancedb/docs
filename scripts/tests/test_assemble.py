"""Tests for the assembler's overlay, navigation, private-root and availability guards.

Each test builds a small site in a temporary directory: a reference root that
owns `enterprise/security` and the navigation, and a private overlay root at
`sophon/docs/web` beside an internal folder that must never be published.

Run with `make test-assemble`.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import assemble  # noqa: E402

PUBLIC = "Enterprise security page\n"
INTERNAL = "Internal engineering notes\n"
NAVIGATION = {
    "navigation": {
        "tabs": [
            {
                "tab": "Documentation",
                "groups": [{"group": "Security", "pages": ["enterprise/security"]}],
            }
        ]
    }
}


@pytest.fixture
def base(tmp_path: Path) -> Path:
    """The test's directory with symlinks resolved.

    With no checkout to anchor on, the private-root guard checks every directory
    on the path, so a symlinked temporary directory (as on macOS) would trip it.
    """
    return tmp_path.resolve()


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def site(base: Path, *, checkout: bool = True) -> dict[str, Path]:
    """A reference root, a Sophon-like checkout, and an internal folder."""
    reference = base / "lancedb/docs/web"
    write(reference / "docs.json", json.dumps(NAVIGATION))
    write(reference / "enterprise/security.mdx", "Open-source security page\n")
    write(reference / "static/styles/style.css", "body {}\n")

    sophon = base / "sophon"
    if checkout:
        (sophon / ".git").mkdir(parents=True)
    write(sophon / "docs/ci.md", INTERNAL)
    internal = base / "internal"
    write(internal / "enterprise/security.mdx", INTERNAL)
    write(internal / "web/enterprise/security.mdx", INTERNAL)
    return {
        "reference": reference,
        "sophon": sophon,
        "public": sophon / "docs/web",
        "internal": internal,
    }


def public_root(paths: dict[str, Path]) -> Path:
    """Fill the overlay root with one valid Enterprise page."""
    write(paths["public"] / "enterprise/security.mdx", PUBLIC)
    return paths["public"]


def config(base: Path, reference: Path, overlay: Path) -> Path:
    path = base / "assemble.yaml"
    path.write_text(
        f"""output: {base / "output"}
roots:
  - name: lancedb
    path: {reference}
    role: reference
  - name: enterprise
    path: {overlay}
    role: overlay
    private: true
"""
    )
    return path


def build(config_path: Path) -> Path:
    """Run every stage `main` runs and return the output directory."""
    loaded = assemble.load_config(config_path)
    resolved = assemble.resolve(loaded)
    docs_json, nav_raw = assemble.assemble_nav(resolved)
    assemble.validate(loaded, resolved, docs_json)
    assemble.merge(resolved)
    assemble.annotate(resolved, docs_json)
    assemble.emit(loaded, resolved, docs_json, nav_raw)
    return loaded.output


@pytest.mark.parametrize("checkout", [True, False])
def test_valid_private_overlay_replaces_the_page(base, checkout):
    paths = site(base, checkout=checkout)
    output = build(config(base, paths["reference"], public_root(paths)))

    assert (output / "enterprise/security.mdx").read_text() == PUBLIC
    assert json.loads((output / "docs.json").read_text()) == NAVIGATION
    assert not (output / "ci.md").exists()
    assert sorted(p.relative_to(output).as_posix() for p in output.rglob("*.*")) == [
        "docs.json",
        "enterprise/security.mdx",
        "static/styles/style.css",
    ]


@pytest.mark.parametrize("checkout", [True, False])
def test_symlinked_private_root_is_refused(base, checkout):
    paths = site(base, checkout=checkout)
    paths["public"].parent.mkdir(parents=True, exist_ok=True)
    paths["public"].symlink_to(paths["internal"], target_is_directory=True)

    with pytest.raises(assemble.AssembleError, match=r"docs/web is a symlink"):
        build(config(base, paths["reference"], paths["public"]))


@pytest.mark.parametrize("checkout", [True, False])
def test_symlinked_parent_of_private_root_is_refused(base, checkout):
    paths = site(base, checkout=checkout)
    (paths["sophon"] / "docs/ci.md").unlink()
    (paths["sophon"] / "docs").rmdir()
    (paths["sophon"] / "docs").symlink_to(paths["internal"], target_is_directory=True)

    with pytest.raises(assemble.AssembleError, match=r"sophon/docs is a symlink"):
        build(config(base, paths["reference"], paths["public"]))


def test_symlinked_checkout_is_refused(base):
    paths = site(base)
    public_root(paths)
    elsewhere = base / "elsewhere"
    paths["sophon"].rename(elsewhere)
    paths["sophon"].symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(assemble.AssembleError, match=r"sophon is a symlink"):
        build(config(base, paths["reference"], paths["public"]))


def test_symlink_above_the_checkout_is_allowed(base):
    real = base / "real"
    paths = site(real)
    public_root(paths)
    (base / "linked").symlink_to(real, target_is_directory=True)
    overlay = base / "linked/sophon/docs/web"

    output = build(config(base, paths["reference"], overlay))

    assert (output / "enterprise/security.mdx").read_text() == PUBLIC


def test_symlink_inside_private_root_is_refused(base):
    paths = site(base)
    page = paths["public"] / "enterprise/security.mdx"
    page.parent.mkdir(parents=True)
    page.symlink_to(paths["internal"] / "enterprise/security.mdx")

    with pytest.raises(assemble.AssembleError, match=r"security.mdx is a symlink"):
        build(config(base, paths["reference"], paths["public"]))


def test_hidden_path_in_private_root_is_refused(base):
    paths = site(base)
    write(public_root(paths) / ".vscode/settings.json", "{}\n")

    with pytest.raises(assemble.AssembleError, match=r"\.vscode is hidden"):
        build(config(base, paths["reference"], paths["public"]))


def test_overlay_navigation_may_only_carry_redirects(base):
    paths = site(base)
    fragment = {
        "insert": [
            {
                "into": ["Documentation"],
                "entry": {"group": "Enterprise", "pages": ["enterprise/security"]},
            }
        ]
    }
    write(public_root(paths) / "docs.nav.json", json.dumps(fragment))

    with pytest.raises(assemble.AssembleError, match=r"may only carry redirects"):
        build(config(base, paths["reference"], paths["public"]))


def test_overlay_redirects_are_merged(base):
    paths = site(base)
    redirect = {"source": "/old", "destination": "/enterprise/security"}
    write(public_root(paths) / "docs.nav.json", json.dumps({"redirects": [redirect]}))

    output = build(config(base, paths["reference"], paths["public"]))

    assert json.loads((output / "docs.json").read_text())["redirects"] == [redirect]


def test_overlay_asset_must_match_the_reference(base):
    paths = site(base)
    write(public_root(paths) / "static/styles/style.css", "body { color: red }\n")

    with pytest.raises(assemble.AssembleError, match=r"style.css differs"):
        build(config(base, paths["reference"], paths["public"]))


def test_identical_overlay_asset_is_accepted(base):
    paths = site(base)
    write(public_root(paths) / "static/styles/style.css", "body {}\n")

    output = build(config(base, paths["reference"], paths["public"]))

    assert (output / "static/styles/style.css").read_text() == "body {}\n"


def test_overlay_without_a_reference_page_is_refused(base):
    paths = site(base)
    write(public_root(paths) / "enterprise/new.mdx", PUBLIC)

    with pytest.raises(assemble.AssembleError, match=r"no reference page"):
        build(config(base, paths["reference"], paths["public"]))


ENTERPRISE_ONLY = """availability:
  oss: unavailable
  enterprise: available
  summary: A deployment checks every request.
"""
BOTH = """availability:
  oss: available
  enterprise: varies
  summary: Local views refresh in full; see [Against a deployment](#deployment).
"""
IMPORTS = "import { Example } from '/snippets/example.mdx';\n"


def page(frontmatter: str = "", body: str = "First paragraph.\n") -> str:
    return f'---\ntitle: "A page"\n{frontmatter}---\n\n{body}'


def availability_site(base: Path, pages: dict[str, str], nav: list[str]) -> Path:
    """A site whose reference root holds `pages`, listed in the order of `nav`."""
    paths = site(base)
    navigation = {
        "tabs": [{"tab": "Documentation", "groups": [{"group": "Pages", "pages": nav}]}]
    }
    write(paths["reference"] / "docs.json", json.dumps({"navigation": navigation}))
    for rel, text in pages.items():
        write(paths["reference"] / rel, text)
    return config(base, paths["reference"], public_root(paths))


def test_availability_renders_a_label_and_a_tag(base):
    body = f"{IMPORTS}\n## Section {{#section}}\n\n<Example />\n"
    source = page(ENTERPRISE_ONLY + "icon: key\n", body)
    output = build(availability_site(base, {"auth.mdx": source}, ["auth"]))

    assert (output / "auth.mdx").read_text() == (
        '---\ntitle: "A page"\nicon: key\ntag: "Enterprise"\n---\n\n'
        f"{IMPORTS}\n"
        '<Badge color="red">Enterprise</Badge> A deployment checks every request.\n\n'
        "## Section {#section}\n\n<Example />\n"
    )


def test_varies_is_labelled_and_untagged(base):
    output = build(availability_site(base, {"views.mdx": page(BOTH)}, ["views"]))

    assert (output / "views.mdx").read_text() == (
        '---\ntitle: "A page"\n---\n\n'
        '<Badge color="green">OSS</Badge> <Badge color="red">Enterprise: varies</Badge> '
        "Local views refresh in full; see [Against a deployment](#deployment).\n\n"
        "First paragraph.\n"
    )


def test_pages_without_availability_are_copied_unchanged(base):
    plain = page(body='<Badge color="red">Enterprise</Badge> Hand-written.\n')
    output = build(availability_site(base, {"plain.mdx": plain}, ["plain"]))

    assert (output / "plain.mdx").read_text() == plain


def test_comparison_lists_declaring_pages_in_navigation_order(base):
    pages = {
        "auth.mdx": page('sidebarTitle: "Auth"\n' + ENTERPRISE_ONLY),
        "views.mdx": page(BOTH),
        "offerings.mdx": page(
            body="Intro.\n\n{/* availability-comparison */}\n\nAfter.\n"
        ),
    }
    output = build(availability_site(base, pages, ["offerings", "views", "auth"]))

    assert (output / "offerings.mdx").read_text() == page(
        body="Intro.\n\n"
        "| Topic | OSS / Enterprise |\n"
        "| --- | --- |\n"
        "| [A page](/views): Local views refresh in full; "
        "see [Against a deployment](/views#deployment). | Yes / Varies |\n"
        "| [Auth](/auth): A deployment checks every request. | No / Yes |\n"
        "\nAfter.\n"
    )


def test_overlay_declaration_replaces_the_reference_one(base):
    paths = site(base)
    write(paths["reference"] / "enterprise/security.mdx", page(BOTH))
    write(
        paths["reference"] / "offerings.mdx",
        page(body="{/* availability-comparison */}\n"),
    )
    write(paths["public"] / "enterprise/security.mdx", page(ENTERPRISE_ONLY, PUBLIC))

    output = build(config(base, paths["reference"], paths["public"]))

    published = (output / "enterprise/security.mdx").read_text()
    assert (
        '<Badge color="red">Enterprise</Badge> A deployment checks every request.'
        in published
    )
    assert f"\n\n{PUBLIC}" in published and "varies" not in published
    assert (
        "A deployment checks every request. | No / Yes |"
        in (output / "offerings.mdx").read_text()
    )


def test_overlay_that_drops_a_declaration_is_reported(base):
    paths = site(base)
    write(paths["reference"] / "enterprise/security.mdx", page(ENTERPRISE_ONLY))
    loaded = assemble.load_config(config(base, paths["reference"], public_root(paths)))
    resolved = assemble.resolve(loaded)
    docs_json, _ = assemble.assemble_nav(resolved)

    warnings = assemble.validate(loaded, resolved, docs_json)

    assert warnings == [
        "enterprise/security.mdx: the enterprise page declares no availability, so "
        "the availability its reference page declares is not published"
    ]


@pytest.mark.parametrize(
    ("frontmatter", "body", "error"),
    [
        (
            ENTERPRISE_ONLY.replace("unavailable", "yes"),
            "Text.\n",
            r"availability.oss is True",
        ),
        (
            ENTERPRISE_ONLY.replace("unavailable", "no"),
            "Text.\n",
            r"availability.oss is False",
        ),
        (
            ENTERPRISE_ONLY.replace("e: available", "e: Available"),
            "Text.\n",
            r"enterprise is 'Available'",
        ),
        (
            ENTERPRISE_ONLY.replace("  summary", "  sumary"),
            "Text.\n",
            r"exactly oss, enterprise and summary",
        ),
        (
            ENTERPRISE_ONLY + "  label: Enterprise only\n",
            "Text.\n",
            r"exactly oss, enterprise",
        ),
        (
            ENTERPRISE_ONLY.replace("enterprise: available", "enterprise: unavailable"),
            "Text.\n",
            r"no offering",
        ),
        (
            ENTERPRISE_ONLY.replace("summary: A", "summary: |\n    One.\n    A"),
            "Text.\n",
            r"one line",
        ),
        (
            ENTERPRISE_ONLY.replace(
                "summary: A deployment checks every request.", "summary: 3"
            ),
            "Text.\n",
            r"one line",
        ),
        ("availability: enterprise only\n", "Text.\n", r"exactly oss, enterprise"),
        (
            ENTERPRISE_ONLY.replace("availability:", '"availability":'),
            "Text.\n",
            r"block of its own",
        ),
        (ENTERPRISE_ONLY + "tag: Enterprise\n", "Text.\n", r"remove `tag`"),
        (
            ENTERPRISE_ONLY,
            '<Badge color="red">Enterprise</Badge> Text.\n',
            r"badge of its own",
        ),
        ("availability: [\n", "Text.\n", r"unreadable frontmatter"),
    ],
)
def test_malformed_availability_is_refused(base, frontmatter, body, error):
    with pytest.raises(assemble.AssembleError, match=error):
        build(availability_site(base, {"auth.mdx": page(frontmatter, body)}, ["auth"]))


@pytest.mark.parametrize(
    ("pages", "error"),
    [
        (
            {
                "a.mdx": page(ENTERPRISE_ONLY),
                "b.mdx": page(body="See {/* availability-comparison */}\n"),
            },
            r"line of its own",
        ),
        (
            {
                "a.mdx": page(ENTERPRISE_ONLY),
                "b.mdx": page(
                    body="{/* availability-comparison */}\n\n{/* availability-comparison */}\n"
                ),
            },
            r"once",
        ),
        (
            {"b.mdx": page(body="{/* availability-comparison */}\n")},
            r"no page declares availability",
        ),
    ],
)
def test_misplaced_comparison_is_refused(base, pages, error):
    with pytest.raises(assemble.AssembleError, match=error):
        build(availability_site(base, pages, sorted(rel[:-4] for rel in pages)))
