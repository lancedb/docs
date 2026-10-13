"""Tests for the assembler's overlay, navigation, private-root, availability,
launch and example stages.

Each test builds a small site in a temporary directory: a reference root that
owns `enterprise/security` and the navigation, and a private overlay root at
`sophon/docs/web` beside an internal folder that must never be published.

Run with `make test-assemble`.
"""

import json
import os
import shutil
import subprocess
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


def config(base: Path, reference: Path, overlay: Path, functions: str | None = None) -> Path:
    path = base / "assemble.yaml"
    launches = f"launches:\n  functions: {functions}\n" if functions else ""
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
{launches}"""
    )
    return path


def build(config_path: Path) -> Path:
    """Run every stage `main` runs and return the output directory."""
    loaded = assemble.load_config(config_path)
    resolved = assemble.resolve(loaded)
    docs_json, nav_raw = assemble.assemble_nav(resolved)
    assemble.validate(loaded, resolved, docs_json)
    assemble.merge(resolved)
    if assemble.gate(loaded, resolved, docs_json):
        nav_raw = None
    assemble.inline_examples(resolved)
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
  enterprise: available
  summary: Local views refresh in full; see [Against a deployment](#deployment).
"""
BOTH_AVAILABLE = """availability:
  oss: available
  enterprise: available
  summary: Works in both offerings.
"""
IMPORTS = "import { Example } from '/snippets/example.mdx';\n"
# Valid MDX whose JavaScript contains blank lines.
MULTILINE_IMPORT = "import {\n  A,\n\n  B\n} from '/snippets/probe.mdx';\n\n<A />\n<B />\n"
MULTILINE_EXPORT = "export const helper = () => {\n\n  return 1;\n};\n\nContent.\n"
# The MDX compiler Mint uses: `<mint>/node_modules/@mdx-js/mdx/index.js`.
MDX_COMPILER = os.environ.get("MDX_COMPILER")


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
        '<Badge color="red">Enterprise</Badge> A deployment checks every request.\n\n'
        f"{IMPORTS}\n"
        "## Section {#section}\n\n<Example />\n"
    )


def render(base: Path, body: str) -> str:
    output = build(
        availability_site(base, {"p.mdx": page(BOTH_AVAILABLE, body)}, ["p"])
    )
    return (output / "p.mdx").read_text()


@pytest.mark.parametrize("body", [MULTILINE_IMPORT, MULTILINE_EXPORT])
def test_label_is_placed_before_any_javascript(base, body):
    assert render(base, body) == (
        '---\ntitle: "A page"\n---\n\n'
        '<Badge color="green">OSS</Badge> <Badge color="red">Enterprise</Badge> '
        f"Works in both offerings.\n\n{body}"
    )


def compile_mdx(text: str) -> subprocess.CompletedProcess:
    script = (
        f"import {{compile}} from {json.dumps(MDX_COMPILER)};\n"
        "let source = '';\n"
        "for await (const chunk of process.stdin) source += chunk;\n"
        "await compile(source);\n"
    )
    body = assemble.split_frontmatter(text)[1]
    return subprocess.run(
        ["node", "--input-type=module", "-e", script],
        input=body,
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(
    not (MDX_COMPILER and shutil.which("node")),
    reason="set MDX_COMPILER to @mdx-js/mdx's index.js and put node on PATH",
)
@pytest.mark.parametrize("body", [MULTILINE_IMPORT, MULTILINE_EXPORT])
def test_labelled_page_still_compiles_as_mdx(base, body):
    source = compile_mdx(page(BOTH_AVAILABLE, body))
    assert source.returncode == 0, source.stderr
    rendered = compile_mdx(render(base, body))
    assert rendered.returncode == 0, rendered.stderr


@pytest.mark.parametrize("declaration", [ENTERPRISE_ONLY, BOTH, BOTH_AVAILABLE])
def test_hand_set_tag_is_refused_whatever_the_declaration(base, declaration):
    source = page(declaration + "tag: Enterprise\n")
    with pytest.raises(assemble.AssembleError, match=r"remove `tag`"):
        build(availability_site(base, {"p.mdx": source}, ["p"]))


def test_supported_feature_with_constraints_is_labelled_and_untagged(base):
    output = build(availability_site(base, {"views.mdx": page(BOTH)}, ["views"]))

    assert (output / "views.mdx").read_text() == (
        '---\ntitle: "A page"\n---\n\n'
        '<Badge color="green">OSS</Badge> <Badge color="red">Enterprise</Badge> '
        "Local views refresh in full; see [Against a deployment](#deployment).\n\n"
        "First paragraph.\n"
    )


@pytest.mark.parametrize("offering", ["oss", "enterprise"])
@pytest.mark.parametrize("status", ["varies", "unknown", "coming-soon"])
def test_nonbinary_support_is_refused_without_coercion(base, offering, status):
    declaration = BOTH_AVAILABLE.replace(
        f"{offering}: available", f"{offering}: {status}"
    )
    error = (
        rf"views.mdx: availability.{offering}.*"
        r"not one of available, unavailable"
    )
    with pytest.raises(assemble.AssembleError, match=error):
        build(availability_site(base, {"views.mdx": page(declaration)}, ["views"]))


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
        "### Pages {#availability-pages}\n\n"
        "| Topic | OSS / Enterprise |\n"
        "| --- | --- |\n"
        "| [A page](/views): Local views refresh in full; "
        "see [Against a deployment](/views#deployment). | Yes / Yes |\n"
        "| [Auth](/auth): A deployment checks every request. | No / Yes |\n"
        "\nAfter.\n"
    )


def test_comparison_has_a_table_per_navigation_section(base):
    paths = site(base)
    navigation = {
        "tabs": [
            {
                "tab": "Documentation",
                "groups": [
                    {
                        "group": "Search",
                        "pages": [
                            "search/index",
                            {"group": "Nested", "pages": ["search/deep"]},
                        ],
                    },
                    {"group": "Security", "pages": ["auth"]},
                ],
            },
            {
                "tab": "Integrations",
                "pages": ["hf", {"group": "Search", "pages": ["search/more"]}],
            },
        ]
    }
    write(paths["reference"] / "docs.json", json.dumps({"navigation": navigation}))
    pages = {
        "search/index.mdx": page('sidebarTitle: "Overview"\n' + BOTH_AVAILABLE),
        "search/deep.mdx": page(BOTH),
        "search/more.mdx": page(BOTH_AVAILABLE),
        "auth.mdx": page(ENTERPRISE_ONLY),
        "hf.mdx": page(BOTH_AVAILABLE),
        "unlisted.mdx": page(ENTERPRISE_ONLY),
        "offerings.mdx": page(body="{/* availability-comparison */}\n"),
    }
    for rel, text in pages.items():
        write(paths["reference"] / rel, text)

    output = build(config(base, paths["reference"], public_root(paths)))

    header = "| Topic | OSS / Enterprise |\n| --- | --- |\n"
    both = "Works in both offerings. | Yes / Yes |\n"
    assert (output / "offerings.mdx").read_text() == page(
        body="### Search {#availability-search}\n\n" + header
        + f"| [Overview](/search): {both}"
        "| [A page](/search/deep): Local views refresh in full; "
        "see [Against a deployment](/search/deep#deployment). | Yes / Yes |\n"
        f"| [A page](/search/more): {both}"
        "\n### Security {#availability-security}\n\n" + header
        + "| [A page](/auth): A deployment checks every request. | No / Yes |\n"
        "\n### Integrations {#availability-integrations}\n\n" + header
        + f"| [A page](/hf): {both}"
        "\n### Other pages {#availability-other-pages}\n\n" + header
        + "| [A page](/unlisted): A deployment checks every request. | No / Yes |\n"
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


# --------------------------------------------------------------------------- #
# launches
# --------------------------------------------------------------------------- #

SHARED = """Tables hold rows.

{/* if functions launched */}
A [function](/management/function) fills a column.
{/* end if */}

{/* if functions not launched */}
Columns are filled by SQL.
{/* end if */}

| Object | What |
| --- | --- |
| Table | rows |
| [Function](/management/function) {/* if functions launched */} | code |

End.
"""
FUNCTION_BODY = """import { PyRegister } from '/snippets/function.mdx';

<CodeBlock filename="Python" language="Python" icon="python">
{PyRegister}
</CodeBlock>
"""
FUNCTION_SNIPPET = 'export const PyRegister = "db.create_function(double)\\n";\n'
FUNCTION_REDIRECT = {"source": "/old-functions", "destination": "/management/function#apply"}
TABLE_REDIRECT = {"source": "/old-tables", "destination": "/management/table"}


def launch_site(
    base: Path, functions: str, shared: str = SHARED, key: str = "launch: functions"
) -> Path:
    paths = site(base)
    navigation = {
        "tabs": [
            {
                "tab": "Documentation",
                "groups": [
                    {"group": "Start", "pages": ["offerings"]},
                    {
                        "group": "Management",
                        "pages": ["management/table", "management/function"],
                    },
                    {"group": "SQL", "pages": ["sql/function"]},
                ],
            }
        ]
    }
    write(
        paths["reference"] / "docs.json",
        json.dumps({"navigation": navigation, "redirects": [FUNCTION_REDIRECT, TABLE_REDIRECT]}),
    )
    pages = {
        "offerings.mdx": page(body="{/* availability-comparison */}\n"),
        "management/table.mdx": page(BOTH_AVAILABLE, shared),
        "management/function.mdx": page(
            f'sidebarTitle: "Function"\n{key}\n' + ENTERPRISE_ONLY, FUNCTION_BODY
        ),
        "sql/function.mdx": page(f"{key}\n", "CREATE FUNCTION.\n"),
        "snippets/function.mdx": FUNCTION_SNIPPET,
    }
    for rel, text in pages.items():
        write(paths["reference"] / rel, text)
    return config(base, paths["reference"], public_root(paths), functions)


def published_files(output: Path) -> list[str]:
    return sorted(p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file())


def test_unlaunched_feature_is_left_out_of_every_output(base):
    output = build(launch_site(base, "unlaunched"))

    assert published_files(output) == [
        "docs.json",
        "enterprise/security.mdx",
        "management/table.mdx",
        "offerings.mdx",
        "static/styles/style.css",
    ]
    docs = json.loads((output / "docs.json").read_text())
    assert docs["navigation"]["tabs"][0]["groups"] == [
        {"group": "Start", "pages": ["offerings"]},
        {"group": "Management", "pages": ["management/table"]},
    ]
    assert docs["redirects"] == [TABLE_REDIRECT]
    table = (output / "management/table.mdx").read_text()
    assert table.split("Works in both offerings.\n\n", 1)[1] == (
        "Tables hold rows.\n\n"
        "Columns are filled by SQL.\n\n"
        "| Object | What |\n| --- | --- |\n| Table | rows |\n\n"
        "End.\n"
    )
    assert "unction" not in (output / "offerings.mdx").read_text()


def test_launched_feature_is_published_with_its_sections(base):
    output = build(launch_site(base, "launched"))

    assert "management/function.mdx" in published_files(output)
    assert "sql/function.mdx" in published_files(output)
    assert "snippets/function.mdx" not in published_files(output)
    docs = json.loads((output / "docs.json").read_text())
    assert [g["group"] for g in docs["navigation"]["tabs"][0]["groups"]] == [
        "Start", "Management", "SQL"
    ]
    assert docs["redirects"] == [FUNCTION_REDIRECT, TABLE_REDIRECT]
    table = (output / "management/table.mdx").read_text()
    assert table.split("Works in both offerings.\n\n", 1)[1] == (
        "Tables hold rows.\n\n"
        "A [function](/management/function) fills a column.\n\n"
        "| Object | What |\n| --- | --- |\n| Table | rows |\n"
        "| [Function](/management/function) | code |\n\n"
        "End.\n"
    )
    function = (output / "management/function.mdx").read_text()
    assert "launch:" not in function
    assert function.startswith(
        '---\ntitle: "A page"\nsidebarTitle: "Function"\ntag: "Enterprise"\n---\n'
    )
    assert function.endswith(
        '```python Python icon="python"\ndb.create_function(double)\n```\n'
    )
    assert "[Function](/management/function): A deployment checks every request." in (
        output / "offerings.mdx"
    ).read_text()


def test_the_unlaunched_build_is_the_default_without_a_launch_listed(base):
    with pytest.raises(assemble.AssembleError, match=r"no launch named 'functions'"):
        build(launch_site(base, ""))


@pytest.mark.parametrize(
    ("shared", "error"),
    [
        (SHARED + "\nSee [Functions](/management/function#apply).\n", r"table.mdx links to /management/function"),
        (SHARED + '\n<Card href="/sql/function" />\n', r"links to /sql/function"),
        (SHARED + "\n{/* if views launched */}\nViews.\n{/* end if */}\n", r"no launch named 'views'"),
        (SHARED + "\n{/* if functions launched */}\nOpen.\n", r"never closed"),
        (SHARED + "\n{/* end if */}\n", r"closes nothing"),
        (
            "{/* if functions launched */}\n{/* if functions not launched */}\n"
            "{/* end if */}\n{/* end if */}\n",
            r"cannot nest",
        ),
        (SHARED + "\n{/* if Functions launched */}\nX\n{/* end if */}\n", r"cannot read the launch marker"),
        (SHARED + "\n{/* endif */}\n", r"cannot read the launch marker"),
        (
            SHARED + "\n| a {/* if functions launched */} | b {/* if functions launched */} |\n",
            r"one launch marker per line",
        ),
        (SHARED + "\n{/* if functions launched */} {/* end if */}\n", r"alone on its line"),
    ],
)
def test_launch_markers_are_checked(base, shared, error):
    with pytest.raises(assemble.AssembleError, match=error):
        build(launch_site(base, "unlaunched", shared))


def published_tree(output: Path) -> dict[str, bytes]:
    return {rel: (output / rel).read_bytes() for rel in published_files(output)}


@pytest.mark.parametrize("functions", ["unlaunched", "launched"])
@pytest.mark.parametrize(
    "key",
    [
        '"launch": functions',
        "launch : functions",
        "'launch': functions",
        'launch: "functions"',
        "launch: functions  # until the launch",
        "launch:\n  functions",
    ],
)
def test_any_yaml_spelling_of_the_launch_key_gates_the_page(base, functions, key):
    expected = build(launch_site(base / "canonical", functions))
    output = build(launch_site(base / "spelled", functions, key=key))

    assert published_tree(output) == published_tree(expected)
    assert ("management/function.mdx" in published_files(output)) == (functions == "launched")


@pytest.mark.parametrize("functions", ["unlaunched", "launched"])
@pytest.mark.parametrize(
    ("key", "error"),
    [
        ("Launch: functions", r"function.mdx: frontmatter `Launch` is not `launch`"),
        ("launches: functions", r"frontmatter `launches` is not `launch`"),
        ("? launch\n: functions", r"write `launch: functions` on a line of its own"),
        ('"\\x6caunch": functions', r"write `launch: functions` on a line of its own"),
        ("launch: functions\nlaunch: functions", r"on a line of its own"),
        ("launch: [functions]", r"`launch` must name one feature"),
    ],
)
def test_a_launch_key_the_gate_cannot_remove_stops_the_build(base, functions, key, error):
    with pytest.raises(assemble.AssembleError, match=error):
        build(launch_site(base, functions, key=key))


UNANNOUNCED = "| Unannounced | code |"
UNREADABLE = r"table.mdx:\d+: cannot read the launch marker"


@pytest.mark.parametrize("functions", ["unlaunched", "launched"])
@pytest.mark.parametrize(
    ("marked", "error"),
    [
        (UNANNOUNCED + " {/* if functions launced */}", UNREADABLE),
        ("{/* if functions launced */}\n" + UNANNOUNCED + "\n{/* end if */}", UNREADABLE),
        ("{/* If functions launched */}\n" + UNANNOUNCED + "\n{/* end if */}", UNREADABLE),
        ("{/* if functions unlaunched */}\n" + UNANNOUNCED + "\n{/* end if */}", UNREADABLE),
        ("{/* if not functions launched */}\n" + UNANNOUNCED + "\n{/* end if */}", UNREADABLE),
        ("{/* iff functions launched */}\n" + UNANNOUNCED + "\n{/* end if */}", UNREADABLE),
        ("{ /* if functions launched */ }\n" + UNANNOUNCED + "\n{/* end if */}", UNREADABLE),
        ("{/** if functions launched **/}\n" + UNANNOUNCED + "\n{/* end if */}", UNREADABLE),
        (
            "{/*\n  if functions launched\n*/}\n" + UNANNOUNCED + "\n{/*\n  end if\n*/}",
            UNREADABLE,
        ),
        (UNANNOUNCED + " {/* if functions launched */", UNREADABLE),
        ("{/* if functions launched */}\n" + UNANNOUNCED + "\n{/* end-if */}", UNREADABLE),
        ("{/* if functions launched */}\n" + UNANNOUNCED + "\n{/* End if */}", UNREADABLE),
        (
            "{/* if function launched */}\n" + UNANNOUNCED + "\n{/* end if */}",
            r"table.mdx: no launch named 'function'",
        ),
    ],
)
def test_a_misspelled_launch_marker_stops_the_build(base, functions, marked, error):
    with pytest.raises(assemble.AssembleError, match=error):
        build(launch_site(base, functions, SHARED + "\n" + marked + "\n"))


@pytest.mark.parametrize("functions", ["unlaunched", "launched"])
def test_other_comments_are_published_unchanged(base, functions):
    comments = (
        "{/* TODO: tidy this table */}\n"
        "{/* HF_SYNC:START — generated by `make hf-sync` */}\n"
        "{/* [Evaluating](./examples/eval) */}\n"
        "{/*\n<div>notes</div>\n*/}\n"
        "{/* endpoint notes, if any */}\n"
    )
    output = build(launch_site(base, functions, SHARED + "\n" + comments))

    assert (output / "management/table.mdx").read_text().endswith("End.\n\n" + comments)


def test_launched_pages_may_link_to_each_other(base):
    shared = SHARED + "\n{/* if functions launched */}\nSee [SQL](/sql/function).\n{/* end if */}\n"
    output = build(launch_site(base, "launched", shared))

    assert "See [SQL](/sql/function)." in (output / "management/table.mdx").read_text()


def test_a_page_launch_must_be_known(base):
    paths = site(base)
    write(paths["reference"] / "p.mdx", page("launch: views\n"))
    with pytest.raises(assemble.AssembleError, match=r"p.mdx: no launch named 'views'"):
        build(config(base, paths["reference"], public_root(paths), "unlaunched"))


@pytest.mark.parametrize(
    ("launches", "error"),
    [
        ("launches:\n  functions: soon\n", r"launches.functions is 'soon'"),
        ("launches:\n  Functions: launched\n", r"'Functions' is not a feature name"),
        ("launches: [functions]\n", r"must map each feature"),
    ],
)
def test_malformed_launches_are_refused(base, launches, error):
    paths = site(base)
    path = config(base, paths["reference"], public_root(paths))
    path.write_text(path.read_text() + launches)
    with pytest.raises(assemble.AssembleError, match=error):
        assemble.load_config(path)


def test_launch_state_can_be_previewed_from_the_environment(base, monkeypatch):
    paths = site(base)
    path = config(base, paths["reference"], public_root(paths), "${DOCS_FUNCTIONS:-unlaunched}")
    assert assemble.load_config(path).launches == {"functions": "unlaunched"}
    monkeypatch.setenv("DOCS_FUNCTIONS", "launched")
    assert assemble.load_config(path).launches == {"functions": "launched"}
    assert assemble.publication(assemble.load_config(path)) == "functions=launched"


# --------------------------------------------------------------------------- #
# examples
# --------------------------------------------------------------------------- #

SNIPPETS = (
    'export const PyConnect = "import lancedb\\n\\ndb = lancedb.connect(\\"data\\")\\n";\n\n'
    'export const PySearch = "table.search([1.0])\\n";\n\n'
    'export const PyFence = "print(\\"```\\")\\n";\n\n'
    'export const RsConnect = "let db = connect(uri);\\n";\n'
)


def example_site(base: Path, body: str, snippets: str = SNIPPETS) -> Path:
    paths = site(base)
    write(paths["reference"] / "snippets/examples.mdx", snippets)
    write(paths["reference"] / "p.mdx", page(body=body))
    return build(config(base, paths["reference"], public_root(paths)))


def test_examples_are_published_as_fenced_code_where_they_are_shown(base):
    body = (
        "import {\n  PyConnect,\n  PySearch as Search,\n  RsConnect,\n} from '/snippets/examples.mdx';\n\n"
        "<CodeGroup>\n"
        '    <CodeBlock filename="Python (sync)" language="Python" icon="python">\n'
        "    {PyConnect + Search}\n"
        "    </CodeBlock>\n\n"
        '    <CodeBlock filename="Rust" language="Rust" icon="rust">\n'
        '    { "use lancedb::connect;\\n" }\n'
        '    { "\\n" }\n'
        "    {RsConnect}\n"
        "    </CodeBlock>\n"
        "</CodeGroup>\n"
    )
    output = example_site(base, body)

    assert (output / "p.mdx").read_text() == page(
        body="\n<CodeGroup>\n"
        '    ```python Python (sync) icon="python"\n'
        "    import lancedb\n\n"
        '    db = lancedb.connect("data")\n'
        "    table.search([1.0])\n"
        "    ```\n\n"
        '    ```rust Rust icon="rust"\n'
        "    use lancedb::connect;\n\n"
        "    let db = connect(uri);\n"
        "    ```\n"
        "</CodeGroup>\n"
    )
    assert not (output / "snippets/examples.mdx").exists()


def test_a_fence_is_longer_than_any_backticks_in_the_code(base):
    body = (
        "import { PyFence } from '/snippets/examples.mdx';\n\n"
        '<CodeBlock filename="Python" language="Python" icon="python">\n{PyFence}\n</CodeBlock>\n'
    )
    text = (example_site(base, body) / "p.mdx").read_text()
    assert '````python Python icon="python"\nprint("```")\n````\n' in text


def test_unused_imports_go_and_used_ones_stay(base):
    body = (
        "import { PyConnect, PySearch } from '/snippets/examples.mdx';\n"
        "import { Widget } from '/snippets/widgets.mdx';\n\n"
        '<CodeBlock filename="Python" language="Python" icon="python">\n{PyConnect}\n</CodeBlock>\n\n'
        "<Widget />\n"
    )
    paths = site(base)
    write(paths["reference"] / "snippets/examples.mdx", SNIPPETS)
    write(paths["reference"] / "snippets/widgets.mdx", "export const Widget = () => null;\n")
    write(paths["reference"] / "p.mdx", page(body=body))
    output = build(config(base, paths["reference"], public_root(paths)))

    text = (output / "p.mdx").read_text()
    assert "examples.mdx" not in text
    assert "import { Widget } from '/snippets/widgets.mdx';\n" in text
    assert (output / "snippets/widgets.mdx").exists()
    assert not (output / "snippets/examples.mdx").exists()


def test_a_code_block_of_plain_text_is_left_alone(base):
    body = '<CodeBlock filename="bash" language="bash" icon="terminal">\npip install lancedb\n</CodeBlock>\n'
    assert (example_site(base, body) / "p.mdx").read_text() == page(body=body)


@pytest.mark.parametrize(
    ("body", "error"),
    [
        (
            '<CodeBlock filename="Python" language="Python" icon="python">\n{Missing}\n</CodeBlock>\n',
            r"\{Missing\} is not imported",
        ),
        (
            "import { PyGone } from '/snippets/examples.mdx';\n\n"
            '<CodeBlock filename="Python" language="Python" icon="python">\n{PyGone}\n</CodeBlock>\n',
            r"exports no PyGone",
        ),
        (
            "import { PyConnect } from '/snippets/nowhere.mdx';\n\n"
            '<CodeBlock filename="Python" language="Python" icon="python">\n{PyConnect}\n</CodeBlock>\n',
            r"imports /snippets/nowhere.mdx, which no root provides",
        ),
        (
            "import { PyConnect } from '/snippets/examples.mdx';\n\n"
            '<CodeBlock filename="Python" language="Python" icon="python">\n{PyConnect}\nprint(1)\n</CodeBlock>\n',
            r"mixes examples with other text",
        ),
        (
            "import { PyConnect } from '/snippets/examples.mdx';\n\n"
            '<CodeBlock filename="Python" language="Python" lines>\n{PyConnect}\n</CodeBlock>\n',
            r"cannot carry",
        ),
    ],
)
def test_unreadable_examples_are_refused(base, body, error):
    with pytest.raises(assemble.AssembleError, match=error):
        example_site(base, body)


@pytest.mark.skipif(
    not (MDX_COMPILER and shutil.which("node")),
    reason="set MDX_COMPILER to @mdx-js/mdx's index.js and put node on PATH",
)
def test_inlined_examples_compile_as_mdx(base):
    body = (
        "import { PyConnect, PyFence } from '/snippets/examples.mdx';\n\n"
        "<Accordion title=\"Records\">\n"
        "    <CodeGroup>\n"
        '        <CodeBlock filename="Python (sync)" language="Python" icon="python">\n'
        "        {PyConnect}\n"
        "        </CodeBlock>\n"
        '        <CodeBlock filename="Python" language="Python" icon="python">\n'
        "        {PyFence}\n"
        "        </CodeBlock>\n"
        "    </CodeGroup>\n"
        "</Accordion>\n"
    )
    result = compile_mdx((example_site(base, body) / "p.mdx").read_text())
    assert result.returncode == 0, result.stderr
