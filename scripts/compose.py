"""
Compose the saved editions into one Mintlify site with product and version
selectors.

Each edition in editions/<product>/<version>/ is published as it was saved,
under /<product>/v<major>-<minor>-<patch>/: its pages, assets, navigation,
redirects and drafts, and an "About this version" page made from its release
record. Links in its pages are pointed into the edition; code is left as
written. compose.yaml orders the products and names each one's preferred
edition. Readers land on the first product's preferred edition, and
unversioned URLs redirect to its matching pages. Nothing is rebuilt from
product sources. See "Composing the site" in README.md.

Usage:

    python scripts/compose.py [--output DIR]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import posixpath
import re
import shutil
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import assemble  # noqa: E402
import candidate  # noqa: E402
import edition  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = "compose.yaml"
OUTPUT = REPO_ROOT / "build" / "versioned"
NAV = assemble.NAV_BASE
IGNORE = ".mintignore"
ABOUT = "about-this-version"
PAGE_SUFFIXES = assemble.PAGE_SUFFIXES
# Mintlify loads every stylesheet and script on every page, so the site takes
# these from its default edition only.
SITE_WIDE = (".css", ".js")
# Mint ignores these beside .mintignore (DEFAULT_MINT_IGNORES, Mint 4.2.888).
MINT_IGNORES = (
    ".git", ".github", ".claude", ".agents", ".idea", ".vscode", ".cursor",
    ".mintlify/**", "!.mintlify/**/", "!.mintlify/skills/**", "!.mintlify/**/*.json",
    "!.mintlify/**/*.yaml", "!.mintlify/**/*.yml",
    "node_modules", "README.md", "LICENSE.md", "CHANGELOG.md", "CONTRIBUTING.md",
)
# Markdown files Mint never publishes as pages, in any folder.
NOT_PAGES = frozenset({"readme", "license", "contributing", "contribute"})
FRONTMATTER_LINKS = ("icon", "og:image", "twitter:image")
LINK_ATTRIBUTES = ("href", "src", "icon", "img", "image", "poster")
# Values of these name an icon unless they are a path or a URL.
NAMES_OR_LINKS = ("icon",)
VERSIONED_RE = re.compile(rf"/(?:{'|'.join(edition.PRODUCTS)})/v[0-9]+-[0-9]+-[0-9]+(?=[/?#]|$)")
SCHEME_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")
PARAM_RE = re.compile(r":([A-Za-z_][A-Za-z0-9_]*)([*+?]?)")
FENCE_RE = re.compile(r"[ \t]*(`{3,}|~{3,})")
INLINE_LINK_RE = re.compile(r"(\]\([ \t]*)(<[^<>\n]*>|[^\s()<>]+)")
DEFINITION_RE = re.compile(r"( {0,3}\[[^\]\n]+\]:[ \t]*)(<[^<>\n]*>|\S+)")
ATTRIBUTE_RE = re.compile(
    rf"""((?<![\w-])({'|'.join(LINK_ATTRIBUTES)})[ \t]*=[ \t]*\{{?[ \t]*)(["'`])([^"'`\n]*)\3"""
)
ANY_ATTRIBUTE_RE = re.compile(r"""[\w:-]+[ \t]*=[ \t]*\{?[ \t]*(["'`])[^"'`\n]*\1""")
LINK_TEXT_RE = re.compile(r"\[[^\]\n]*\]")
IMPORT_RE = re.compile(r"""([ \t]*(?:import\b[^'"\n]*?\bfrom|\}[ \t]*from)[ \t]*)(["'])(/[^'"\n]+)\2""")
FRONTMATTER_RE = re.compile(
    rf"""((?:{'|'.join(map(re.escape, FRONTMATTER_LINKS))})[ \t]*:[ \t]*)(["']?)([^"'\s]+)\2([ \t]*)"""
)


class ComposeError(Exception):
    """Inputs that must stop composition before anything is written."""


# .mintignore


@dataclass(frozen=True)
class Rule:
    regex: re.Pattern
    negated: bool
    directory: bool


def glob_regex(glob: str) -> str:
    """A gitignore glob as a regular expression, as the `ignore` package Mint uses reads it."""
    out, i = "", 0
    while i < len(glob):
        at_segment = i == 0 or glob[i - 1] == "/"
        if at_segment and glob.startswith("**/", i):
            out, i = out + "(?:.*/)?", i + 3
        elif at_segment and glob.startswith("**", i) and i + 2 == len(glob):
            out, i = out + ".*", i + 2
        elif glob[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif glob[i] == "?":
            out, i = out + "[^/]", i + 1
        elif glob[i] == "\\" and i + 1 < len(glob):
            out, i = out + re.escape(glob[i + 1]), i + 2
        elif glob[i] == "[" and (end := glob.find("]", i + 2)) > 0:
            body = glob[i + 1 : end]
            negated = body.startswith(("!", "^"))
            body = body[1:] if negated else body
            out += "[" + ("^" if negated else "") + body.replace("\\", "\\\\") + "]"
            i = end + 1
        else:
            out, i = out + re.escape(glob[i]), i + 1
    return out


def ignore_rules(lines) -> list[Rule]:
    rules = []
    for line in lines:
        # Mint trims every line before matching.
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        line = line[1:] if negated else line
        directory = line.endswith("/")
        line = line.rstrip("/")
        if not line:
            continue
        body = glob_regex(line.lstrip("/"))
        rules.append(Rule(re.compile(body if "/" in line else f"(?:.*/)?{body}"), negated, directory))
    return rules


def ignored(rules: list[Rule], path: str) -> bool:
    """Whether gitignore rules leave out `path`, relative to the site."""
    parts = path.split("/")
    for depth in range(1, len(parts) + 1):
        current, is_dir = "/".join(parts[:depth]), depth < len(parts)
        verdict = False
        for rule in rules:
            if (is_dir or not rule.directory) and rule.regex.fullmatch(current):
                verdict = not rule.negated
        # Nothing in an ignored directory can be included again.
        if verdict:
            return True
    return False


def pointed_ignore(line: str, prefix: str) -> str:
    """An edition's .mintignore line, for the same files under its prefix."""
    text = line.strip()
    if not text or text.startswith("#"):
        return text
    negated = text.startswith("!")
    body = text[1:] if negated else text
    directory = body.endswith("/")
    body = body.rstrip("/")
    if not body:
        return ""
    # Without a slash a pattern matches at any depth below the edition.
    body = f"/{prefix}/{body.lstrip('/')}" if "/" in body else f"/{prefix}/**/{body}"
    return ("!" if negated else "") + body + ("/" if directory else "")


# redirects


@dataclass(frozen=True)
class Redirect:
    source: str
    # A site path with a leading slash, or an external URL.
    destination: str
    permanent: bool | None
    pattern: re.Pattern

    @property
    def wildcard(self) -> bool:
        return PARAM_RE.search(self.source) is not None

    @property
    def external(self) -> bool:
        return SCHEME_RE.match(self.destination) is not None

    def target(self, path: str) -> str:
        """Where this redirect sends `path`, which its source matches."""
        found = self.pattern.fullmatch(path)
        values = found.groupdict() if found else {}
        target = PARAM_RE.sub(lambda m: values.get(m[1]) or "", self.destination)
        if SCHEME_RE.match(target):
            return target
        route, mark, anchor = target.partition("#")
        return normal(re.sub(r"/+", "/", route)) + mark + anchor


def redirect_pattern(source: str) -> re.Pattern:
    regex = ""
    for segment in [s for s in source.split("/") if s]:
        param = PARAM_RE.fullmatch(segment)
        if param is None:
            regex += "/" + re.escape(segment)
        elif param[2] == "*":
            regex += rf"(?:/(?P<{param[1]}>.*))?"
        elif param[2] == "+":
            regex += rf"/(?P<{param[1]}>.+)"
        elif param[2] == "?":
            regex += rf"(?:/(?P<{param[1]}>[^/]+))?"
        else:
            regex += rf"/(?P<{param[1]}>[^/]+)"
    try:
        return re.compile(regex or "/")
    except re.error as exc:
        raise ComposeError(f"the redirect source {source!r} cannot be read: {exc}") from None


def redirect(item: object, where: str) -> Redirect:
    if (
        not isinstance(item, dict)
        or not isinstance(item.get("source"), str)
        or not isinstance(item.get("destination"), str)
        or not item["source"].startswith("/")
    ):
        raise ComposeError(f"{where}: a redirect needs a source path and a destination, not {item!r}")
    destination = item["destination"]
    # Some sources write `a/b` for `/a/b`.
    if not SCHEME_RE.match(destination) and not destination.startswith("/"):
        destination = "/" + destination
    source = normal(item["source"])
    return Redirect(source, destination, item.get("permanent"), redirect_pattern(source))


def entry(source: str, destination: str, permanent: bool | None) -> dict:
    return {"source": source, "destination": destination,
            **({} if permanent is None else {"permanent": permanent})}


def normal(path: str) -> str:
    return path.rstrip("/") or "/"


def versioned(prefix: str, path: str) -> str:
    """A site path of an edition, at its place in the composed site."""
    return f"/{prefix}" + ("" if path in ("/", "") else path)


# editions


@dataclass
class Edition:
    record: dict
    prefix: str
    docs: dict
    # Its .mintignore, pointed at its prefix.
    ignore: list[str]
    # The files copied under its prefix, and whether each of its pages is published.
    files: list[str]
    pages: dict[str, bool]
    # What it publishes, as site paths: page routes, folders and files.
    routes: set[str]
    paths: set[str]
    assets: set[str]
    redirects: list[Redirect]
    site_wide: list[str]

    @property
    def name(self) -> str:
        return f"{self.record['product']} {self.record['version']}"

    @property
    def version(self) -> tuple[int, ...]:
        return tuple(int(part) for part in self.record["version"].split("."))

    def match(self, path: str) -> Redirect | None:
        return next((r for r in self.redirects if r.pattern.fullmatch(path)), None)

    def publishes(self, path: str) -> bool:
        """Whether a site path names a page, folder or file of this edition, or one of its redirects."""
        path = normal(urllib.parse.unquote(path))
        if path in self.paths or (path.endswith(PAGE_SUFFIXES) and path.rsplit(".", 1)[0] in self.routes):
            return True
        return self.match(path) is not None

    def landing(self, path: str) -> tuple[str, str] | None:
        """Where a reader of `path` lands, following redirects, and the anchor they add."""
        anchor, seen, path = "", set(), normal(path)
        while path not in self.paths:
            found = self.match(path)
            if found is None or path in seen:
                return None
            seen.add(path)
            target = found.target(path)
            if SCHEME_RE.match(target):
                return None
            path, _, added = target.partition("#")
            anchor = added or anchor
        return path, anchor


def read_edition(repo: Path, record: dict) -> Edition:
    rel = record["edition"]
    root = repo / rel
    prefix = f"{record['product']}/v{record['version'].replace('.', '-')}"
    # Exactly the files its checksum covers.
    saved = sorted(edition.saved_hashes(repo, rel))
    try:
        docs = json.loads((root / NAV).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ComposeError(f"{rel}/{NAV}: {exc}") from None
    if not isinstance(docs, dict) or not isinstance(docs.get("navigation"), dict):
        raise ComposeError(f"{rel}/{NAV} has no navigation")
    text = (root / IGNORE).read_text(encoding="utf-8") if IGNORE in saved else ""
    ignore = [pointed_ignore(line, prefix) for line in text.splitlines()]
    rules = ignore_rules([*MINT_IGNORES, *ignore])
    kept = [p for p in saved if p not in (NAV, IGNORE)]
    published = {p for p in kept if not ignored(rules, f"{prefix}/{p}")}
    files = [p for p in kept if not p.endswith(SITE_WIDE)]
    pages = {
        p: p in published
        for p in files
        if p.endswith(PAGE_SUFFIXES)
        # Snippet modules are code that pages import.
        and not p.startswith("snippets/")
        and not (p.endswith(".md") and PurePosixPath(p).stem.lower() in NOT_PAGES)
    }
    routes = {assemble.page_route(p) for p, shown in pages.items() if shown}
    # A link may also name a folder's index page as `folder/index`.
    stems = {"/" + p.rsplit(".", 1)[0] for p, shown in pages.items() if shown}
    folders = {"/".join(r.split("/")[:depth]) for r in routes for depth in range(2, r.count("/") + 1)}
    assets = {f"/{p}" for p in published if p in files and p not in pages}
    redirects = [redirect(item, f"{rel}/{NAV}") for item in docs.get("redirects") or []]
    site_wide = [p for p in kept if p.endswith(SITE_WIDE) and p in published]
    return Edition(record, prefix, docs, ignore, files, pages, routes, routes | stems | folders | assets,
                   assets, redirects, site_wide)


# pages


def blank(match: re.Match) -> str:
    return " " * len(match[0])


def monospace(text: str) -> str:
    """`text` as inline code."""
    fence = "`" * (max((len(run) for run in re.findall(r"`+", text)), default=0) + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def plain(text: str) -> str:
    """`text` shown as itself in MDX."""
    return re.sub(r"([\\`*_{}\[\]<>|~#!&])", r"\\\1", text.strip())


class Linker:
    """Points the links in one edition's pages into the edition."""

    def __init__(self, ed: Edition, host: str, problems: list[str], warnings: list[str]):
        self.ed, self.host = ed, host
        self.problems, self.warnings = problems, warnings
        # Where code uses a URL on this site, and the URL's path.
        self.code_urls: list[tuple[str, str]] = []
        self.page, self.line, self.quiet = "", 0, False
        self.site_url = re.compile(
            rf"(?i)(?<![\w.-])https?://{re.escape(host)}(?![\w.-])([^\s\"'`<>()\[\]{{}}|\\]*)"
        )

    def where(self) -> str:
        return f"{self.ed.record['edition']}/{self.page}:{self.line}"

    def fail(self, message: str) -> None:
        if not self.quiet:
            self.problems.append(f"{self.where()}: {message}")

    def note(self, message: str) -> None:
        if not self.quiet:
            self.warnings.append(f"{self.where()}: {message}")

    def point(self, raw: str) -> str:
        """A link target, pointed into this edition."""
        bracketed = raw.startswith("<") and raw.endswith(">")
        target = raw[1:-1] if bracketed else raw
        if not target or target.startswith("#"):
            return raw
        parts = urllib.parse.urlsplit(target)
        ed = self.ed
        if parts.scheme or target.startswith("//"):
            if parts.scheme.lower() not in ("", "http", "https") or (parts.hostname or "") != self.host:
                return raw
            landed = ed.landing(parts.path or "/")
            if landed is None:
                self.note(f"{target} has no page in {ed.name}; it now points at that path in {ed.name}")
                landed = (normal(parts.path or "/"), "")
            path, anchor = landed
            # The link's own anchor wins over one a redirect adds.
            anchor = parts.fragment or anchor
            pointed = versioned(ed.prefix, path) + (f"?{parts.query}" if parts.query else "") + (
                f"#{anchor}" if anchor else "")
        elif target.startswith("/"):
            found = VERSIONED_RE.match(parts.path)
            if found:
                if found[0] != f"/{ed.prefix}":
                    self.fail(f"{target} points into another edition; an edition links only within itself")
                return raw
            if not ed.publishes(parts.path):
                self.note(f"{target} leads to nothing {ed.name} publishes")
            pointed = versioned(ed.prefix, parts.path) + target[len(parts.path):]
        else:
            # Mintlify resolves a relative link against the page's own file.
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(self.page), parts.path))
            if resolved == ".." or resolved.startswith("../"):
                self.fail(f"{target} leaves the edition; a relative link must stay within it")
            elif parts.path and not ed.publishes("/" + ("" if resolved == "." else resolved)):
                self.note(f"{target} leads to nothing {ed.name} publishes")
            return raw
        return f"<{pointed}>" if bracketed else pointed

    def code(self, text: str) -> None:
        """Note each URL of this site that code uses; code is never rewritten."""
        if self.quiet:
            return
        for found in self.site_url.finditer(text):
            path = re.split(r"[?#]", found[1])[0].rstrip(".,;:!")
            self.code_urls.append((self.where(), normal(path or "/")))

    def snippet(self, rel: str, text: str) -> None:
        """A snippet module is code that pages import: note its URLs, keep its text."""
        self.page, self.quiet = rel, False
        for number, line in enumerate(text.splitlines(), 1):
            self.line = number
            self.code(line)

    def attribute(self, match: re.Match) -> str:
        lead, name, quote, value = match[1], match[2], match[3], match[4]
        if name in NAMES_OR_LINKS and not value.startswith("/") and not SCHEME_RE.match(value):
            return match[0]
        return lead + quote + self.point(value) + quote

    def prose(self, segment: str, at_start: bool) -> str:
        """Text outside code and comments, with its links pointed into this edition."""
        # What is left once link targets, link text and attributes are blanked
        # is text, which Mintlify would turn into a link to this host.
        masked = INLINE_LINK_RE.sub(lambda m: m[1] + " " * len(m[2]), segment)
        definition = DEFINITION_RE.match(masked) if at_start else None
        if definition:
            masked = definition[1] + " " * len(definition[2]) + masked[definition.end():]
        masked = ANY_ATTRIBUTE_RE.sub(blank, LINK_TEXT_RE.sub(blank, masked))
        for found in self.site_url.finditer(masked):
            self.fail(
                f"{found[0]} is a bare link to {self.host}; write it as a Markdown link, which "
                "composition points into this edition"
            )
        if at_start:
            found = IMPORT_RE.match(segment)
            if found:
                module = found[3]
                if module.lstrip("/") not in self.ed.files:
                    self.note(f"imports {module}, which {self.ed.name} does not have")
                pointed = versioned(self.ed.prefix, module)
                return found[1] + found[2] + pointed + found[2] + segment[found.end():]
            found = DEFINITION_RE.match(segment)
            if found:
                segment = found[1] + self.point(found[2]) + segment[found.end():]
        segment = INLINE_LINK_RE.sub(lambda m: m[1] + self.point(m[2]), segment)
        return ATTRIBUTE_RE.sub(self.attribute, segment)

    def prose_line(self, line: str, comment: bool) -> tuple[str, bool]:
        """A line outside code fences: its prose rewritten, its code spans and MDX comments kept."""
        out, i = [], 0
        while i < len(line):
            if comment:
                end = line.find("*/}", i)
                if end < 0:
                    out.append(line[i:])
                    return "".join(out), True
                out.append(line[i : end + 3])
                i, comment = end + 3, False
                continue
            stops = [p for p in (line.find("`", i), line.find("{/*", i)) if p >= 0]
            stop = min(stops) if stops else len(line)
            out.append(self.prose(line[i:stop], at_start=i == 0))
            if stop == len(line):
                break
            if line.startswith("{/*", stop):
                i, comment = stop, True
                continue
            run = len(line) - stop - len(line[stop:].lstrip("`"))
            close = re.compile(rf"(?<!`){'`' * run}(?!`)").search(line, stop + run)
            end = close.end() if close else stop + run
            self.code(line[stop:end])
            out.append(line[stop:end])
            i = end
        return "".join(out), comment

    def frontmatter(self, line: str) -> str:
        body = line.rstrip("\r\n")
        found = FRONTMATTER_RE.fullmatch(body)
        if not found or not (found[3].startswith("/") or SCHEME_RE.match(found[3])):
            return line
        return found[1] + found[2] + self.point(found[3]) + found[2] + found[4] + line[len(body):]

    def rewrite(self, rel: str, text: str, published: bool) -> str:
        """One page, its links pointed into this edition; a draft's problems are not reported."""
        self.page, self.quiet = rel, not published
        lines = text.splitlines(keepends=True)
        out, start = [], 0
        if text.startswith("---\n"):
            end = next((i for i in range(1, len(lines)) if lines[i].rstrip("\r\n") == "---"), None)
            if end is not None:
                out.append(lines[0])
                for number in range(1, end):
                    self.line = number + 1
                    out.append(self.frontmatter(lines[number]))
                out.append(lines[end])
                start = end + 1
        fence, comment = None, False
        for number in range(start, len(lines)):
            self.line, line = number + 1, lines[number]
            if fence is not None:
                closing = FENCE_RE.match(line)
                if (closing and closing[1][0] == fence[0] and len(closing[1]) >= len(fence)
                        and not line[closing.end():].strip()):
                    fence = None
                else:
                    self.code(line)
                out.append(line)
                continue
            opening = None if comment else FENCE_RE.match(line)
            # A backtick in a backtick fence's info string makes it inline code.
            if opening and not (opening[1][0] == "`" and "`" in line[opening.end():]):
                fence = opening[1]
                out.append(line)
                continue
            line, comment = self.prose_line(line, comment)
            out.append(line)
        return "".join(out)


# navigation and the About page


def prefixed_nav(node, prefix: str, key: str | None = None):
    if isinstance(node, dict):
        if "openapi" in node or "asyncapi" in node:
            raise ComposeError("its navigation generates API pages, which cannot be composed under a version")
        return {k: prefixed_nav(v, prefix, k) for k, v in node.items()}
    if isinstance(node, list):
        return [prefixed_nav(item, prefix, key) for item in node]
    if isinstance(node, str):
        if key in ("pages", "root") and not SCHEME_RE.match(node):
            return f"{prefix}/{node.lstrip('/')}"
        if key in ("href", "icon") and node.startswith("/") and not node.startswith("//"):
            return versioned(prefix, node)
    return node


def first_pages(node) -> list | None:
    if isinstance(node, dict):
        if isinstance(node.get("pages"), list):
            return node["pages"]
        for key in ("tabs", "anchors", "dropdowns", "groups", "menu"):
            for child in node.get(key) or []:
                found = first_pages(child)
                if found is not None:
                    return found
    return None


def version_navigation(ed: Edition) -> dict:
    navigation = ed.docs["navigation"]
    nested = sorted(set(navigation) & {"products", "versions", "languages"})
    where = f"{ed.record['edition']}/{NAV}"
    if nested:
        raise ComposeError(f"{where}: navigation by {' and '.join(nested)} cannot be composed under a version")
    try:
        composed = prefixed_nav(copy.deepcopy(navigation), ed.prefix)
    except ComposeError as exc:
        raise ComposeError(f"{where}: {exc}") from None
    pages = first_pages(composed)
    if pages is None:
        raise ComposeError(f"{where} has no navigation group to list {ABOUT} in")
    pages.append(f"{ed.prefix}/{ABOUT}")
    return composed


def about_page(ed: Edition, product: dict) -> str:
    """The release and sources of an edition, from its record."""
    record, release, sources = ed.record, ed.record["release"], ed.record["sources"]
    title = f"{product['name']} {record['version']}"
    if record["product"] == "enterprise":
        facts = [
            ("Release", f"{title}, a qualified release"),
            ("Shared pages and examples",
             f"{monospace('lancedb/lancedb')} commit {monospace(sources['lancedb']['commit'])}, the LanceDB this release bundles"),
            ("Enterprise pages", "the Enterprise documentation of this release"),
        ]
        relation = (
            "An Enterprise release bundles a particular LanceDB, which can be older than the newest "
            "LanceDB OSS release. The pages it shares with LanceDB OSS can then differ from the newest "
            "LanceDB OSS documentation."
        )
    else:
        pinned = re.fullmatch(r"v([0-9]+\.[0-9]+\.[0-9]+)", release["enterprise_pages"]["ref"])
        facts = [
            ("Release", title),
            ("Pages and examples",
             f"{monospace('lancedb/lancedb')} tag {monospace(release['tag'])}, commit {monospace(release['commit'])}"),
            ("Enterprise pages",
             f"the Enterprise documentation of LanceDB Enterprise {pinned[1]}" if pinned
             else "a pinned revision of the Enterprise documentation"),
        ]
        relation = (
            "The Enterprise pages are included so that the navigation is complete. They come from the "
            "Enterprise documentation named above, pinned when this edition was prepared."
        )
    facts.append(("Geneva, dataset and REST API pages",
                  f"{monospace('lancedb/docs')} commit {monospace(sources['build']['commit'])}"))
    if "openapi" in sources:
        spec = sources["openapi"]
        facts.append(("REST API specification", f"{monospace(spec['repository'])} {monospace(spec['release'])}"))
    corrections = [
        f"- {plain(c['summary'])}" + (f" ({', '.join(monospace(f) for f in c['files'])})" if c.get("files") else "")
        for c in record["corrections"]
    ]
    return "\n".join([
        "---",
        'title: "About this version"',
        f"description: {json.dumps(f'The release and sources of the {title} documentation.')}",
        'icon: "circle-info"',
        "---",
        "",
        f"These pages document {title}. They were saved from the sources below when this edition was "
        "prepared, and later documentation changes reach them only as the corrections listed here.",
        "",
        *(f"- **{label}:** {value}" for label, value in facts),
        "",
        relation,
        "",
        "## Corrections {#corrections}",
        "",
        *(corrections or ["None."]),
        "",
    ])


# unversioned URLs and checks


def legacy_redirects(editions: list[Edition], default: Edition) -> tuple[list[dict], list[tuple[str, list[str]]]]:
    """Redirects from each unversioned path to the default edition, and the paths it lacks."""
    known: dict[str, list[str]] = {}
    for ed in editions:
        for path in sorted(ed.routes | {r.source for r in ed.redirects}):
            known.setdefault(path, []).append(ed.name)
    sources = {r.source: r for r in default.redirects}
    redirects, unresolved = [], []
    for path in sorted(known):
        # Mintlify itself sends / to the default edition.
        if path == "/":
            continue
        moved = sources.get(path)
        if path in default.routes:
            destination = versioned(default.prefix, path)
        elif moved is not None and moved.external:
            destination = moved.destination
        elif moved is not None and moved.wildcard:
            destination = versioned(default.prefix, moved.destination)
        elif moved is not None and (landed := default.landing(path)) is not None:
            destination = versioned(default.prefix, landed[0]) + (f"#{landed[1]}" if landed[1] else "")
        else:
            unresolved.append((path, known[path]))
            continue
        # Temporary, so that browsers follow a change of preferred edition.
        redirects.append(entry(path, destination, False))
    return redirects, unresolved


def check_redirects(redirects: list[dict], legacy: list[dict], routes: set[str], paths: set[str]) -> list[str]:
    """Duplicate sources, legacy redirects that capture pages, and chains that loop or lead nowhere."""
    problems, counts = [], {}
    for item in redirects:
        counts[item["source"]] = counts.get(item["source"], 0) + 1
    problems += [f"redirects {source} more than once" for source, n in sorted(counts.items()) if n > 1]
    parsed = [redirect(item, "the composed redirects") for item in redirects]
    exact = {}
    for r in parsed:
        if not r.wildcard:
            exact.setdefault(r.source, r)
    wildcards = [r for r in parsed if r.wildcard]
    for item in legacy:
        r = redirect(item, "an unversioned redirect")
        captured = sorted(route for route in routes if r.pattern.fullmatch(route))
        if captured:
            problems.append(f"the unversioned redirect from {r.source} would capture {captured[0]}")
    for r in parsed:
        if r.wildcard or r.source in paths:
            continue
        path, seen = r.source, []
        while path not in paths:
            found = exact.get(path) or next((w for w in wildcards if w.pattern.fullmatch(path)), None)
            if found is None:
                problems.append(f"the redirect from {r.source} leads to {path}, which the site does not have")
                break
            if path in seen:
                problems.append(f"the redirect from {r.source} loops: {' -> '.join([*seen, path])}")
                break
            seen.append(path)
            target = found.target(path)
            if SCHEME_RE.match(target):
                break
            path = target.partition("#")[0]
    return problems


def check_output(repo: Path, output: Path) -> None:
    """Refuse an output whose clearing would delete the editions or their sources."""
    for name in ("editions", "releases", "docs", "scripts"):
        kept = (repo / name).resolve()
        if output == kept or kept.is_relative_to(output) or output.is_relative_to(kept):
            raise ComposeError(f"the output {output} overlaps {kept}; composing would delete it")


# compose


def load_config(path: Path) -> dict:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ComposeError(f"{path.name}: unreadable: {exc}") from None
    if not isinstance(raw, dict) or set(raw) != {"host", "products"}:
        raise ComposeError(f"{path.name} sets `host` and `products`, and nothing else")
    if not isinstance(raw["host"], str) or not re.fullmatch(r"[a-z0-9.-]+", raw["host"]):
        raise ComposeError(f"{path.name}: host is the site's host name, such as docs.lancedb.com")
    products = raw["products"]
    named = [p.get("product") for p in products if isinstance(p, dict)] if isinstance(products, list) else []
    if not isinstance(products, list) or len(named) != len(products) or sorted(named) != sorted(edition.PRODUCTS):
        raise ComposeError(f"{path.name}: products lists {' and '.join(edition.PRODUCTS)}, once each, in selector order")
    for product in products:
        where = f"{path.name}: {product['product']}"
        extra = sorted(set(product) - {"product", "label", "name", "description", "icon", "preferred"})
        if extra:
            raise ComposeError(f"{where}: {', '.join(extra)} is not a field here")
        for key in ("label", "name", "description", "icon"):
            value = product.get(key)
            if (key in ("label", "name") or key in product) and not (isinstance(value, str) and value.strip()):
                raise ComposeError(f"{where}: {key} is text")
        preferred = product.get("preferred")
        if preferred is not None and not (isinstance(preferred, str) and edition.VERSION_RE.fullmatch(preferred)):
            raise ComposeError(f"{where}: preferred is {preferred!r}, not a version such as 1.2.3")
    return raw


def compose(repo: Path = REPO_ROOT, output: Path = OUTPUT) -> dict:
    """Write the composed site to `output` and return what it holds."""
    output = output.resolve()
    config = load_config(repo / CONFIG)
    check_output(repo, output)
    _states, problems = edition.check(repo)
    if problems:
        raise ComposeError("the saved editions do not match their records; see edition.py check:\n  "
                           + "\n  ".join(problems))
    records = [edition.load_record(path) for path in sorted((repo / "releases").glob("*.yaml"))]
    if not records:
        raise ComposeError("there are no saved editions to compose")
    editions = [read_edition(repo, record) for record in records]
    products = {p["product"]: p for p in config["products"]}
    shown = {
        name: sorted((e for e in editions if e.record["product"] == name), key=lambda e: e.version, reverse=True)
        for name in products
    }
    preferred = {}
    for name, product in products.items():
        wanted = product.get("preferred")
        found = next((e for e in shown[name] if e.record["version"] == wanted), None)
        if shown[name] and wanted is None:
            raise ComposeError(f"{CONFIG} names no preferred {name} edition; it has "
                               + ", ".join(e.record["version"] for e in shown[name]))
        if wanted is not None and found is None:
            raise ComposeError(f"{CONFIG} prefers {name} {wanted}, which is not a saved edition")
        if found is not None:
            preferred[name] = found
    first = config["products"][0]["product"]
    if first not in preferred:
        raise ComposeError(f"{CONFIG} lists {first} first, so readers land on its preferred edition, "
                           "but it has no saved edition")
    default = preferred[first]

    problems, warnings = [], []
    files: dict[str, bytes | Path] = {}
    # Files at the site's root: path -> (SHA-256, source, why).
    root: dict[str, tuple[str, Path, str]] = {}
    downloads: dict[str, list[str]] = {}

    def at_root(rel: str, ed: Edition, why: str) -> None:
        source = repo / ed.record["edition"] / rel
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        held = root.setdefault(rel, (digest, source, f"{why} of {ed.name}"))
        if held[0] != digest:
            problems.append(f"/{rel} is the {held[2]} and the {why} of {ed.name}, and they differ; "
                            "a file at an unversioned path must be the same for every edition that uses it")

    navigation, ignore, redirects = [], [f"# Composed by scripts/compose.py from each edition's {IGNORE}."], []
    for name, product in products.items():
        versions = []
        for ed in shown[name]:
            linker = Linker(ed, config["host"], problems, warnings)
            for rel in ed.files:
                source = repo / ed.record["edition"] / rel
                if rel in ed.pages:
                    text = source.read_text(encoding="utf-8")
                    files[f"{ed.prefix}/{rel}"] = linker.rewrite(rel, text, ed.pages[rel]).encode("utf-8")
                else:
                    if rel.startswith("snippets/") and rel.endswith(PAGE_SUFFIXES):
                        linker.snippet(rel, source.read_text(encoding="utf-8"))
                    files[f"{ed.prefix}/{rel}"] = source
            if any(f"{ABOUT}{suffix}" in ed.files for suffix in PAGE_SUFFIXES):
                problems.append(f"{ed.record['edition']} has its own {ABOUT} page")
            files[f"{ed.prefix}/{ABOUT}.mdx"] = about_page(ed, product).encode("utf-8")

            # Code fetches from the address it shows, so what it fetches is also
            # served there, the same for every edition.
            for where, path in linker.code_urls:
                found = VERSIONED_RE.match(path)
                if found:
                    if found[0] != f"/{ed.prefix}":
                        problems.append(f"{where}: code uses {path}, in another edition")
                    continue
                inside = sorted(a for a in ed.assets if a == path or a.startswith(path.rstrip("/") + "/"))
                for asset in inside:
                    at_root(asset[1:], ed, "download")
                    downloads.setdefault(asset, []).append(ed.name)
                if not inside:
                    warnings.append(
                        f"{where}: code uses https://{config['host']}{path}, which is left as written: "
                        + ("it opens the default edition's page" if ed.publishes(path)
                           else f"{ed.name} publishes nothing there")
                    )

            for r in ed.redirects:
                if not r.wildcard and not r.external and not ed.publishes(r.destination.partition("#")[0]):
                    problems.append(f"{ed.record['edition']}/{NAV}: the redirect from {r.source} leads to "
                                    f"{r.destination}, which {ed.name} does not publish")
                destination = r.destination if r.external else versioned(ed.prefix, r.destination)
                redirects.append(entry(versioned(ed.prefix, r.source), destination, r.permanent))
            if any(ed.ignore):
                ignore += ["", f"# {ed.name}", *ed.ignore]
            version = {"version": ed.record["version"]}
            if ed is preferred[name]:
                version.update(default=True, tag="Latest")
            versions.append({**version, **version_navigation(ed)})
        if versions:
            navigation.append({
                "product": product["label"],
                **{key: product[key] for key in ("description", "icon") if key in product},
                "versions": versions,
            })

    for rel in default.site_wide:
        at_root(rel, default, "site-wide file")
    settings = {key: value for key, value in default.docs.items() if key not in ("navigation", "redirects")}

    def chrome(value) -> None:
        if isinstance(value, dict):
            for item in value.values():
                chrome(item)
        elif isinstance(value, list):
            for item in value:
                chrome(item)
        elif isinstance(value, str) and value.startswith("/") and f"/{value.lstrip('/')}" in default.assets:
            at_root(value.lstrip("/"), default, "site setting")

    chrome(settings)
    legacy, unresolved = legacy_redirects(editions, default)
    redirects += legacy
    composed_routes = {versioned(e.prefix, r) for e in editions for r in e.routes}
    composed_routes |= {f"/{e.prefix}/{ABOUT}" for e in editions}
    composed_paths = composed_routes | {versioned(e.prefix, p) for e in editions for p in e.paths}
    composed_paths |= {f"/{rel}" for rel in root} | {f"/{e.record['product']}" for e in editions}
    problems += check_redirects(redirects, legacy, composed_routes, composed_paths)
    if problems:
        shown_problems = problems[:50] + ([f"... and {len(problems) - 50} more"] if len(problems) > 50 else [])
        raise ComposeError("the editions cannot be composed:\n  " + "\n  ".join(shown_problems))

    site = {}
    for key, value in default.docs.items():
        if key == "navigation":
            site[key] = {"products": navigation}
        elif key != "redirects":
            site[key] = value
    site["redirects"] = redirects
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    written = {**files, **{rel: source for rel, (_digest, source, _why) in root.items()}}
    for rel, data in sorted(written.items()):
        dest = output / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, Path):
            shutil.copyfile(data, dest)
        else:
            dest.write_bytes(data)
    (output / NAV).write_text(json.dumps(site, indent=2) + "\n", encoding="utf-8")
    (output / IGNORE).write_text("\n".join(ignore) + "\n", encoding="utf-8")
    return {
        "output": output,
        "editions": [(e.name, versioned(e.prefix, "/"), e is preferred[e.record["product"]])
                     for name in products for e in shown[name]],
        "default": default.name,
        "edition_redirects": len(redirects) - len(legacy),
        "legacy_redirects": len(legacy),
        "unresolved": unresolved,
        "downloads": downloads,
        "warnings": warnings,
        "files": len(written) + 2,
        "sha256": candidate.site_checksum(candidate.file_hashes(output)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--output", type=Path, default=OUTPUT,
                        help="where to write the site (default: build/versioned)")
    args = parser.parse_args(argv)
    try:
        summary = compose(REPO_ROOT, args.output)
    except (ComposeError, edition.EditionError, candidate.CandidateError) as exc:
        print(f"compose: {exc}", file=sys.stderr)
        return 1
    for warning in summary["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"composed {len(summary['editions'])} editions, {summary['files']} files, "
          f"sha256 {summary['sha256']}, into {summary['output']}")
    for name, path, is_preferred in summary["editions"]:
        print(f"  {name} at {path}" + (" (preferred)" if is_preferred else ""))
    print(f"redirects: {summary['edition_redirects']} within editions, "
          f"{summary['legacy_redirects']} from unversioned paths to {summary['default']}")
    for path, users in sorted(summary["downloads"].items()):
        print(f"served unversioned for code that downloads it: {path} ({', '.join(users)})")
    unresolved = summary["unresolved"]
    if unresolved:
        print(f"unversioned paths without a page in {summary['default']}, left without a redirect "
              f"for review ({len(unresolved)}):")
        for path, names in unresolved:
            print(f"  {path} ({', '.join(names)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
