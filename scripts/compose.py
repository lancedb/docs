"""
Compose the saved editions into one Mintlify site with product and version
selectors.

Each edition in editions/<product>/<version>/ is published as it was saved,
under /<product>/v<major>-<minor>-<patch>/: its pages, assets, navigation,
redirects and drafts, and an "About this version" page made from its release
record. Links in its pages, snippets and navigation are pointed into the
edition; code is left as written. compose.yaml orders the products and names
each one's preferred edition. Readers land on the first product's preferred
edition, and unversioned URLs redirect to its matching pages. Nothing is
rebuilt from product sources. See "Composing the site" in README.md.

Usage:

    python scripts/compose.py [--output DIR]
"""

from __future__ import annotations

import argparse
import bisect
import copy
import hashlib
import json
import posixpath
import re
import shutil
import subprocess
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
MDX = Path(__file__).resolve().parent / "mdx"
CONFIG = "compose.yaml"
SITE = "site"
SETTINGS = "settings.json"
OUTPUT = REPO_ROOT / "build" / "versioned"
NAV = assemble.NAV_BASE
IGNORE = ".mintignore"
ABOUT = "about-this-version"
PAGE_SUFFIXES = assemble.PAGE_SUFFIXES
# Mintlify loads these on every page; only site/ supplies them.
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
NAMES_OR_LINKS = ("icon",)
VERSIONED_RE = re.compile(rf"/(?:{'|'.join(edition.PRODUCTS)})/v[0-9]+-[0-9]+-[0-9]+(?=[/?#]|$)")
SCHEME_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")
PARAM_RE = re.compile(r":([A-Za-z_][A-Za-z0-9_]*)([*+?]?)")
FRONTMATTER_RE = re.compile(
    rf"""((?:{'|'.join(map(re.escape, FRONTMATTER_LINKS))})[ \t]*:[ \t]*)(["']?)([^"'\s]+)\2([ \t]*)"""
)


class ComposeError(Exception):
    """Inputs that must stop composition before anything is written."""


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


@dataclass(frozen=True)
class Redirect:
    source: str
    destination: str
    permanent: bool | None
    pattern: re.Pattern

    @property
    def wildcard(self) -> bool:
        return PARAM_RE.search(self.source) is not None

    @property
    def external(self) -> bool:
        return external_url(self.destination)

    def target(self, path: str) -> str:
        """Where this redirect sends `path`, which its source matches."""
        found = self.pattern.fullmatch(path)
        values = found.groupdict() if found else {}
        target = PARAM_RE.sub(lambda m: values.get(m[1]) or "", self.destination)
        if external_url(target):
            return target
        parts = urllib.parse.urlsplit(target)
        return urllib.parse.urlunsplit(parts._replace(path=normal(safe_path(parts.path))))


def redirect_pattern(source: str) -> re.Pattern:
    regex, wildcards = "", 0
    for segment in [s for s in source.split("/") if s]:
        param = PARAM_RE.fullmatch(segment)
        if param and param[2] == "*":
            wildcards += 1
            regex += rf"(?:/(?P<{param[1]}>.*))?"
        elif not any(char in segment for char in "*:?+#"):
            regex += "/" + re.escape(segment)
        else:
            raise ComposeError(f"unsupported redirect source {source!r}; use a literal path or one :name* segment")
    if wildcards > 1:
        raise ComposeError(f"unsupported redirect source {source!r}; use only one :name* segment")
    return re.compile(regex or "/")


def redirect(item: object, where: str, host: str | None = None) -> Redirect:
    if (
        not isinstance(item, dict)
        or not isinstance(item.get("source"), str)
        or not isinstance(item.get("destination"), str)
        or not item["source"].startswith("/")
    ):
        raise ComposeError(f"{where}: a redirect needs a source path and a destination, not {item!r}")
    destination = item["destination"]
    parts = urllib.parse.urlsplit(destination)
    if host and parts.hostname == host and parts.scheme.lower() in ("", "http", "https"):
        destination = urllib.parse.urlunsplit(("", "", parts.path or "/", parts.query, parts.fragment))
    if not external_url(destination):
        parts = urllib.parse.urlsplit(destination)
        path = parts.path if parts.path.startswith("/") else "/" + parts.path
        destination = urllib.parse.urlunsplit(parts._replace(path=safe_path(path)))
    source = normal(safe_path(item["source"]))
    return Redirect(source, destination, item.get("permanent"), redirect_pattern(source))


def entry(source: str, destination: str, permanent: bool | None) -> dict:
    return {"source": source, "destination": destination,
            **({} if permanent is None else {"permanent": permanent})}


def normal(path: str) -> str:
    return path.rstrip("/") or "/"


def external_url(target: str) -> bool:
    return bool(SCHEME_RE.match(target) or target.startswith("//"))


def safe_path(path: str) -> str:
    # Browsers normalize encoded dot segments too, before requesting a page.
    segments = []
    for segment in path.split("/"):
        decoded = urllib.parse.unquote(segment)
        if "\\" in decoded or "/" in decoded:
            raise ComposeError(f"{path} has an encoded separator or backslash; use a plain path")
        if decoded == "..":
            if not segments:
                raise ComposeError(f"{path} leaves the edition")
            segments.pop()
        elif decoded not in ("", "."):
            segments.append(segment)
    result = ("/" if path.startswith("/") else "") + "/".join(segments)
    return result + ("/" if path.endswith("/") and segments else "")


def matching_redirect(path: str, redirects: list[Redirect]) -> Redirect | None:
    return next((r for r in redirects if not r.wildcard and r.source == path), None) or next(
        (r for r in redirects if r.wildcard and r.pattern.fullmatch(path)), None)


def follow_redirects(start: str, redirects: list[Redirect], paths: set[str], *, symbolic=False) -> str | None:
    path, query, anchor, seen = normal(start), "", "", set()
    for _ in range(100):
        found = matching_redirect(path, redirects)
        if found is None:
            if path not in paths and not (symbolic and PARAM_RE.search(path)):
                return None
            return urllib.parse.urlunsplit(("", "", path, query, anchor))
        if path in seen:
            raise ComposeError(f"the redirect from {start} loops at {path}")
        seen.add(path)
        target = found.target(path)
        if external_url(target):
            return target
        # Mintlify ignores a wildcard's no-op match at its parent page.
        if found.wildcard and target == path and found.source != found.destination:
            return path if path in paths else None
        parts = urllib.parse.urlsplit(target)
        path, query, anchor = normal(parts.path), parts.query or query, parts.fragment or anchor
    raise ComposeError(f"the redirect from {start} does not terminate within 100 hops")


def versioned(prefix: str, path: str) -> str:
    return f"/{prefix}" + ("" if path in ("/", "") else path)


@dataclass
class Edition:
    record: dict
    prefix: str
    docs: dict
    ignore: list[str]
    files: list[str]
    pages: dict[str, bool]
    routes: set[str]
    paths: set[str]
    assets: set[str]
    redirects: list[Redirect]

    @property
    def name(self) -> str:
        return f"{self.record['product']} {self.record['version']}"

    @property
    def version(self) -> tuple[int, ...]:
        return tuple(int(part) for part in self.record["version"].split("."))

    def match(self, path: str) -> Redirect | None:
        return matching_redirect(path, self.redirects)

    def publishes(self, path: str) -> bool:
        """Whether a site path names a page, folder or file of this edition, or one of its redirects."""
        path = normal(urllib.parse.unquote(path))
        if path in self.paths or (path.endswith(PAGE_SUFFIXES) and path.rsplit(".", 1)[0] in self.routes):
            return True
        return self.match(path) is not None

    def landing(self, path: str) -> str | None:
        try:
            return follow_redirects(path, self.redirects, self.paths)
        except ComposeError:
            # check_redirects reports the cycle with the composed route.
            return None


def read_edition(repo: Path, record: dict, host: str) -> Edition:
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
        and not p.startswith("snippets/")
        and not (p.endswith(".md") and PurePosixPath(p).stem.lower() in NOT_PAGES)
    }
    routes = {assemble.page_route(p) for p, shown in pages.items() if shown}
    # A link may also name a folder's index page as `folder/index`.
    stems = {"/" + p.rsplit(".", 1)[0] for p, shown in pages.items() if shown}
    folders = {"/".join(r.split("/")[:depth]) for r in routes for depth in range(2, r.count("/") + 1)}
    assets = {f"/{p}" for p in published if p in files and p not in pages}
    redirects = [redirect(item, f"{rel}/{NAV}", host) for item in docs.get("redirects") or []]
    return Edition(record, prefix, docs, ignore, files, pages, routes, routes | stems | folders | assets,
                   assets, redirects)


def monospace(text: str) -> str:
    fence = "`" * (max((len(run) for run in re.findall(r"`+", text)), default=0) + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def plain(text: str) -> str:
    """`text` shown as itself in MDX."""
    return re.sub(r"([\\`*_{}\[\]<>|~#!&])", r"\\\1", text.strip())


def mdx_links(texts: list[str]) -> list[dict]:
    """The links, imports and code of each MDX text, as the parser Mint uses reads them."""
    digest = hashlib.sha256((MDX / "package-lock.json").read_bytes()).hexdigest()
    installed = MDX / "node_modules" / ".package-lock.sha256"
    try:
        if not installed.is_file() or installed.read_text() != digest:
            subprocess.run(["npm", "ci", "--ignore-scripts", "--no-audit", "--no-fund", "--loglevel=error"],
                           cwd=MDX, check=True, capture_output=True, encoding="utf-8")
            installed.write_text(digest)
        result = subprocess.run(["node", str(MDX / "links.mjs")], check=True, capture_output=True,
                                encoding="utf-8", input=json.dumps({"attributes": LINK_ATTRIBUTES, "texts": texts}))
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ComposeError("cannot run the MDX parser in scripts/mdx, which needs Node.js and npm: "
                           f"{getattr(exc, 'stderr', None) or exc}") from None
    return json.loads(result.stdout)


class Linker:
    """Points the links in one edition's pages, snippets and navigation into the edition."""

    def __init__(self, ed: Edition, host: str, problems: list[str], warnings: list[str]):
        self.ed, self.host = ed, host
        self.problems, self.warnings = problems, warnings
        self.code_urls: list[tuple[str, str]] = []
        self.page, self.line, self.quiet, self.snippet = "", 0, False, False
        self.site_url = re.compile(
            rf"(?i)(?<![\w.-])https?://{re.escape(host)}(?![\w.-])([^\s\"'`<>()\[\]{{}}|\\]*)"
        )

    def where(self) -> str:
        return f"{self.ed.record['edition']}/{self.page}" + (f":{self.line}" if self.line else "")

    def fail(self, message: str) -> None:
        if not self.quiet:
            self.problems.append(f"{self.where()}: {message}")

    def note(self, message: str) -> None:
        if not self.quiet:
            self.warnings.append(f"{self.where()}: {message}")

    def point(self, target: str) -> str:
        if not target or target.startswith("#"):
            return target
        parts = urllib.parse.urlsplit(target)
        ed = self.ed
        absolute = bool(parts.scheme or target.startswith("//"))
        if absolute and (parts.scheme.lower() not in ("", "http", "https") or parts.hostname != self.host):
            return target
        try:
            if not absolute and not target.startswith("/"):
                if self.snippet:
                    if ".." in (urllib.parse.unquote(s) for s in parts.path.split("/")):
                        raise ComposeError(f"{target} climbs out of the folder of each page that imports this "
                                           "snippet; link from the root instead, such as /guide")
                    safe_path(parts.path)
                    return target
                try:
                    resolved = safe_path(posixpath.join(posixpath.dirname(self.page), parts.path))
                except ComposeError as exc:
                    raise ComposeError(f"{target} leaves the edition or has an unsafe path: {exc}") from None
                if parts.path and not ed.publishes("/" + resolved):
                    self.note(f"{target} leads to nothing {ed.name} publishes")
                return target
            original = parts.path or "/"
            found = VERSIONED_RE.match(original)
            if found:
                if found[0] != f"/{ed.prefix}":
                    self.fail(f"{target} points into another edition; an edition links only within itself")
                    return target
                original = original[len(found[0]):] or "/"
            path = safe_path(original)
            if absolute:
                landed = ed.landing(path)
                if landed is not None and external_url(landed):
                    return landed
                if landed is None:
                    self.note(f"{target} has no page in {ed.name}; it now points at that path in {ed.name}")
                resolved = urllib.parse.urlsplit(landed or path)
                path = resolved.path
                parts = parts._replace(query=parts.query or resolved.query, fragment=parts.fragment or resolved.fragment)
            elif not ed.publishes(path):
                self.note(f"{target} leads to nothing {ed.name} publishes")
            pointed = urllib.parse.urlunsplit(("", "", versioned(ed.prefix, path), parts.query, parts.fragment))
        except ComposeError as exc:
            self.fail(str(exc))
            return target
        return pointed

    def code(self, text: str) -> None:
        """Note each URL of this site that code uses; code is never rewritten."""
        if self.quiet:
            return
        for found in self.site_url.finditer(text):
            path = re.split(r"[?#]", found[1])[0].rstrip(".,;:!")
            self.code_urls.append((self.where(), normal(path or "/")))

    def frontmatter(self, line: str) -> str:
        body = line.rstrip("\r\n")
        found = FRONTMATTER_RE.fullmatch(body)
        if not found or not (found[3].startswith("/") or SCHEME_RE.match(found[3])):
            return line
        return found[1] + found[2] + self.point(found[3]) + found[2] + found[4] + line[len(body):]

    def rewrite(self, rel: str, text: str, found: dict, published: bool, snippet: bool = False) -> str:
        """One page or snippet, its links pointed into this edition; a draft's problems are not reported."""
        self.page, self.quiet, self.snippet = rel, not published, snippet
        newlines = [m.start() for m in re.finditer("\n", text)]

        def at(offset: int) -> None:
            self.line = bisect.bisect_left(newlines, offset) + 1

        if "error" in found:
            line, reason = found["error"]
            self.line = line or 0
            self.fail(f"is not valid MDX: {reason}")
            return text
        edits = []
        for start, end, value, name in found["links"]:
            at(start)
            named_icon = name in NAMES_OR_LINKS and not value.startswith("/") and not SCHEME_RE.match(value)
            pointed = value if named_icon else self.point(value)
            if pointed != value:
                edits.append((start, end, value, pointed))
        for start, end, module in found["imports"]:
            if module.startswith("/"):
                at(start)
                if module.lstrip("/") not in self.ed.files:
                    self.note(f"imports {module}, which {self.ed.name} does not have")
                edits.append((start, end, module, versioned(self.ed.prefix, module)))
        for start, name in found["computed"]:
            at(start)
            self.fail(f"{name} is computed, so composition cannot point it into this edition; "
                      f'write the path itself, such as {name}="/guide"')
        for start, url in found["bare"]:
            at(start)
            if self.site_url.match(url):
                self.fail(f"{url} is a bare link to {self.host}; write it as a Markdown link, which "
                          "composition points into this edition")
        for start, code in found["code"]:
            at(start)
            self.code(code)
        out, last = [], 0
        for start, end, value, pointed in sorted(edits):
            if text[start:end] != value:
                at(start)
                self.fail(f"{text[start:end]} has escapes or character references; write the path itself")
                continue
            out += [text[last:start], pointed]
            last = end
        lines = "".join([*out, text[last:]]).splitlines(keepends=True)
        if text.removeprefix("\ufeff").startswith("---\n"):
            end = next((i for i in range(1, len(lines)) if lines[i].rstrip("\r\n") == "---"), 0)
            for number in range(1, end):
                self.line = number + 1
                lines[number] = self.frontmatter(lines[number])
        return "".join(lines)


def prefixed_nav(node, prefix: str, linker: Linker, key: str | None = None):
    if isinstance(node, dict):
        if "openapi" in node or "asyncapi" in node:
            raise ComposeError("its navigation generates API pages, which cannot be composed under a version")
        return {k: prefixed_nav(v, prefix, linker, k) for k, v in node.items()}
    if isinstance(node, list):
        return [prefixed_nav(item, prefix, linker, key) for item in node]
    if isinstance(node, str):
        if key in ("pages", "root") and not SCHEME_RE.match(node):
            return f"{prefix}/{node.lstrip('/')}"
        if key == "href" or (key == "icon" and (node.startswith("/") or SCHEME_RE.match(node))):
            return linker.point(node)
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


def version_navigation(ed: Edition, linker: Linker) -> dict:
    navigation = ed.docs["navigation"]
    nested = sorted(set(navigation) & {"products", "versions", "languages"})
    where = f"{ed.record['edition']}/{NAV}"
    if nested:
        raise ComposeError(f"{where}: navigation by {' and '.join(nested)} cannot be composed under a version")
    linker.page, linker.line, linker.quiet, linker.snippet = NAV, 0, False, False
    try:
        composed = prefixed_nav(copy.deepcopy(navigation), ed.prefix, linker)
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
            destination = landed if external_url(landed) else versioned(default.prefix, landed)
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
    for item in legacy:
        r = redirect(item, "an unversioned redirect")
        captured = sorted(route for route in routes if r.pattern.fullmatch(route))
        if captured:
            problems.append(f"the unversioned redirect from {r.source} would capture {captured[0]}")
    for r in parsed:
        probes = {r.source}
        if r.wildcard:
            probes.update(path for path in paths | {p.source for p in parsed if not p.wildcard}
                          if r.pattern.fullmatch(path))
        for path in sorted(probes):
            try:
                landed = follow_redirects(path, parsed, paths, symbolic=r.wildcard and path == r.source)
                if landed is None:
                    problems.append(f"the redirect from {r.source} leads nowhere the site publishes")
            except ComposeError as exc:
                problems.append(str(exc))
    return problems


def check_output(repo: Path, output: Path) -> None:
    """Refuse an output whose clearing would delete the editions or their sources."""
    for name in ("editions", "releases", "docs", "scripts", SITE):
        kept = (repo / name).resolve()
        if output == kept or kept.is_relative_to(output) or output.is_relative_to(kept):
            raise ComposeError(f"the output {output} overlaps {kept}; composing would delete it")


def shared_site(repo: Path) -> tuple[dict, dict[str, tuple[str, Path, str]]]:
    root = repo / SITE
    if root.is_symlink():
        raise ComposeError(f"{SITE} is a symlink")
    hashes = candidate.file_hashes(root)
    try:
        settings = json.loads((root / SETTINGS).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ComposeError(f"{SITE}/{SETTINGS}: unreadable: {exc}") from None
    if not isinstance(settings, dict) or not isinstance(settings.get("name"), str):
        raise ComposeError(f"{SITE}/{SETTINGS} must be an object with the site's name")
    if set(settings) & {"navigation", "redirects"}:
        raise ComposeError(f"{SITE}/{SETTINGS} cannot set navigation or redirects; composition generates them")
    assets = {}
    for rel, digest in hashes.items():
        if rel == SETTINGS:
            continue
        if not rel.startswith("static/") or rel.endswith(PAGE_SUFFIXES):
            raise ComposeError(f"{SITE}/{rel}: shared assets belong in static/ and cannot be pages")
        assets[rel] = digest, root / rel, f"shared asset {SITE}/{rel}"
    return settings, assets


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
    editions = [read_edition(repo, record, config["host"]) for record in records]
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
    settings, root = shared_site(repo)
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
            texts = {
                rel: (repo / ed.record["edition"] / rel).read_text(encoding="utf-8")
                for rel in ed.files
                if rel in ed.pages or (rel.startswith("snippets/") and rel.endswith(PAGE_SUFFIXES))
            }
            found = dict(zip(texts, mdx_links(list(texts.values()))))
            for rel in ed.files:
                source = repo / ed.record["edition"] / rel
                if rel in ed.pages:
                    pointed = linker.rewrite(rel, texts[rel], found[rel], ed.pages[rel])
                    files[f"{ed.prefix}/{rel}"] = pointed.encode("utf-8")
                elif rel in texts:
                    pointed = linker.rewrite(rel, texts[rel], found[rel], f"/{rel}" in ed.assets, snippet=True)
                    # A snippet with nothing to point is copied byte for byte.
                    files[f"{ed.prefix}/{rel}"] = source if pointed == texts[rel] else pointed.encode("utf-8")
                else:
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
                if not r.wildcard and not r.external and not ed.publishes(urllib.parse.urlsplit(r.destination).path):
                    problems.append(f"{ed.record['edition']}/{NAV}: the redirect from {r.source} leads to "
                                    f"{r.destination}, which {ed.name} does not publish")
                destination = r.destination if r.external else versioned(ed.prefix, r.destination)
                redirects.append(entry(versioned(ed.prefix, r.source), destination, r.permanent))
            if any(ed.ignore):
                ignore += ["", f"# {ed.name}", *ed.ignore]
            version = {"version": ed.record["version"]}
            if ed is preferred[name]:
                version.update(default=True, tag="Latest")
            versions.append({**version, **version_navigation(ed, linker)})
        if versions:
            navigation.append({
                "product": product["label"],
                **{key: product[key] for key in ("description", "icon") if key in product},
                "versions": versions,
            })

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

    site = {**settings, "navigation": {"products": navigation}, "redirects": redirects}
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
