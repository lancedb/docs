"""Render JSON-LD as an MDX script in the server-rendered page HTML."""

import json
import re

CODE = re.compile(r"`([^`]+)`")
TARGET = r"\((?:[^()\n]|\([^()\n]*\))*\)"
IMAGE = re.compile(r"!\[[^\]]*\]" + TARGET)
LINK = re.compile(r"\[([^\]]+)\]" + TARGET)
EMPHASIS = re.compile(r"\*+([^*]+)\*+|(?<!\w)_+([^_]+)_+(?!\w)")


def strip_markdown(text: str) -> str:
    """Reduce inline Markdown to its visible text without altering code spans."""
    code = []

    def protect_code(match: re.Match) -> str:
        code.append(match.group(1))
        return f"\x00{len(code) - 1}\x00"

    text = CODE.sub(protect_code, text)
    text = IMAGE.sub("", text)
    text = LINK.sub(r"\1", text)
    text = EMPHASIS.sub(lambda match: match.group(1) or match.group(2), text)
    return re.sub(r"\x00(\d+)\x00", lambda match: code[int(match.group(1))], text)


def render_jsonld(data: dict) -> str:
    # A literal children prop also renders without Mintlify's client-side
    # expression evaluator. Escape script terminators and JSX attribute quotes.
    payload = json.dumps(data, ensure_ascii=False, indent=2).replace("<", "\\u003c")
    payload = payload.replace("&", "&amp;").replace("'", "&#39;")
    return (
        '<script type="application/ld+json" children=\'' + payload + "' />\n"
    )


def dataset_schema(name: str, description: str, slug: str, hub_url: str) -> dict:
    url = f"https://docs.lancedb.com/datasets/{slug}"
    return {
        "@context": "https://schema.org",
        "@type": "Dataset",
        "@id": f"{url}#dataset",
        "name": name,
        "description": description,
        "url": url,
        "sameAs": hub_url,
    }
