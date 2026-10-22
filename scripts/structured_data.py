"""Render JSON-LD as an MDX script in the server-rendered page HTML."""

import json


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
