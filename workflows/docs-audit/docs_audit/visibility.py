from __future__ import annotations


EXCLUDED_TOPIC_TERMS = (
    "helm",
    "helm chart",
    "kubernetes chart",
    "enterprise deployment",
    "enterprise deploy",
)


def visibility_class(text: str) -> str:
    normalized = text.casefold()
    if any(term in normalized for term in EXCLUDED_TOPIC_TERMS):
        return "excluded"
    return "public-doc-gap"


def is_public_finding(text: str) -> bool:
    return visibility_class(text) == "public-doc-gap"

