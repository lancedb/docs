from __future__ import annotations

from docs_audit.report_parser import embedding_text, parse_report_findings


def test_parse_report_findings_resolves_page_metadata() -> None:
    report = """# Missing Documentation Observations

## Full-Text Search Index

- Model-backed FTS tokenizers are missing from the tokenizer documentation.
- The n-gram parameter names are inconsistent with the Python API.
"""
    findings = parse_report_findings(
        report,
        run_id="run-1",
        area="indexing",
        page_lookup={
            "full-text search index": {
                "page_id": "fts-index",
                "page_title": "Full-Text Search Index",
                "page_path": "docs/indexing/fts.mdx",
            }
        },
    )

    assert [finding.finding_index for finding in findings] == [1, 2]
    assert findings[0].id == "run-1:001"
    assert findings[0].page_id == "fts-index"
    assert findings[0].visibility_class == "public-doc-gap"
    assert findings[0].finding_hash.startswith("sha256:")
    assert "Area: indexing" in embedding_text(findings[0])
    assert "Docs path: docs/indexing/fts.mdx" in embedding_text(findings[0])


def test_parse_report_findings_marks_excluded_topics() -> None:
    report = """# Missing Documentation Observations

## Deployment

- The helm chart values for enterprise deployment are not documented.
"""
    findings = parse_report_findings(report, run_id="run-1", area="storage")

    assert len(findings) == 1
    assert findings[0].visibility_class == "excluded"

