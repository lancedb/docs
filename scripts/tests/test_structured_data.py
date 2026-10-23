"""Guard JSON-LD generation against losing visible content or going stale."""

import html
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sync_faq_schema import END, START, faq_schema, plain_text, update_page
from sync_hf_datasets import Dataset, transform_card
from structured_data import dataset_schema, render_jsonld


def jsonld_payload(markup: str) -> str:
    return markup.split("children='", 1)[1].split("' />", 1)[0]


class JsonLdTests(unittest.TestCase):
    def test_literal_payload_preserves_quotes_and_blocks_script_termination(self):
        data = {"text": "LanceDB's \"FAQ\" & </script><script>alert(1)</script>\nNext line"}
        markup = render_jsonld(data)
        self.assertNotIn("</script>", markup)
        self.assertNotIn("JSON.stringify", markup)
        payload = jsonld_payload(markup)
        self.assertNotIn("'", payload)
        self.assertEqual(json.loads(html.unescape(payload)), data)


class DatasetSchemaTests(unittest.TestCase):
    def test_card_page_starts_with_dataset_schema_from_opening_paragraph(self):
        ds = Dataset(dir="example", slug="example", hf="example-lance",
                     title="Example", category="Vision")
        card = (
            "---\nlicense: mit\n---\n# Example\n\n"
            "A **small** [dataset](https://example.com/data_(v2)) of `image_bytes`.\n\n"
            "## Usage\n\nLater text.\n"
        )
        mdx, description = transform_card(ds, card)
        self.assertEqual(description, "A small dataset of image_bytes.")
        self.assertLess(mdx.index("<script"), mdx.index("<Card"))
        self.assertEqual(
            json.loads(html.unescape(jsonld_payload(mdx))),
            dataset_schema(
                "Example", "A small dataset of image_bytes.", "example",
                "https://huggingface.co/datasets/lance-format/example-lance",
            ),
        )

    def test_dataset_schema_identifies_docs_page_and_hub_copy(self):
        schema = dataset_schema("DocVQA", "Documents.", "docvqa", "https://hf.co/d")
        self.assertEqual(schema["@type"], "Dataset")
        self.assertEqual(schema["url"], "https://docs.lancedb.com/datasets/docvqa")
        self.assertEqual(schema["@id"], "https://docs.lancedb.com/datasets/docvqa#dataset")
        self.assertEqual(schema["sameAs"], "https://hf.co/d")


class FaqSchemaTests(unittest.TestCase):
    def test_extracts_questions_without_category_headings(self):
        source = """## Architecture

### Is replication available?

Yes, with two replicas.

## Monitoring

### How can I monitor it?

Use Prometheus.
"""
        questions = faq_schema(source, "faq/example")["mainEntity"]
        self.assertEqual([q["name"] for q in questions], [
            "Is replication available?", "How can I monitor it?",
        ])
        self.assertEqual(questions[0]["acceptedAnswer"]["text"], "Yes, with two replicas.")

    def test_preserves_lists_callouts_and_code_identifiers(self):
        source = """Use `weak_read_consistency_interval_seconds`.

<Tip>
**Recommended setting:** 30–60 seconds.
</Tip>

- Call `table.add()`.
- Read [the guide](https://example.com/guide_(v2)).

<Warning>
Do not use `fork` &amp; threads together.
</Warning>

![](/chart.png)
"""
        self.assertEqual(
            plain_text(source),
            "Use weak_read_consistency_interval_seconds. Recommended setting: 30–60 seconds. "
            "Call table.add(). Read the guide. Do not use fork & threads together.",
        )

    def test_strips_trailing_heading_space_and_underscore_emphasis(self):
        questions = faq_schema("## Is it _fast_? \n\nYes, __very__ fast.\n", "faq")["mainEntity"]
        self.assertEqual(questions[0]["name"], "Is it fast?")
        self.assertEqual(questions[0]["acceptedAnswer"]["text"], "Yes, very fast.")

    def test_regeneration_replaces_old_answers_and_is_idempotent(self):
        source = "## Is it available?\n\nYes.\n"
        generated = update_page(source, "faq")
        self.assertEqual(update_page(generated, "faq"), generated)
        revised = generated.replace("\nYes.\n", "\nNo.\n")
        updated = update_page(revised, "faq")
        self.assertIn('"text": "No."', updated)
        self.assertNotIn('"text": "Yes."', updated)
        self.assertEqual(updated.count('"@type": "FAQPage"'), 1)

    def test_code_operators_are_not_treated_as_emphasis_or_mdx(self):
        self.assertEqual(plain_text('Fetch `rf * k` *full* vectors and set `{"n": 2}`.'),
                         'Fetch rf * k full vectors and set {"n": 2}.')

    def test_rejects_missing_answers_and_unsupported_markdown(self):
        for source in [
            "## Empty?\n",
            "## Code?\n\n```python\npass\n```",
            "## Dynamic?\n\n{answer}",
            "## Component?\n\n<Card>Text</Card>",
            "## Table?\n\n| A | B |\n|---|---|\n| 1 | 2 |",
            "## Quote?\n\n> Quoted text.",
            "## Reference?\n\nSee [the guide][guide].\n\n[guide]: https://example.com",
        ]:
            with self.subTest(source=source), self.assertRaises(ValueError):
                faq_schema(source, "faq")

    def test_rejects_text_that_would_be_left_out_of_answers(self):
        for source in [
            "### Is it fast?\n\nYes.\n\n#### Benchmark\n\nUnder 10 ms.\n",
            "### Can I use it? (Python)\n\nYes.\n",
            "## Overview\n\nIntro text.\n\n## Is it fast?\n\nYes.\n",
        ]:
            with self.subTest(source=source), self.assertRaises(ValueError):
                faq_schema(source, "faq")

    def test_errors_name_the_question(self):
        with self.assertRaisesRegex(ValueError, "in question: Dynamic\\?"):
            faq_schema("## Dynamic?\n\n{answer}", "faq")

    def test_rejects_damaged_markers_without_dropping_content(self):
        source = "## Available?\n\nYes.\n"
        for suffix in [START, END, END + START, START + END + "more content"]:
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                update_page(source + suffix, "faq")


if __name__ == "__main__":
    unittest.main()
