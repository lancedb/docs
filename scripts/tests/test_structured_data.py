"""Guard FAQ extraction against losing visible answer content or going stale."""

import html
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sync_faq_schema import END, START, faq_schema, plain_text, update_page
from structured_data import render_jsonld


class JsonLdTests(unittest.TestCase):
    def test_literal_payload_preserves_quotes_and_blocks_script_termination(self):
        data = {"text": "LanceDB's \"FAQ\" & </script><script>alert(1)</script>\nNext line"}
        markup = render_jsonld(data)
        self.assertNotIn("</script>", markup)
        self.assertNotIn("JSON.stringify", markup)
        payload = markup.split("children='", 1)[1].rsplit("' />", 1)[0]
        self.assertNotIn("'", payload)
        self.assertEqual(json.loads(html.unescape(payload)), data)


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
        self.assertEqual(plain_text(source),
            "Use weak_read_consistency_interval_seconds. Recommended setting: 30–60 seconds. "
            "Call table.add(). Read the guide. Do not use fork & threads together.")

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

    def test_rejects_missing_answers_and_unsupported_mdx(self):
        for source in ["## Empty?\n", "## Code?\n\n```python\npass\n```",
                       "## Dynamic?\n\n{answer}", "## Component?\n\n<Card>Text</Card>"]:
            with self.subTest(source=source), self.assertRaises(ValueError):
                faq_schema(source, "faq")

    def test_rejects_damaged_markers_without_dropping_content(self):
        source = "## Available?\n\nYes.\n"
        for suffix in [START, END, END + START, START + END + "more content"]:
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                update_page(source + suffix, "faq")


if __name__ == "__main__":
    unittest.main()
