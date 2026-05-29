from __future__ import annotations

from docs_audit.openai_client import extract_response_text


def test_extract_response_text_prefers_top_level_output_text() -> None:
    assert extract_response_text({"output_text": "hello"}) == "hello"


def test_extract_response_text_reads_message_content() -> None:
    payload = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "first"},
                    {"type": "output_text", "text": "second"},
                ],
            }
        ]
    }

    assert extract_response_text(payload) == "first\nsecond"

