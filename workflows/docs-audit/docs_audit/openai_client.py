from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


OPENAI_API_BASE = "https://api.openai.com/v1"


class OpenAIClient:
    def __init__(self, *, api_key: str, timeout_seconds: int = 900) -> None:
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{OPENAI_API_BASE}{path}",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API request failed: {exc.code} {detail}") from exc

    def response_text(
        self,
        *,
        model: str,
        reasoning_effort: str,
        input_text: str,
    ) -> str:
        payload = {
            "model": model,
            "input": input_text,
            "reasoning": {"effort": reasoning_effort},
            "store": False,
        }
        data = self._post_json("/responses", payload)
        text = extract_response_text(data)
        if not text.strip():
            raise RuntimeError("OpenAI response did not include output text")
        return text

    def embeddings(self, *, model: str, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []
        data = self._post_json("/embeddings", {"model": model, "input": inputs})
        rows = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
        embeddings = [row.get("embedding") for row in rows]
        if len(embeddings) != len(inputs) or any(not item for item in embeddings):
            raise RuntimeError("OpenAI embeddings response did not match input count")
        return embeddings


def extract_response_text(data: dict[str, Any]) -> str:
    top_level = data.get("output_text")
    if isinstance(top_level, str) and top_level.strip():
        return top_level

    chunks: list[str] = []
    for item in data.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    return "\n".join(chunks)

