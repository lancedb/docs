from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _unquote_env_value(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    return stripped


def load_env_file(path: Path | None = None) -> None:
    env_path = path or ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = _unquote_env_value(raw_value)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    audit_model: str
    audit_reasoning_effort: str
    embedding_model: str
    openai_timeout_seconds: int
    lancedb_api_key: str
    lancedb_host_override: str
    lancedb_region: str
    docs_audit_db_uri: str


def settings_from_env() -> Settings:
    return Settings(
        openai_api_key=(os.getenv("OPENAI_API_KEY") or "").strip(),
        audit_model=(os.getenv("DOCS_AUDIT_OPENAI_MODEL") or "gpt-5.5").strip(),
        audit_reasoning_effort=(
            os.getenv("DOCS_AUDIT_OPENAI_REASONING_EFFORT") or "high"
        ).strip(),
        embedding_model=(
            os.getenv("DOCS_AUDIT_EMBEDDING_MODEL") or "text-embedding-3-large"
        ).strip(),
        openai_timeout_seconds=int(os.getenv("DOCS_AUDIT_OPENAI_TIMEOUT_SECONDS", "900")),
        lancedb_api_key=(os.getenv("LANCEDB_API_KEY") or "").strip(),
        lancedb_host_override=(os.getenv("LANCEDB_HOST_OVERRIDE") or "").strip(),
        lancedb_region=(os.getenv("LANCEDB_REGION") or "us-east-1").strip()
        or "us-east-1",
        docs_audit_db_uri=(os.getenv("DOCS_AUDIT_DB_URI") or "db://docs-audit").strip(),
    )

