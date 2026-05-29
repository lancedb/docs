from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import lancedb
import pyarrow as pa


RUNS_TABLE = "docs_audit_runs"
FINDINGS_TABLE = "docs_audit_findings"
TABLE_READY_TIMEOUT_SECONDS = 30.0
TABLE_READY_SLEEP_SECONDS = 0.5
TABLE_READY_MAX_ATTEMPTS = int(TABLE_READY_TIMEOUT_SECONDS / TABLE_READY_SLEEP_SECONDS)


RUNS_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string()),
        pa.field("completed_at", pa.timestamp("us", tz="UTC")),
        pa.field("areas", pa.list_(pa.string())),
        pa.field("report_text", pa.string()),
        pa.field("report_path", pa.string()),
        pa.field("repo_shas", pa.string()),
        pa.field("selected_pages", pa.list_(pa.string())),
        pa.field("changed_pages", pa.list_(pa.string())),
        pa.field("refresh", pa.string()),
        pa.field("metadata", pa.string()),
    ]
)


def findings_schema(embedding_dimension: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("run_id", pa.string()),
            pa.field("completed_at", pa.timestamp("us", tz="UTC")),
            pa.field("area", pa.string()),
            pa.field("page_id", pa.string()),
            pa.field("page_title", pa.string()),
            pa.field("page_path", pa.string()),
            pa.field("report_heading", pa.string()),
            pa.field("finding_index", pa.int64()),
            pa.field("finding_text", pa.string()),
            pa.field("finding_hash", pa.string()),
            pa.field("visibility_class", pa.string()),
            pa.field("embedding_text", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), list_size=embedding_dimension)),
            pa.field("metadata", pa.string()),
        ]
    )


class DocsAuditEnterpriseStore:
    def __init__(
        self,
        *,
        uri: str,
        api_key: str,
        host_override: str,
        region: str,
    ) -> None:
        if not api_key:
            raise RuntimeError("Missing LANCEDB_API_KEY")
        self._validate_host_override(host_override)
        self.db = lancedb.connect(
            uri=uri,
            api_key=api_key,
            host_override=host_override,
            region=region,
        )

    @staticmethod
    def _validate_host_override(host_override: str) -> None:
        if not host_override:
            raise RuntimeError("Missing LANCEDB_HOST_OVERRIDE")
        parsed = urlparse(host_override)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RuntimeError("Invalid LANCEDB_HOST_OVERRIDE")

    def ensure_tables(self, *, embedding_dimension: int) -> None:
        self._create_table_ready(RUNS_TABLE, RUNS_SCHEMA)
        self._create_table_ready(FINDINGS_TABLE, findings_schema(embedding_dimension))

    def replace_run(
        self,
        *,
        run_row: dict[str, Any],
        finding_rows: list[dict[str, Any]],
    ) -> dict[str, int]:
        self._create_table_ready(RUNS_TABLE, RUNS_SCHEMA)
        runs = self._open_table(RUNS_TABLE)
        findings = None
        if finding_rows:
            embedding_dimension = len(finding_rows[0]["embedding"])
            self.ensure_tables(embedding_dimension=embedding_dimension)
            findings = self._open_table(FINDINGS_TABLE)
        run_id = str(run_row["run_id"])
        run_id_sql = run_id.replace("'", "''")

        try:
            runs.delete(f"run_id = '{run_id_sql}'")
        except Exception as exc:
            if not self._is_table_not_found_error(exc):
                raise
        if findings is not None:
            try:
                findings.delete(f"run_id = '{run_id_sql}'")
            except Exception as exc:
                if not self._is_table_not_found_error(exc):
                    raise

        runs.add([run_row], mode="append")
        if finding_rows and findings is not None:
            findings.add(finding_rows, mode="append")
        return {"runs": 1, "findings": len(finding_rows)}

    def _create_table_ready(self, table_name: str, schema: pa.Schema) -> None:
        last_error: Exception | None = None
        for _attempt in range(TABLE_READY_MAX_ATTEMPTS):
            try:
                self.db.create_table(table_name, schema=schema, mode="exist_ok")
                return
            except Exception as exc:
                last_error = exc
                if self._is_terminal_table_error(exc):
                    raise RuntimeError(
                        f"Terminal error while ensuring table '{table_name}': {exc}"
                    ) from exc
                time.sleep(TABLE_READY_SLEEP_SECONDS)
        raise RuntimeError(
            f"Timed out ensuring table '{table_name}' after "
            f"{TABLE_READY_TIMEOUT_SECONDS:.0f}s. Last error: {last_error}"
        ) from last_error

    def _open_table(self, table_name: str):
        last_error: Exception | None = None
        for _attempt in range(TABLE_READY_MAX_ATTEMPTS):
            try:
                return self.db.open_table(table_name)
            except Exception as exc:
                last_error = exc
                if self._is_terminal_table_error(exc):
                    raise RuntimeError(
                        f"Terminal error while opening table '{table_name}': {exc}"
                    ) from exc
                time.sleep(TABLE_READY_SLEEP_SECONDS)
        raise RuntimeError(
            f"Timed out opening table '{table_name}' after "
            f"{TABLE_READY_TIMEOUT_SECONDS:.0f}s. Last error: {last_error}"
        ) from last_error

    @staticmethod
    def _is_terminal_table_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        terminal_tokens = (
            "401",
            "403",
            "unauthorized",
            "forbidden",
            "permission denied",
            "invalid api key",
            "invalid url",
            "relativeurlwithoutbase",
            "schema",
            "type mismatch",
            "invalid type",
        )
        transient_tokens = (
            "404",
            "503",
            "table not found",
            "was not found",
            "_versions",
            "service unavailable",
            "temporarily unavailable",
            "retry limit",
            "timed out",
        )
        if any(token in msg for token in transient_tokens):
            return False
        return any(token in msg for token in terminal_tokens)

    @staticmethod
    def _is_table_not_found_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "not found" in msg or "_versions" in msg


def parse_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def json_string(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
