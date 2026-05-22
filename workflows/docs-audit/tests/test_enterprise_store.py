from __future__ import annotations

import pytest

from docs_audit import enterprise_store
from docs_audit.enterprise_store import DocsAuditEnterpriseStore


def test_store_requires_api_key() -> None:
    with pytest.raises(RuntimeError, match="Missing LANCEDB_API_KEY"):
        DocsAuditEnterpriseStore(
            uri="db://docs-audit",
            api_key="",
            host_override="https://enterprise.example.com",
            region="us-east-1",
        )


def test_store_rejects_invalid_host_override() -> None:
    with pytest.raises(RuntimeError, match="Invalid LANCEDB_HOST_OVERRIDE"):
        DocsAuditEnterpriseStore(
            uri="db://docs-audit",
            api_key="key",
            host_override="enterprise.example.com",
            region="us-east-1",
        )


def test_store_uses_enterprise_connection_kwargs(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_connect(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(enterprise_store.lancedb, "connect", fake_connect)

    DocsAuditEnterpriseStore(
        uri="db://docs-audit",
        api_key="enterprise-key",
        host_override="https://enterprise-host.example.com",
        region="us-west-2",
    )

    assert captured == {
        "uri": "db://docs-audit",
        "api_key": "enterprise-key",
        "host_override": "https://enterprise-host.example.com",
        "region": "us-west-2",
    }

