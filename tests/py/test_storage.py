# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The LanceDB Authors

import lancedb
import pytest


class DummyTable:
    def __init__(self, name: str, storage_options: dict | None = None):
        self.name = name
        self.storage_options = storage_options


class DummyConnection:
    def __init__(self, uri: str, options: dict):
        self.uri = uri
        self.options = options
        self.created_tables: list[DummyTable] = []

    def create_table(self, name: str, data, storage_options: dict | None = None):
        table = DummyTable(name, storage_options=storage_options)
        self.created_tables.append(table)
        return table


@pytest.fixture
def fake_connect(monkeypatch):
    calls: list[DummyConnection] = []

    def _fake_connect(uri: str, **kwargs):
        conn = DummyConnection(uri, kwargs)
        calls.append(conn)
        return conn

    monkeypatch.setattr(lancedb, "connect", _fake_connect)
    return calls


def test_storage_snippets(fake_connect):
    # --8<-- [start:storage_connect_s3]
    db = lancedb.connect("s3://bucket/path")
    # --8<-- [end:storage_connect_s3]

    # --8<-- [start:storage_connect_gcs]
    db = lancedb.connect("gs://bucket/path")
    # --8<-- [end:storage_connect_gcs]

    # --8<-- [start:storage_connect_azure]
    db = lancedb.connect("az://bucket/path")
    # --8<-- [end:storage_connect_azure]

    # --8<-- [start:storage_connect_timeout]
    db = lancedb.connect(
        "s3://bucket/path",
        storage_options={"timeout": "60s"},
    )
    # --8<-- [end:storage_connect_timeout]

    # --8<-- [start:storage_table_timeout]
    table = db.create_table(
        "table",
        [{"a": 1, "b": 2}],
        storage_options={"timeout": "60s"},
    )
    # --8<-- [end:storage_table_timeout]

    # --8<-- [start:storage_s3_ddb]
    db = lancedb.connect(
        "s3+ddb://bucket/path?ddbTableName=my-dynamodb-table",
    )
    # --8<-- [end:storage_s3_ddb]

    # --8<-- [start:storage_s3_minio]
    db = lancedb.connect(
        "s3://bucket/path",
        storage_options={
            "region": "us-east-1",
            "endpoint": "http://minio:9000",
        },
    )
    # --8<-- [end:storage_s3_minio]

    # --8<-- [start:storage_s3_express]
    db = lancedb.connect(
        "s3://my-bucket--use1-az4--x-s3/path",
        storage_options={
            "region": "us-east-1",
            "s3_express": "true",
        },
    )
    # --8<-- [end:storage_s3_express]

    # --8<-- [start:storage_gcs_service_account]
    db = lancedb.connect(
        "gs://my-bucket/my-database",
        storage_options={
            "service_account": "path/to/service-account.json",
        },
    )
    # --8<-- [end:storage_gcs_service_account]

    # --8<-- [start:storage_azure_account]
    db = lancedb.connect(
        "az://my-container/my-database",
        storage_options={
            "account_name": "some-account",
            "account_key": "some-key",
        },
    )
    # --8<-- [end:storage_azure_account]

    # --8<-- [start:storage_tigris_connect]
    db = lancedb.connect(
        "s3://your-bucket/path",
        storage_options={
            "endpoint": "https://t3.storage.dev",
            "region": "auto",
        },
    )
    # --8<-- [end:storage_tigris_connect]

    assert len(fake_connect) == 10


def test_storage_options_provider_snippet():
    """Snippet for StorageOptionsProvider documentation."""
    # --8<-- [start:storage_options_provider]
    from __future__ import annotations

    from typing import Dict

    import lancedb
    from lancedb.io import StorageOptionsProvider

    class MyProvider(StorageOptionsProvider):
        def fetch_storage_options(self) -> Dict[str, str]:
            # Return the same keys you would normally pass via `storage_options`.
            # Example: fetch credentials from your secret manager / STS / metadata service.
            return {
                "aws_access_key_id": "...",
                "aws_secret_access_key": "...",
                "aws_session_token": "...",
                "region": "us-east-1",
            }

    db = lancedb.connect("s3://bucket/path")

    # Table creation
    _ = db.create_table(
        "my_table",
        [{"id": 1, "vector": [0.0, 1.0]}],
        storage_options_provider=MyProvider(),
    )

    # Table open
    _ = db.open_table(
        "my_table",
        storage_options_provider=MyProvider(),
    )
    # --8<-- [end:storage_options_provider]
    assert all(
        conn.uri.startswith(("s3://", "gs://", "az://", "s3+ddb://"))
        for conn in fake_connect
    )
