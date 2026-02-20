# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The LanceDB Authors

from pathlib import Path
import shutil


def test_connection():
    # --8<-- [start:connect]
    import lancedb

    uri = "ex_lancedb"
    db = lancedb.connect(uri)
    # --8<-- [end:connect]
    assert db is not None
    shutil.rmtree(uri, ignore_errors=True)
    assert not Path(uri).exists()


async def connect_async_example():
    # --8<-- [start:connect_async]
    import lancedb

    uri = "ex_lancedb"
    async_db = await lancedb.connect_async(uri)
    # --8<-- [end:connect_async]

    return async_db


# --8<-- [start:connect_cloud]
uri = "db://your-database-uri"
api_key = "your-api-key"
region = "us-east-1"
# --8<-- [end:connect_cloud]


# --8<-- [start:connect_cloud_async]
uri = "db://your-database-uri"
api_key = "your-api-key"
region = "us-east-1"
# --8<-- [end:connect_cloud_async]


def connect_object_storage_config():
    # --8<-- [start:connect_object_storage]
    import lancedb

    uri = "s3://your-bucket/path"
    # You can also use "gs://your-bucket/path" or "az://your-container/path".
    db = lancedb.connect(uri)
    # --8<-- [end:connect_object_storage]

    return db


def namespace_table_ops_example():
    # --8<-- [start:namespace_table_ops]
    import lancedb

    db = lancedb.connect("./data/sample-lancedb")
    namespace = ["prod", "search"]

    db.create_table(
        "users",
        data=[{"id": 1, "name": "alice"}],
        mode="overwrite",
        namespace=namespace,
    )

    table = db.open_table("users", namespace=namespace)
    tables = db.list_tables(namespace=namespace).tables

    db.drop_table("users", namespace=namespace)
    # drop_all_tables is namespace-aware as well:
    # db.drop_all_tables(namespace=namespace)
    # --8<-- [end:namespace_table_ops]
    return table, tables


def namespace_admin_ops_example():
    # --8<-- [start:namespace_admin_ops]
    import lancedb

    db = lancedb.connect("./data/sample-lancedb")
    db.create_namespace(["prod"])
    db.create_namespace(["prod", "search"])

    child_namespaces = db.list_namespaces(namespace=["prod"]).namespaces
    metadata = db.describe_namespace(["prod", "search"])

    db.drop_namespace(["prod", "search"], mode="skip")
    db.drop_namespace(["prod"], mode="skip")
    # --8<-- [end:namespace_admin_ops]
    return child_namespaces, metadata

async def connect_object_storage_config_async():
    # --8<-- [start:connect_object_storage_async]
    import lancedb

    uri = "s3://your-bucket/path"
    # You can also use "gs://your-bucket/path" or "az://your-container/path".
    async_db = await lancedb.connect_async(uri)
    # --8<-- [end:connect_object_storage_async]

    return async_db
