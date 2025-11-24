# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The LanceDB Authors

# --8<-- [start:basic_usage_imports]
import lancedb
import pandas as pd
import pyarrow as pa
# --8<-- [end:basic_usage_imports]

import pytest
from numpy.random import randint, random


# Naming convention: test_<functionality> for test functions that represent a particular
# section in the docs. E.g., basic_usage corresponds to the
# "basic usage" section in the user guide.
def test_basic_usages(example_dir):
    # --8<-- [start:basic_usage_set_uri]
    uri = "./adventure-db"
    # --8<-- [end:basic_usage_set_uri]
    uri = example_dir / "adventure-db"
    # --8<-- [start:basic_usage_connect]
    db = lancedb.connect(str(uri))
    # --8<-- [end:basic_usage_connect]

    # --8<-- [start:basic_usage_create_table]
    data = [
        {"id": "1", "text": "knight", "vector": [0.9, 0.4, 0.8], "level": 12},
        {"id": "2", "text": "ranger", "vector": [0.8, 0.4, 0.7], "level": 10},
        {"id": "9", "text": "priest", "vector": [0.6, 0.2, 0.6], "level": 8},
        {"id": "4", "text": "rogue", "vector": [0.7, 0.4, 0.7], "level": 9},
    ]

    tbl = db.create_table("adventurers", data=data, mode="overwrite")
    # --8<-- [end:basic_usage_create_table]

    # --8<-- [start:basic_usage_create_table_pandas]
    df = pd.DataFrame(data)
    db.create_table("camp_roster", data=df, mode="overwrite")
    # --8<-- [end:basic_usage_create_table_pandas]

    # --8<-- [start:basic_usage_create_empty_table]
    schema = pa.schema([pa.field("vector", pa.list_(pa.float32(), list_size=3))])
    db.create_table("empty_table", schema=schema, mode="overwrite")
    # --8<-- [end:basic_usage_create_empty_table]
    # --8<-- [start:basic_usage_open_table]
    tbl = db.open_table("adventurers")
    # --8<-- [end:basic_usage_open_table]
    # --8<-- [start:basic_usage_table_names]
    assert "adventurers" in db.table_names()
    # --8<-- [end:basic_usage_table_names]
    # --8<-- [start:basic_usage_add_data]
    recruits = [
        {"id": "7", "text": "mage", "vector": [0.6, 0.3, 0.4], "level": 11},
        {"id": "8", "text": "bard", "vector": [0.3, 0.8, 0.4], "level": 9},
    ]
    tbl.add(recruits)

    df = pd.DataFrame(recruits)
    tbl.add(df)
    # --8<-- [end:basic_usage_add_data]
    # --8<-- [start:basic_usage_vector_search]
    query_vector = [0.8, 0.3, 0.8]
    tbl.search(query_vector).limit(2).to_pandas()
    # --8<-- [end:basic_usage_vector_search]
    tbl.add(
        [
            {
                "id": f"scout-{i}",
                "text": "scout",
                "vector": random(3),
                "level": int(randint(15, 25)),
            }
            for i in range(50)
        ]
    )
    # --8<-- [start:basic_usage_add_columns]
    tbl.add_columns({"power_score": "cast((level * 1.5) as float)"})
    # --8<-- [end:basic_usage_add_columns]
    # --8<-- [start:basic_usage_alter_columns]
    tbl.alter_columns(
        {
            "path": "power_score",
            "rename": "power_level",
            "data_type": pa.float64(),
            "nullable": True,
        }
    )
    # --8<-- [end:basic_usage_alter_columns]
    # --8<-- [start:basic_usage_alter_columns_vector]
    tbl.alter_columns(
        {
            "path": "vector",
            "data_type": pa.list_(pa.float16(), list_size=3),
        }
    )
    # --8<-- [end:basic_usage_alter_columns_vector]
    # Change it back since we can get a panic with fp16
    tbl.alter_columns(
        {
            "path": "vector",
            "data_type": pa.list_(pa.float32(), list_size=3),
        }
    )
    # --8<-- [start:basic_usage_drop_columns]
    tbl.drop_columns(["power_level"])
    # --8<-- [end:basic_usage_drop_columns]
    # --8<-- [start:basic_usage_delete_rows]
    tbl.delete('text = "rogue"')
    # --8<-- [end:basic_usage_delete_rows]
    # --8<-- [start:basic_usage_drop_table]
    db.drop_table("adventurers")
    # --8<-- [end:basic_usage_drop_table]


@pytest.mark.asyncio
async def test_working_with_tables_async(example_dir):
    uri = example_dir / "adventure-db-async"
    # --8<-- [start:basic_usage_async_connect]
    db = await lancedb.connect_async(str(uri))
    # --8<-- [end:basic_usage_async_connect]
    # --8<-- [start:basic_usage_async_create_table]
    data = [
        {"id": "1", "text": "knight", "vector": [0.9, 0.4, 0.8], "level": 12},
        {"id": "2", "text": "ranger", "vector": [0.8, 0.4, 0.7], "level": 10},
        {"id": "9", "text": "priest", "vector": [0.6, 0.2, 0.6], "level": 8},
        {"id": "4", "text": "rogue", "vector": [0.7, 0.4, 0.7], "level": 9},
    ]

    tbl = await db.create_table("adventurers_async", data=data, mode="overwrite")
    # --8<-- [end:basic_usage_async_create_table]
    # --8<-- [start:basic_usage_async_create_table_pandas]
    df = pd.DataFrame(
        [
            {"id": "1", "text": "knight", "vector": [0.9, 0.4, 0.8], "level": 12},
            {"id": "2", "text": "ranger", "vector": [0.8, 0.4, 0.7], "level": 10},
            {"id": "9", "text": "priest", "vector": [0.6, 0.2, 0.6], "level": 8},
            {"id": "4", "text": "rogue", "vector": [0.7, 0.4, 0.7], "level": 9},
        ]
    )

    await db.create_table("camp_roster_async", data=df, mode="overwrite")
    # --8<-- [end:basic_usage_async_create_table_pandas]
    # --8<-- [start:basic_usage_async_create_empty_table]
    schema = pa.schema([pa.field("vector", pa.list_(pa.float32(), list_size=3))])
    await db.create_table("empty_table_async", schema=schema, mode="overwrite")
    # --8<-- [end:basic_usage_async_create_empty_table]
    # --8<-- [start:basic_usage_async_open_table]
    tbl = await db.open_table("adventurers_async")
    # --8<-- [end:basic_usage_async_open_table]
    # --8<-- [start:basic_usage_async_table_names]
    assert "adventurers_async" in await db.table_names()
    # --8<-- [end:basic_usage_async_table_names]
    # --8<-- [start:basic_usage_async_add_data]
    # Option 1: Add a list of dicts to a table
    data = [
        {"id": "7", "text": "mage", "vector": [0.6, 0.3, 0.4], "level": 11},
        {"id": "8", "text": "bard", "vector": [0.3, 0.8, 0.4], "level": 9},
    ]
    await tbl.add(data)

    # Option 2: Add a pandas DataFrame to a table
    df = pd.DataFrame(data)
    await tbl.add(df)
    # --8<-- [end:basic_usage_async_add_data]
    # Add sufficient data for training
    data = [
        {
            "id": f"scout-{x}",
            "text": "scout",
            "vector": random(3),
            "level": int(randint(15, 25)),
        }
        for x in range(100)
    ]
    await tbl.add(data)
    # --8<-- [start:basic_usage_async_vector_search]
    await tbl.vector_search([0.8, 0.3, 0.8]).limit(2).to_polars()
    # --8<-- [end:basic_usage_async_vector_search]
    # --8<-- [start:basic_usage_async_add_columns]
    await tbl.add_columns({"power_score": "cast((level * 1.5) as float)"})
    # --8<-- [end:basic_usage_async_add_columns]
    # --8<-- [start:basic_usage_async_alter_columns]
    await tbl.alter_columns(
        {
            "path": "power_score",
            "rename": "power_level",
            "data_type": pa.float64(),
            "nullable": True,
        }
    )
    # --8<-- [end:basic_usage_async_alter_columns]
    # --8<-- [start:basic_usage_async_alter_columns_vector]
    await tbl.alter_columns(
        {
            "path": "vector",
            "data_type": pa.list_(pa.float16(), list_size=3),
        }
    )
    # --8<-- [end:basic_usage_async_alter_columns_vector]
    # Change it back since we can get a panic with fp16
    await tbl.alter_columns(
        {
            "path": "vector",
            "data_type": pa.list_(pa.float32(), list_size=3),
        }
    )
    # --8<-- [start:basic_usage_async_drop_columns]
    await tbl.drop_columns(["power_level"])
    # --8<-- [end:basic_usage_async_drop_columns]
    await tbl.vector_search([0.7, 0.3, 0.5]).limit(2).to_pandas()
    # --8<-- [start:basic_usage_async_delete_rows]
    await tbl.delete('text = "rogue"')
    # --8<-- [end:basic_usage_async_delete_rows]
    # --8<-- [start:basic_usage_async_drop_table]
    await db.drop_table("adventurers_async")
    # --8<-- [end:basic_usage_async_drop_table]
