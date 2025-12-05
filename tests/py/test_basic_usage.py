# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The LanceDB Authors

from random import randint, random

# --8<-- [start:basic_imports]
import lancedb
import pandas as pd
import polars as pl
import pyarrow as pa

# --8<-- [end:basic_imports]
import pytest


def test_basic_usage(db_path_factory):
    uri = "./basic_usage_db"
    uri = db_path_factory("basic_usage_db")
    db = lancedb.connect(uri)

    # --8<-- [start:basic_create_table]
    data = [
        {"id": "1", "text": "unicorn", "vector": [0.4, 0.1, 0.7]},
        {"id": "2", "text": "dragon", "vector": [0.8, 0.7, 0.1]},
        {"id": "3", "text": "centaur", "vector": [0.3, 0.3, 0.6]},
    ]

    table = db.create_table("creatures", data=data, mode="overwrite")
    # --8<-- [end:basic_create_table]
    assert len(table) == 4

    # --8<-- [start:basic_create_table_pandas]
    pandas_df = pd.DataFrame(data)
    db.create_table("camp_roster", data=pandas_df, mode="overwrite")
    # --8<-- [end:basic_create_table_pandas]

    # --8<-- [start:basic_create_empty_table]
    schema = pa.schema([pa.field("vector", pa.list_(pa.float32(), list_size=3))])
    db.create_table("empty_table", schema=schema, mode="overwrite")
    # --8<-- [end:basic_create_empty_table]
    # --8<-- [start:basic_open_table]
    table = db.open_table("creatures")
    # --8<-- [end:basic_open_table]
    # --8<-- [start:basic_table_names]
    assert "creatures" in db.table_names()
    # --8<-- [end:basic_table_names]
    # --8<-- [start:basic_add_data]
    recruits = [
        {"id": "7", "text": "mage", "vector": [0.6, 0.3, 0.4], "level": 11},
        {"id": "8", "text": "bard", "vector": [0.3, 0.8, 0.4], "level": 9},
    ]
    table.add(recruits)

    df = pd.DataFrame(recruits)
    table.add(df)
    # --8<-- [end:basic_add_data]
    # --8<-- [start:basic_vector_search]
    query_vector = [0.8, 0.3, 0.8]
    table.search(query_vector).limit(2).to_pandas()
    # --8<-- [end:basic_vector_search]
    table.add(
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
    # --8<-- [start:basic_add_columns]
    table.add_columns({"power_score": "cast((level * 1.5) as float)"})
    # --8<-- [end:basic_add_columns]

    # --8<-- [start:basic_drop_columns]
    table.drop_columns(["power_level"])
    # --8<-- [end:basic_drop_columns]
    # --8<-- [start:basic_delete_rows]
    table.delete('text = "rogue"')
    # --8<-- [end:basic_delete_rows]
    # --8<-- [start:basic_drop_table]
    db.drop_table("creatures")
    # --8<-- [end:basic_drop_table]


@pytest.mark.asyncio
async def test_working_with_tables_async(db_path_factory):
    uri = db_path_factory("adventure-db-async")
    # --8<-- [start:basic_async_connect]
    db = await lancedb.connect_async(str(uri))
    # --8<-- [end:basic_async_connect]
    # --8<-- [start:basic_async_create_table]
    data = [
        {"id": "1", "text": "knight", "vector": [0.9, 0.4, 0.8], "level": 12},
        {"id": "2", "text": "ranger", "vector": [0.8, 0.4, 0.7], "level": 10},
        {"id": "9", "text": "priest", "vector": [0.6, 0.2, 0.6], "level": 8},
        {"id": "4", "text": "rogue", "vector": [0.7, 0.4, 0.7], "level": 9},
    ]

    table = await db.create_table("adventurers_async", data=data, mode="overwrite")
    # --8<-- [end:basic_async_create_table]
    # --8<-- [start:basic_async_create_table_pandas]
    df = pd.DataFrame(
        [
            {"id": "1", "text": "knight", "vector": [0.9, 0.4, 0.8], "level": 12},
            {"id": "2", "text": "ranger", "vector": [0.8, 0.4, 0.7], "level": 10},
            {"id": "9", "text": "priest", "vector": [0.6, 0.2, 0.6], "level": 8},
            {"id": "4", "text": "rogue", "vector": [0.7, 0.4, 0.7], "level": 9},
        ]
    )

    await db.create_table("camp_roster_async", data=df, mode="overwrite")
    # --8<-- [end:basic_async_create_table_pandas]
    # --8<-- [start:basic_async_create_empty_table]
    schema = pa.schema([pa.field("vector", pa.list_(pa.float32(), list_size=3))])
    await db.create_table("empty_table_async", schema=schema, mode="overwrite")
    # --8<-- [end:basic_async_create_empty_table]
    # --8<-- [start:basic_async_open_table]
    table = await db.open_table("adventurers_async")
    # --8<-- [end:basic_async_open_table]
    # --8<-- [start:basic_async_table_names]
    assert "adventurers_async" in await db.table_names()
    # --8<-- [end:basic_async_table_names]
    # --8<-- [start:basic_async_add_data]
    # Option 1: Add a list of dicts to a table
    data = [
        {"id": "7", "text": "mage", "vector": [0.6, 0.3, 0.4], "level": 11},
        {"id": "8", "text": "bard", "vector": [0.3, 0.8, 0.4], "level": 9},
    ]
    await table.add(data)

    # Option 2: Add a pandas DataFrame to a table
    df = pd.DataFrame(data)
    await table.add(df)
    # --8<-- [end:basic_async_add_data]
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
    await table.add(data)
    # --8<-- [start:basic_async_vector_search]
    await table.vector_search([0.8, 0.3, 0.8]).limit(2).to_polars()
    # --8<-- [end:basic_async_vector_search]
    # --8<-- [start:basic_async_add_columns]
    await table.add_columns({"power_score": "cast((level * 1.5) as float)"})
    # --8<-- [end:basic_async_add_columns]
    # --8<-- [start:basic_async_alter_columns]
    await table.alter_columns(
        {
            "path": "power_score",
            "rename": "power_level",
            "data_type": pa.float64(),
            "nullable": True,
        }
    )
    # --8<-- [end:basic_async_alter_columns]
    # --8<-- [start:basic_async_alter_columns_vector]
    await table.alter_columns(
        {
            "path": "vector",
            "data_type": pa.list_(pa.float16(), list_size=3),
        }
    )
    # --8<-- [end:basic_async_alter_columns_vector]
    # Change it back since we can get a panic with fp16
    await table.alter_columns(
        {
            "path": "vector",
            "data_type": pa.list_(pa.float32(), list_size=3),
        }
    )
    # --8<-- [start:basic_async_drop_columns]
    await table.drop_columns(["power_level"])
    # --8<-- [end:basic_async_drop_columns]
    await table.vector_search([0.7, 0.3, 0.5]).limit(2).to_pandas()
    # --8<-- [start:basic_async_delete_rows]
    await table.delete('text = "rogue"')
    # --8<-- [end:basic_async_delete_rows]
    # --8<-- [start:basic_async_drop_table]
    await db.drop_table("adventurers_async")
    # --8<-- [end:basic_async_drop_table]
