# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The LanceDB Authors

import lancedb
import pytest

def test_quickstart(db_path_factory):
    uri = "quickstart_db"
    uri = db_path_factory("quickstart_db")
    db = lancedb.connect(uri)

    # --8<-- [start:quickstart_create_table]
    data = [
        {"id": "1", "text": "knight", "vector": [0.9, 0.4, 0.8]},
        {"id": "2", "text": "ranger", "vector": [0.8, 0.4, 0.7]},
        {"id": "9", "text": "priest", "vector": [0.6, 0.2, 0.6]},
        {"id": "4", "text": "rogue", "vector": [0.7, 0.4, 0.7]},
    ]
    table = db.create_table("adventurers", data=data, mode="overwrite")
    # --8<-- [end:quickstart_create_table]
    assert len(table) == 4
    # Drop the table to test create without overwrite
    db.drop_table("adventurers")

    # --8<-- [start:quickstart_create_table_no_overwrite]
    table = db.create_table("adventurers", data=data)
    # --8<-- [end:quickstart_create_table_no_overwrite]
    assert len(table) == 4
    # --8<-- [start:quickstart_vector_search_1]
    # Let's search for vectors similar to "warrior"
    query_vector = [0.8, 0.3, 0.8]

    # Ensure you run `pip install polars` beforehand
    result = table.search(query_vector).limit(2).to_polars()
    print(result)
    # --8<-- [end:quickstart_vector_search_1]
    assert result.head(1)["text"][0] == "knight"

    # --8<-- [start:quickstart_output_pandas]
    # Ensure you run `pip install pandas` beforehand
    result = table.search(query_vector).limit(2).to_pandas()
    print(result)
    # --8<-- [end:quickstart_output_pandas]
    assert result.iloc[0]["text"] == "knight"

    # --8<-- [start:quickstart_open_table]
    table = db.open_table("adventurers")
    # --8<-- [end:quickstart_open_table]

    # --8<-- [start:quickstart_add_data]
    more_data = [
        {"id": "7", "text": "mage", "vector": [0.6, 0.3, 0.4]},
        {"id": "8", "text": "bard", "vector": [0.3, 0.8, 0.4]},
    ]

    # Add data to table
    table.add(more_data)
    # --8<-- [end:quickstart_add_data]
    assert len(table) == 6

    # --8<-- [start:quickstart_vector_search_2]
    # Let's search for vectors similar to "wizard"
    query_vector = [0.7, 0.3, 0.5]

    results = table.search(query_vector).limit(2).to_polars()
    print(results)
    # --8<-- [end:quickstart_vector_search_2]
    assert results.head(1)["text"][0] == "mage"


@pytest.mark.asyncio
async def test_quickstart_async_api(db_path_factory):
    db_uri = db_path_factory("quickstart_async_db")
    import lancedb
    async_db = await lancedb.connect_async(db_uri)

    data = [
        {"id": "1", "text": "knight", "vector": [0.9, 0.4, 0.8]},
        {"id": "2", "text": "ranger", "vector": [0.8, 0.4, 0.7]},
        {"id": "9", "text": "priest", "vector": [0.6, 0.2, 0.6]},
        {"id": "4", "text": "rogue", "vector": [0.7, 0.4, 0.7]},
    ]

    # --8<-- [start:quickstart_create_table_async]
    async_table = await async_db.create_table(
        "adventurers",
        data=data,
        mode="overwrite",
    )
    # --8<-- [end:quickstart_create_table_async]

    # --8<-- [start:quickstart_vector_search_1_async]
    # Let's search for vectors similar to "warrior"
    query_vector = [0.8, 0.3, 0.8]

    # Ensure you run `pip install polars` beforehand
    async_result = await (await async_table.search(query_vector)).limit(2).to_polars()
    print(async_result)
    # --8<-- [end:quickstart_vector_search_1_async]

    assert async_result.head(1)["text"][0] == "knight"
