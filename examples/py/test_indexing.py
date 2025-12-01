# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The LanceDB Authors

import random
import string
import uuid

import lancedb
import numpy as np
import pyarrow as pa


def _make_vector_rows(count: int, dim: int, column: str = "vector"):
    return [
        {column: np.random.random(dim).astype(np.float32).tolist(), "id": i}
        for i in range(count)
    ]


def test_vector_index_configure_ivf(tmp_db):
    table = tmp_db.create_table(
        "vector_index_configure_ivf",
        _make_vector_rows(512, 4),
        mode="overwrite",
    )

    # --8<-- [start:vector_index_configure_ivf]
    table.create_index(metric="l2", num_partitions=16, num_sub_vectors=4)
    # --8<-- [end:vector_index_configure_ivf]

    assert table.list_indices()


def test_vector_index_setup(tmp_db):
    tmp_db.create_table(
        "vector_index_setup",
        _make_vector_rows(8, 4),
        mode="overwrite",
    )

    # --8<-- [start:vector_index_setup]
    import lancedb

    db = tmp_db
    table_name = "vector_index_setup"
    table = db.open_table(table_name)
    # --8<-- [end:vector_index_setup]

    assert table.name == table_name


def test_vector_index_build_ivf(tmp_db):
    table = tmp_db.create_table(
        "vector_index_build_ivf",
        _make_vector_rows(512, 4, column="keywords_embeddings"),
        mode="overwrite",
    )

    # --8<-- [start:vector_index_build_ivf]
    table.create_index(
        metric="cosine",
        vector_column_name="keywords_embeddings",
    )
    # --8<-- [end:vector_index_build_ivf]

    assert table.list_indices()


def test_vector_index_query_ivf(tmp_db):
    dim = 1536
    data = [
        {"id": i, "keywords_embeddings": np.random.random(dim).tolist()}
        for i in range(512)
    ]
    table = tmp_db.create_table(
        "vector_index_query_ivf", data, mode="overwrite"
    )
    table.create_index(
        metric="cosine",
        vector_column_name="keywords_embeddings",
    )

    # --8<-- [start:vector_index_query_ivf]
    tbl = table
    tbl.search(np.random.random((1536))) \
        .limit(2) \
        .nprobes(20) \
        .refine_factor(10) \
        .to_pandas()
    # --8<-- [end:vector_index_query_ivf]


def test_vector_index_hnsw(tmp_db):
    table = tmp_db.create_table(
        "vector_index_hnsw",
        _make_vector_rows(64, 16),
        mode="overwrite",
    )

    # --8<-- [start:vector_index_build_hnsw]
    table.create_index(index_type="IVF_HNSW_SQ")
    # --8<-- [end:vector_index_build_hnsw]

    # --8<-- [start:vector_index_query_hnsw]
    tbl = table
    tbl.search(np.random.random((16))) \
        .limit(2) \
        .to_pandas()
    # --8<-- [end:vector_index_query_hnsw]


def test_vector_index_binary(tmp_db):
    table_name = "test-hamming"
    ndim = 256
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("vector", pa.list_(pa.uint8(), ndim // 8)),
        ]
    )

    # --8<-- [start:vector_index_binary_schema]
    table = tmp_db.create_table(table_name, schema=schema, mode="overwrite")
    # --8<-- [end:vector_index_binary_schema]

    data = []
    for i in range(64):
        vector = np.random.randint(0, 2, size=ndim)
        vector = np.packbits(vector)
        data.append({"id": i, "vector": vector})

    # --8<-- [start:vector_index_binary_add_data]
    table.add(data)
    # --8<-- [end:vector_index_binary_add_data]

    # --8<-- [start:vector_index_binary_build_index]
    table.create_index(
        metric="hamming",
        vector_column_name="vector",
        index_type="IVF_FLAT",
    )
    # --8<-- [end:vector_index_binary_build_index]

    # --8<-- [start:vector_index_binary_search]
    query = np.random.randint(0, 2, size=ndim)
    query = np.packbits(query)
    df = table.search(query).metric("hamming").limit(10).to_pandas()
    df.vector = df.vector.apply(np.unpackbits)
    # --8<-- [end:vector_index_binary_search]

    assert not df.empty


def test_vector_index_check_status(tmp_db):
    table = tmp_db.create_table(
        "vector_index_check_status",
        _make_vector_rows(512, 8, column="keywords_embeddings"),
        mode="overwrite",
    )
    table.create_index(
        metric="cosine",
        vector_column_name="keywords_embeddings",
    )

    # --8<-- [start:vector_index_check_status]
    import time

    index_name = "keywords_embeddings_idx"
    table.wait_for_index([index_name])
    print(table.index_stats(index_name))
    # --8<-- [end:vector_index_check_status]


def test_scalar_index_build(tmp_db):
    table = tmp_db.create_table(
        "scalar_index_build",
        [
            {"book_id": 1, "publisher": "A", "vector": [0.1, 0.2]},
            {"book_id": 2, "publisher": "B", "vector": [0.2, 0.3]},
        ],
        mode="overwrite",
    )

    # --8<-- [start:scalar_index_build]
    import lancedb

    db = tmp_db
    tbl = db.open_table("scalar_index_build")
    tbl.create_scalar_index("book_id")
    tbl.create_scalar_index("publisher", index_type="BITMAP")
    # --8<-- [end:scalar_index_build]

    assert tbl.list_indices()


def test_scalar_index_wait(tmp_db):
    table = tmp_db.create_table(
        "scalar_index_wait",
        [{"label": "fiction"}],
        mode="overwrite",
    )
    table.create_scalar_index("label")

    # --8<-- [start:scalar_index_wait]
    index_name = "label_idx"
    table.wait_for_index([index_name])
    # --8<-- [end:scalar_index_wait]


def test_scalar_index_optimize(tmp_db):
    table = tmp_db.create_table(
        "scalar_index_optimize",
        [{"vector": [7.0, 8.0], "book_id": 3}],
        mode="overwrite",
    )

    # --8<-- [start:scalar_index_optimize]
    table.add([{"vector": [7, 8], "book_id": 4}])
    table.optimize()
    # --8<-- [end:scalar_index_optimize]


def test_scalar_index_filter(tmp_db):
    table = tmp_db.create_table(
        "books",
        [
            {"vector": [1.1, 1.2], "book_id": 1},
            {"vector": [2.1, 2.2], "book_id": 2},
        ],
        mode="overwrite",
    )

    # --8<-- [start:scalar_index_filter]
    import lancedb

    db = tmp_db
    table = db.open_table("books")
    result = table.search().where("book_id = 2").limit(10).to_pandas()
    # --8<-- [end:scalar_index_filter]

    assert len(result) == 1


def test_scalar_index_prefilter(tmp_db):
    table = tmp_db.create_table(
        "book_with_embeddings",
        [
            {"vector": [1.2, 1.3], "book_id": 1},
            {"vector": [4.2, 4.3], "book_id": 2},
        ],
        mode="overwrite",
    )

    # --8<-- [start:scalar_index_prefilter]
    import lancedb

    db = tmp_db
    table = db.open_table("book_with_embeddings")
    table.search([1.2] * 2) \
        .where("book_id != 3") \
        .limit(10) \
        .to_pandas()
    # --8<-- [end:scalar_index_prefilter]


def test_scalar_index_uuid(tmp_db):
    # --8<-- [start:scalar_index_uuid_type]
    import pyarrow as pa

    class UuidType(pa.ExtensionType):
        def __init__(self):
            super().__init__(pa.binary(16), "my.uuid")

        def __arrow_ext_serialize__(self):
            return b"uuid-metadata"

        @classmethod
        def __arrow_ext_deserialize__(cls, storage_type, serialized):
            return UuidType()

    pa.register_extension_type(UuidType())
    # --8<-- [end:scalar_index_uuid_type]

    # --8<-- [start:scalar_index_uuid_data]
    import uuid

    def generate_random_string(length=10):
        charset = string.ascii_letters + string.digits
        return "".join(random.choices(charset, k=length))

    def generate_uuids(num_items):
        return [uuid.uuid4().bytes for _ in range(num_items)]

    n = 10
    uuids = generate_uuids(n)
    names = [generate_random_string() for _ in range(n)]
    # --8<-- [end:scalar_index_uuid_data]

    # --8<-- [start:scalar_index_uuid_table]
    import lancedb

    db = tmp_db
    uuid_array = pa.array(uuids, pa.binary(16))
    name_array = pa.array(names, pa.string())
    extension_array = pa.ExtensionArray.from_storage(UuidType(), uuid_array)
    schema = pa.schema(
        [
            pa.field("id", UuidType()),
            pa.field("name", pa.string()),
        ]
    )
    data_table = pa.Table.from_arrays([extension_array, name_array], schema=schema)
    table_name = "index-on-uuid-test"
    table = db.create_table(table_name, data=data_table, mode="overwrite")
    # --8<-- [end:scalar_index_uuid_table]

    # --8<-- [start:scalar_index_uuid_wait]
    table.create_scalar_index("id")
    index_name = "id_idx"
    table.wait_for_index([index_name])
    # --8<-- [end:scalar_index_uuid_wait]

    # --8<-- [start:scalar_index_uuid_upsert]
    new_users = [
        {"id": uuid.uuid4().bytes, "name": "Bobby"},
        {"id": uuid.uuid4().bytes, "name": "Charlie"},
    ]

    table.merge_insert("id") \
        .when_matched_update_all() \
        .when_not_matched_insert_all() \
        .execute(new_users)
    # --8<-- [end:scalar_index_uuid_upsert]


def test_fts_index_create(tmp_db):
    table = tmp_db.create_table(
        "fts_index_create",
        [{"text": "hello world", "vector": [0.1, 0.2]}],
        mode="overwrite",
    )

    # --8<-- [start:fts_index_create]
    import lancedb

    db = tmp_db
    table_name = "fts_index_create"
    table = db.open_table(table_name)
    table.create_fts_index("text")
    # --8<-- [end:fts_index_create]

    assert table.list_indices()


def test_fts_index_wait(tmp_db):
    table = tmp_db.create_table(
        "fts_index_wait",
        [{"text": "full text search"}],
        mode="overwrite",
    )
    table.create_fts_index("text")

    # --8<-- [start:fts_index_wait]
    index_name = "text_idx"
    table.wait_for_index([index_name])
    # --8<-- [end:fts_index_wait]


def test_gpu_index_snippets(tmp_db, monkeypatch):
    table = tmp_db.create_table(
        "gpu_index",
        _make_vector_rows(32, 8),
        mode="overwrite",
    )

    calls = []

    def fake_create_index(*args, **kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr(table, "create_index", fake_create_index)

    # --8<-- [start:gpu_index_cuda]
    table.create_index(
        num_partitions=256,
        num_sub_vectors=96,
        accelerator="cuda",
    )
    # --8<-- [end:gpu_index_cuda]

    # --8<-- [start:gpu_index_mps]
    table.create_index(
        num_partitions=256,
        num_sub_vectors=96,
        accelerator="mps",
    )
    # --8<-- [end:gpu_index_mps]

    assert calls[0]["accelerator"] == "cuda"
    assert calls[1]["accelerator"] == "mps"


def test_reindexing_incremental(tmp_db):
    table = tmp_db.create_table(
        "reindexing_incremental",
        [{"vector": [3.1, 4.1], "text": "Frodo was a happy puppy"}],
        mode="overwrite",
    )

    # --8<-- [start:reindexing_incremental]
    import lancedb

    db = tmp_db
    table = db.open_table("reindexing_incremental")
    table.add([{"vector": [3.1, 4.1], "text": "Frodo was a happy puppy"}])
    table.optimize()
    # --8<-- [end:reindexing_incremental]

