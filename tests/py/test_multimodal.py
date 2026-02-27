# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The LanceDB Authors

import pytest
try:
    import lancedb
    import numpy as np
    import pyarrow as pa
    import io
    from PIL import Image
except ImportError:
    pass

# --8<-- [start:multimodal_imports]
import lancedb
import pyarrow as pa
import pandas as pd
import numpy as np
import io
from PIL import Image
# --8<-- [end:multimodal_imports]

def test_multimodal_ingestion(db_path_factory):
    # Ensure dependencies are available
    pytest.importorskip("PIL")
    pytest.importorskip("lancedb")
    pytest.importorskip("numpy")

    # --8<-- [start:create_dummy_data]
    # Create some dummy images
    def create_dummy_image(color):
        img = Image.new('RGB', (100, 100), color=color)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

    # Create dataset with metadata, vectors, and image blobs
    data = [
        {
            "id": 1,
            "filename": "red_square.png",
            "vector": np.random.rand(128).astype(np.float32),
            "image_blob": create_dummy_image('red'),
            "label": "red"
        },
        {
            "id": 2,
            "filename": "blue_square.png",
            "vector": np.random.rand(128).astype(np.float32),
            "image_blob": create_dummy_image('blue'),
            "label": "blue"
        }
    ]
    # --8<-- [end:create_dummy_data]

    # --8<-- [start:define_schema]
    # Define schema explictly to ensure image_blob is treated as binary
    schema = pa.schema([
        pa.field("id", pa.int32()),
        pa.field("filename", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 128)),
        pa.field("image_blob", pa.binary()), # Important: Use pa.binary() for blobs
        pa.field("label", pa.string())
    ])
    # --8<-- [end:define_schema]

    db_uri = db_path_factory("multimodal_db")
    db = lancedb.connect(db_uri)

    # --8<-- [start:ingest_data]
    tbl = db.create_table("images", data=data, schema=schema, mode="overwrite")
    # --8<-- [end:ingest_data]
   
    assert len(tbl) == 2

    # --8<-- [start:search_data]
    # Search for similar images
    query_vector = np.random.rand(128).astype(np.float32)
    results = tbl.search(query_vector).limit(1).to_pandas()
    # --8<-- [end:search_data]

    # --8<-- [start:process_results]
    # Convert back to PIL Image
    for _, row in results.iterrows():
        image_bytes = row['image_blob']
        image = Image.open(io.BytesIO(image_bytes))
        print(f"Retrieved image: {row['filename']}, Size: {image.size}")
        # You can now use 'image' with other libraries or display it
    # --8<-- [end:process_results]
   
    assert len(results) == 1

def test_blob_api_definition(db_path_factory):
    # --8<-- [start:blob_api_schema]
    import pyarrow as pa

    # Define schema with Blob API metadata for lazy loading
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field(
            "video", 
            pa.large_binary(), 
            metadata={"lance-encoding:blob": "true"} # Enable Blob API
        ),
    ])
    # --8<-- [end:blob_api_schema]

    # --8<-- [start:blob_api_ingest]
    import lancedb

    db = lancedb.connect(db_path_factory("blob_db"))
    
    # Create sample data
    data = [
        {"id": 1, "video": b"fake_video_bytes_1"},
        {"id": 2, "video": b"fake_video_bytes_2"}
    ]
    
    # Create the table
    tbl = db.create_table("videos", data=data, schema=schema)
    # --8<-- [end:blob_api_ingest]
    assert len(tbl) == 2
