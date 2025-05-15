---
title: Multivector Search
description: "Working with multiple vectors per document in LanceDB Enterprise"
---

# Multivector Search with LanceDB Enterprise

A complete example for Multivector search is in this [notebook](https://colab.research.google.com/github/lancedb/vectordb-recipes/blob/main/examples/saas_examples/python_notebook/Multivector_on_LanceDB_Cloud.ipynb)

## Multivector Type

LanceDB natively supports multivector data types, enabling advanced search scenarios where 
a single data item is represented by multiple embeddings (e.g., using models like ColBERT 
or CoLPali). In this framework, documents and queries are encoded as collections of 
contextualized vectors—precomputed for documents and indexed for queries.

Key features:
- Indexing on multivector column: store and index multiple vectors per row.
- Supporint query being a single vector or multiple vectors
- Optimized search performance with [XTR](https://arxiv.org/abs/2501.17788) with improved recall.

<Info>
Currently, only the `cosine` metric is supported for multivector search. 
The vector value type can be `float16`, `float32`, or `float64`.
</Info>

## Using Multivector in Python

Currently, multivector search is only support in our Python SDK. 
Below is an example of using multivector search in LanceDB:

```Python Python
import lancedb
import numpy as np
import pyarrow as pa

# Connect to LanceDB Cloud
db = lancedb.connect(
  uri="db://your-project-slug",
  api_key="your-api-key",
  region="your-cloud-region"
)

# Define schema with multivector field
schema = pa.schema(
    [
        pa.field("id", pa.int64()),
        # float16, float32, and float64 are supported
        pa.field("vector", pa.list_(pa.list_(pa.float32(), 256))),
    ]
)

# Create data with multiple vectors per document
data = [
    {
        "id": i,
        "vector": np.random.random(size=(2, 256)).tolist(),  # Each document has 2 vectors
    }
    for i in range(1024)
]

# Create table
tbl = db.create_table("multivector_example", data=data, schema=schema)

# Create index - only cosine similarity is supported for multi-vectors
tbl.create_index(metric="cosine", vector_column_name="vector")

# Query with a single vector
query = np.random.random(256)
results_single = tbl.search(query).limit(5).to_pandas()

# Query with multiple vectors
query_multi = np.random.random(size=(2, 256))
results_multi = tbl.search(query_multi).limit(5).to_pandas()
```