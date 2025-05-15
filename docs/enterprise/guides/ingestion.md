---
title: Ingest data
---

# Ingesting Data with LanceDB Enterprise

We support high-throughput writes, comfortably handling 4GB per second. 
Our client SDK maintains 1:1 parity with the open-source version, 
enabling existing users to migrate seamlessly—zero refactoring required.

LanceDB supports table creation using multiple data formats, including:
* Pandas DataFrames (example below)
* [Polars](https://pola.rs/) DataFrames
* Apache Arrow Tables

For the Python SDK, you can also define tables flexibly using:
* PyArrow schemas (for explicit schema control)
* `LanceModel` (a **Pydantic**-based model for structured data validation and serialization)

This ensures compatibility with modern data workflows while maintaining performance and type safety.

## Insert data

=== "Python"
    ```python
    import lancedb
    import pyarrow as pa

    # connect to LanceDB Cloud
    db = lancedb.connect(
      uri="db://your-project-slug",
      api_key="your-api-key",
      region="us-east-1"
    )

    # create an empty table with schema
    data = [
        {"vector": [3.1, 4.1], "item": "foo", "price": 10.0},
        {"vector": [5.9, 26.5], "item": "bar", "price": 20.0},
        {"vector": [10.2, 100.8], "item": "baz", "price": 30.0},
        {"vector": [1.4, 9.5], "item": "fred", "price": 40.0},
    ]

    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), 2)),
        pa.field("item", pa.utf8()),
        pa.field("price", pa.float32()),
    ])

    table_name = "basic_ingestion_example"
    table = db.create_table(table_name, schema=schema, mode="overwrite")
    table.add(data)
    ```

=== "TypeScript"
    ```typescript
    import * as lancedb from "@lancedb/lancedb"
    import { Schema, Field, Float32, FixedSizeList, Utf8 } from "apache-arrow";

    const db = await lancedb.connect({
      uri: "db://your-project-slug",
      apiKey: "your-api-key",
      region: "us-east-1"
    });

    console.log("Creating table from JavaScript objects");
    const data = [
        { vector: [3.1, 4.1], item: "foo", price: 10.0 },
        { vector: [5.9, 26.5], item: "bar", price: 20.0 },
        { vector: [10.2, 100.8], item: "baz", price: 30.0},
        { vector: [1.4, 9.5], item: "fred", price: 40.0},
    ]

    const tableName = "js_objects_example";
    const table = await db.createTable(tableName, data, {
        mode: "overwrite"
    });

    console.log("\nCreating a table with a predefined schema then add data to it");
    const tableName = "schema_example";

    // Define schema
    // create an empty table with schema
    const schema = new Schema([
        new Field(
        "vector",
        new FixedSizeList(2, new Field("float32", new Float32())),
        ),
        new Field("item", new Utf8()),
        new Field("price", new Float32()),
    ]);

    // Create an empty table with schema
    const table = await db.createEmptyTable(tableName, schema, {
        mode: "overwrite",
    });

    // Add data to the schema-defined table
    const data = [
        { vector: [3.1, 4.1], item: "foo", price: 10.0 },
        { vector: [5.9, 26.5], item: "bar", price: 20.0 },
        { vector: [10.2, 100.8], item: "baz", price: 30.0},
        { vector: [1.4, 9.5], item: "fred", price: 40.0},
    ]

    await table.add(data);
    ```

!!! info
    The vector column needs to be a pyarrow.FixedSizeList type.

### Using Pydantic Models

```python
from lancedb.pydantic import Vector, LanceModel

import pyarrow as pa

# Define a Pydantic model
class Content(LanceModel):
    movie_id: int
    vector: Vector(128)
    genres: str
    title: str
    imdb_id: int

    @property
    def imdb_url(self) -> str:
        return f"https://www.imdb.com/title/tt{self.imdb_id}"

# Create table with Pydantic model schema
table_name = "pydantic_example"
table = db.create_table(table_name, schema=Content, mode="overwrite")
```

### Using Nested Models

You can use nested Pydantic models to represent complex data structures. 
For example, you may want to store the document string and the document source name as a nested Document object:
```python
from pydantic import BaseModel

class Document(BaseModel):
    content: str
    source: str
```
This can be used as the type of a LanceDB table column:
```python
class NestedSchema(LanceModel):
    id: str
    vector: Vector(128)
    document: Document

# Create table with nested schema
table_name = "nested_model_example"
table = db.create_table(table_name, schema=NestedSchema, mode="overwrite")
```
This creates a struct column called `document` that has two subfields called `content` and `source`:
```
In [28]: table.schema
Out[28]:
id: string not null
vector: fixed_size_list<item: float>[128] not null
    child 0, item: float
document: struct<content: string not null, source: string not null> not null
    child 0, content: string not null
    child 1, source: string not null
```

## Insert large datasets
It is recommended to use itertators to add large datasets in batches when creating 
your table in one go. Data will be automatically compacted for the best query performance.

=== "Python"
    ```python
    import pyarrow as pa

    def make_batches():
        for i in range(5):  # Create 3 batches
            yield pa.RecordBatch.from_arrays(
                [
                    pa.array([[3.1, 4.1], [5.9, 26.5]],
                            pa.list_(pa.float32(), 2)),
                    pa.array([f"item{i*2+1}", f"item{i*2+2}"]),
                    pa.array([float((i*2+1)*10), float((i*2+2)*10)]),
                ],
                ["vector", "item", "price"],
            )

    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), 2)),
        pa.field("item", pa.utf8()),
        pa.field("price", pa.float32()),
    ])
    # Create table with batches
    table_name = "batch_ingestion_example"
    table = db.create_table(table_name, make_batches(), schema=schema, mode="overwrite")
    ```

=== "TypeScript"
    ```typescript
    console.log("\nBatch ingestion example with product catalog data");
    const tableName = "product_catalog";

    // Vector dimension for product embeddings (realistic dimension for text embeddings)
    const vectorDim = 128;

    // Create random embedding vector of specified dimension
    const createRandomEmbedding = (dim: number) => Array(dim).fill(0).map(() => Math.random() * 2 - 1);

    // Create table with initial batch of products
    const initialBatch = Array(10).fill(0).map((_, i) => ({
        product_id: `PROD-${1000 + i}`,
        name: `Product ${i + 1}`,
        category: ["electronics", "home", "office"][i % 3],
        price: 10.99 + (i * 5.99),
        vector: createRandomEmbedding(vectorDim)
    }));

    const table = await db.createTable(tableName, initialBatch, { 
        mode: "overwrite"
    });

    // Second batch - 25 more products
    const batch2 = Array(25).fill(0).map((_, i) => ({
        product_id: `PROD-${2000 + i}`,
        name: `Premium Product ${i + 1}`,
        category: ["electronics", "kitchen", "outdoor", "office", "gaming"][i % 5],
        price: 25.99 + (i * 7.49),
        vector: createRandomEmbedding(vectorDim)
    }));

    await table.add(batch2);

    // Third batch - 15 more products in a different category
    const batch3 = Array(15).fill(0).map((_, i) => ({
        product_id: `PROD-${3000 + i}`,
        name: `Budget Product ${i + 1}`,
        category: ["essentials", "budget", "basics"][i % 3],
        price: 5.99 + (i * 2.50),
        vector: createRandomEmbedding(vectorDim)
    }));

    await table.add(batch3);
    ```

Explore full documentation in our SDK guides: [Python](https://lancedb.github.io/lancedb/python/python/) and [Typescript](https://lancedb.github.io/lancedb/js/globals/).

[^1]:We suggest the best batch size to be 500k