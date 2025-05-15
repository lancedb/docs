---
title: Update Data
---

# Updating Data with LanceDB Enterprise

Our enterprise solution efficiently manages updates across millions of tables, 
supporting several hundred transactions per second (TPS) per table.

LanceDB supports various data modification operations:

- **Basic Operations**: Update and delete existing data
- **Merge Insert Operations**: Upsert, insert-if-not-exists, and replace range

## Basic Operations

### Update

Update existing rows that match a condition.

<CodeGroup>
```Python Python
import lancedb

# Connect to LanceDB
db = lancedb.connect(
  uri="db://your-project-slug",
  api_key="your-api-key",
  region="us-east-1"
)

data = [
    {"vector": [3.1, 4.1], "item": "foo", "price": 10.0},
    {"vector": [5.9, 26.5], "item": "bar", "price": 20.0},
    {"vector": [2.9, 18.2], "item": "zoo", "price": 30.0},
]

table = db.create_table("update_table_example", data, mode="overwrite")

# Update with direct values
table.update(where="price < 20.0", values={"vector": [2, 2], "item": "foo-updated"})

# Update using SQL expression
table.update(where="price < 20.0", values_sql={"price": "price * 1.1"})
```

```typescript TypeScript
import * as lancedb from "@lancedb/lancedb"

// Connect to LanceDB
const db = await lancedb.connect({
  uri: "db://your-project-slug",
  apiKey: "your-api-key",
  region: "us-east-1"
});

// Create a table with sample data
const data = [
  { vector: [3.1, 4.1], item: "foo", price: 10.0 },
  { vector: [5.9, 26.5], item: "bar", price: 20.0 },
  { vector: [2.9, 18.2], item: "zoo", price: 30.0},
];

const table = await db.createTable("update_table_example", data, { mode: "overwrite" });

// Update with direct values
await table.update({
  where: "price < 20.0",
  values: { vector: [2, 2], item: "foo-updated" },
});

// Update using SQL expression
await table.update({
  where: "price < 20.0", 
  valuesSql: { price: "price * 1.1" },
});
```
</CodeGroup>

<Info>
When rows are updated, LanceDB will update the existing index on the column. 
The updated data is available for search immediately.
</Info>
Check out the details of `update` from [Python SDK](https://lancedb.github.io/lancedb/python/python/#lancedb.table.Table.update) and [TypeScript SDK](https://lancedb.github.io/lancedb/js/classes/Table/#update).

### Delete

Remove rows that match a condition.

<CodeGroup>
```Python Python
import lancedb

# Connect to LanceDB
db = lancedb.connect(
  uri="db://your-project-slug",
  api_key="your-api-key",
  region="us-east-1"
)
table = db.open_table("update_table_example")

# delete data
predicate = "price = 30.0"
table.delete(predicate)
```

```typescript TypeScript
import * as lancedb from "@lancedb/lancedb"

// Connect to LanceDB
const db = await lancedb.connect({
  uri: "db://your-project-slug",
  apiKey: "your-api-key",
  region: "us-east-1"
});
table = await db.openTable("update_table_example");

// delete data
const predicate = "price = 30.0";
await table.delete(predicate);
```
</CodeGroup>

<Warning>
Delete operations are permanent and cannot be undone. Always ensure you have backups or are certain before deleting data.
</Warning>
Check out the details of `delete` from [Python SDK](https://lancedb.github.io/lancedb/python/python/#lancedb.table.Table.delete) and [TypeScript SDK](https://lancedb.github.io/lancedb/js/classes/Table/#delete).


## Merge Operations

The merge insert command is a flexible API that can be used to perform _upsert_, 
_insert_if_not_exists_, and _replace_range_ operations.

<Tip>
  For optimal update performance:
  - Use batch updates when possible (add multiple rows at once)
  - Consider creating a [scalar index](../guides/build-index.md#scalar-index) on your ID column if updating by ID frequently
  - For large-scale updates, consider using multiple concurrent connections
</Tip>

<Info>
**Embedding Functions**

Like the create table and add APIs, the merge insert API will automatically compute embeddings 
if the table has an embedding definition in its schema. If the input data doesn't contain the 
source column, or the vector column is already filled, the embeddings won't be computed.
</Info>

### Upsert

`upsert` updates rows if they exist and inserts them if they don't. To do this with merge insert, 
enable both `when_matched_update_all()` and `when_not_matched_insert_all()`.

<CodeGroup>
```Python Python
import lancedb

# Connect to LanceDB
db = lancedb.connect(
  uri="db://your-project-slug",
  api_key="your-api-key",
  region="us-east-1"
)

# Create example table
users_table_name = "users_example"
table = db.create_table(
    users_table_name,
    [
        {"id": 0, "name": "Alice"},
        {"id": 1, "name": "Bob"},
    ],
    mode="overwrite",
)
print(f"Created users table with {table.count_rows()} rows")

# Prepare data for upsert
new_users = [
    {"id": 1, "name": "Bobby"},  # Will update existing record
    {"id": 2, "name": "Charlie"},  # Will insert new record
]

# Upsert by id
(
    users_table.merge_insert("id")
    .when_matched_update_all()
    .when_not_matched_insert_all()
    .execute(new_users)
)

# Verify results - should be 3 records total
print(f"Total users: {users_table.count_rows()}")  # 3
```

```typescript TypeScript
import * as lancedb from "@lancedb/lancedb"

// Connect to LanceDB
const db = await lancedb.connect({
  uri: "db://your-project-slug",
  apiKey: "your-api-key",
  region: "us-east-1"
});

// Create example table
const table = await db.createTable("users", [
  { id: 0, name: "Alice" },
  { id: 1, name: "Bob" },
]);

// Prepare data for upsert
const newUsers = [
  { id: 1, name: "Bobby" },  // Will update existing record
  { id: 2, name: "Charlie" },  // Will insert new record
];

// Upsert by id
await table
  .mergeInsert("id")
  .whenMatchedUpdateAll()
  .whenNotMatchedInsertAll()
  .execute(newUsers);

// Verify results - should be 3 records total
const count = await table.countRows();
console.log(`Total users: ${count}`);  // 3
```
</CodeGroup>

### Insert-if-not-exists

This will only insert rows that do not have a match in the target table, thus 
preventing duplicate rows. To do this with merge insert, enable just 
`when_not_matched_insert_all()`.

<CodeGroup>
```Python Python
import lancedb

# Connect to LanceDB
db = lancedb.connect(
  uri="db://your-project-slug",
  api_key="your-api-key",
  region="us-east-1"
)

# Create example table
table = db.create_table(
    "domains",
    [
        {"domain": "google.com", "name": "Google"},
        {"domain": "github.com", "name": "GitHub"},
    ],
)

# Prepare new data - one existing and one new record
new_domains = [
    {"domain": "google.com", "name": "Google"},
    {"domain": "facebook.com", "name": "Facebook"},
]

# Insert only if domain doesn't exist
table.merge_insert("domain").when_not_matched_insert_all().execute(new_domains)

# Verify count - should be 3 (original 2 plus 1 new)
print(f"Total domains: {table.count_rows()}")  # 3
```

```typescript TypeScript
import * as lancedb from "@lancedb/lancedb"

// Connect to LanceDB
const db = await lancedb.connect({
  uri: "db://your-project-slug",
  apiKey: "your-api-key",
  region: "us-east-1"
});

// Create example table
const table = await db.createTable(
  "domains", 
  [
    { domain: "google.com", name: "Google" },
    { domain: "github.com", name: "GitHub" },
  ]
);

// Prepare new data - one existing and one new record
const newDomains = [
  { domain: "google.com", name: "Google" },
  { domain: "facebook.com", name: "Facebook" },
];

// Insert only if domain doesn't exist
await table.merge_insert("domain")
  .whenNotMatchedInsertAll()
  .execute(newDomains);

// Verify count - should be 3 (original 2 plus 1 new)
const count = await table.countRows();
console.log(`Total domains: ${count}`);  // 3
```
</CodeGroup>

### Replace range

You can also replace a range of rows in the target table with the input data. 
For example, if you have a table of document chunks, where each chunk has both 
a `doc_id` and a `chunk_id`, you can replace all chunks for a given `doc_id` with updated chunks. 

This can be tricky otherwise because if you try to use `upsert` when the new data has fewer 
chunks you will end up with extra chunks. To avoid this, add another clause to delete any chunks 
for the document that are not in the new data, with `when_not_matched_by_source_delete`.

<CodeGroup>
```Python Python
import lancedb

# Connect to LanceDB
db = lancedb.connect(
  uri="db://your-project-slug",
  api_key="your-api-key",
  region="us-east-1"
)

# Create example table with document chunks
table = db.create_table(
    "chunks",
    [
        {"doc_id": 0, "chunk_id": 0, "text": "Hello"},
        {"doc_id": 0, "chunk_id": 1, "text": "World"},
        {"doc_id": 1, "chunk_id": 0, "text": "Foo"},
        {"doc_id": 1, "chunk_id": 1, "text": "Bar"},
        {"doc_id": 2, "chunk_id": 0, "text": "Baz"},
    ],
)

# New data - replacing all chunks for doc_id 1 with just one chunk
new_chunks = [
    {"doc_id": 1, "chunk_id": 0, "text": "Zoo"},
]

# Replace all chunks for doc_id 1
(
    table.merge_insert(["doc_id"])
    .when_matched_update_all()
    .when_not_matched_insert_all()
    .when_not_matched_by_source_delete("doc_id = 1")
    .execute(new_chunks)
)

# Verify count for doc_id = 1 - should be 2 
print(f"Chunks for doc_id = 1: {table.count_rows('doc_id = 1')}")  # 2
```

```typescript TypeScript
import * as lancedb from "@lancedb/lancedb"

// Connect to LanceDB
const db = await lancedb.connect({
  uri: "db://your-project-slug",
  apiKey: "your-api-key",
  region: "us-east-1"
});

// Create example table with document chunks
const table = await db.createTable(
  "chunks", 
  [
    { doc_id: 0, chunk_id: 0, text: "Hello" },
    { doc_id: 0, chunk_id: 1, text: "World" },
    { doc_id: 1, chunk_id: 0, text: "Foo" },
    { doc_id: 1, chunk_id: 1, text: "Bar" },
    { doc_id: 2, chunk_id: 0, text: "Baz" },
  ]
);

// New data - replacing all chunks for doc_id 1 with just one chunk
const newChunks = [
  { doc_id: 1, chunk_id: 0, text: "Zoo" }
];

// Replace all chunks for doc_id 1
await table.merge_insert(["doc_id"])
  .whenMatchedUpdateAll()
  .whenNotMatchedInsertAll()
  .whenNotMatchedBySourceDelete("doc_id = 1")
  .execute(newChunks);

// Verify count for doc_id = 1 - should be 2 
const count = await table.countRows("doc_id = 1");
console.log(`Chunks for doc_id =1: ${count}`);  // 2
```
</CodeGroup>

For more detailed information, refer to the [`merge_insert` from Python SDK](https://lancedb.github.io/lancedb/python/python/#lancedb.table.Table.merge_insert)
and [`mergeInsert from TypeScript SDK`](https://lancedb.github.io/lancedb/js/classes/Table/#mergeinsert)