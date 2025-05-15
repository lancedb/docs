---
title: Schema Evolution
---

# Schema Evolution with LanceDB Enterprise

Schema evolution enables non-breaking modifications to a database table's structure — such as adding columns, 
altering data types, or dropping fields — to adapt to evolving data requirements without service interruptions. 
LanceDB supports ACID-compliant schema evolution through granular operations (add/alter/drop columns), allowing you to:

* Iterate Safely: Modify schemas in production with versioned datasets and backward compatibility
* Scale Seamlessly: Handle ML model iterations, regulatory changes, or feature additions
* Optimize Continuously: Remove unused fields or enforce new constraints without downtime


## Schema Evolution Operations

LanceDB supports three primary schema evolution operations:

1. **Adding new columns**: Extend your table with additional attributes
2. **Altering existing columns**: Change column names, data types, or nullability
3. **Dropping columns**: Remove unnecessary columns from your schema


<Tip>
  Schema evolution operations are applied immediately but do not typically require rewriting all data. 
  However, data type changes may involve more substantial operations.
</Tip>

## Adding New Columns

You can add new columns to a table with the [`add_columns`](https://lancedb.github.io/lancedb/python/python/#lancedb.table.Table.add_columns) 
method in Python or [`addColumns`](https://lancedb.github.io/lancedb/js/classes/Table/#addcolumns) in TypeScript/JavaScript. 
New columns are populated based on SQL expressions you provide.

<CodeGroup>
```Python Python
table_name = "schema_evolution_add_example"

data = [
    {
        "id": 1,
        "name": "Laptop",
        "price": 1200.00,
        "vector": np.random.random(128).tolist(),
    },
    {
        "id": 2,
        "name": "Smartphone",
        "price": 800.00,
        "vector": np.random.random(128).tolist(),
    },
    {
        "id": 3,
        "name": "Headphones",
        "price": 150.00,
        "vector": np.random.random(128).tolist(),
    },
    {
        "id": 4,
        "name": "Monitor",
        "price": 350.00,
        "vector": np.random.random(128).tolist(),
    },
    {
        "id": 5,
        "name": "Keyboard",
        "price": 80.00,
        "vector": np.random.random(128).tolist(),
    },
]

table = db.create_table(table_name, data, mode="overwrite")

# 1. Add a new column calculating a discount price
table.add_columns({"discounted_price": "cast((price * 0.9) as float)"})

# 2. Add a status column with a default value
table.add_columns({"in_stock": "cast(true as boolean)"})

# 3. Add a nullable column with NULL values
table.add_columns({"last_ordered": "cast(NULL as timestamp)"})
```
```Typescript Typescript
const tableName = "schema_evolution_add_example";

const data = [
  {
    id: 1,
    name: "Laptop",
    price: 1200.0,
    vector: Array.from({ length: 128 }, () => Math.random()),
  },
  {
    id: 2,
    name: "Smartphone",
    price: 800.0,
    vector: Array.from({ length: 128 }, () => Math.random()),
  },
  {
    id: 3,
    name: "Headphones",
    price: 150.0,
    vector: Array.from({ length: 128 }, () => Math.random()),
  },
  {
    id: 4,
    name: "Monitor",
    price: 350.0,
    vector: Array.from({ length: 128 }, () => Math.random()),
  },
  {
    id: 5,
    name: "Keyboard",
    price: 80.0,
    vector: Array.from({ length: 128 }, () => Math.random()),
  },
];

const table = await db.createTable(tableName, data, { mode: "overwrite" });

// 1. Add a new column calculating a discount price
await table.addColumns([
  { name: "discounted_price", valueSql: "cast((price * 0.9) as float)" },
]);

// 2. Add a status column with a default value
await table.addColumns([
  { name: "in_stock", valueSql: "cast(true as boolean)" },
]);

// 3. Add a nullable column with NULL values
await table.addColumns([
  { name: "last_ordered", valueSql: "cast(NULL as timestamp)" },
]);
```
</CodeGroup>

<Warning>
  When adding columns that should contain NULL values, be sure to cast the NULL 
  to the appropriate type, e.g., `cast(NULL as timestamp)`.
</Warning>

## Altering Existing Columns

You can alter columns using the [`alter_columns`](https://lancedb.github.io/lancedb/python/python/#lancedb.table.Table.alter_columns)
method in Python or [`alterColumns`](https://lancedb.github.io/lancedb/js/classes/Table/#altercolumns) in TypeScript/JavaScript. This allows you to:

- Rename a column
- Change a column's data type
- Modify nullability (whether a column can contain NULL values)

<CodeGroup>
```Python Python
table_name = "schema_evolution_alter_example"

data = [
    {
        "id": 1,
        "name": "Laptop",
        "price": 1200,
        "discount_price": 1080.0,
        "vector": np.random.random(128).tolist(),
    },
    {
        "id": 2,
        "name": "Smartphone",
        "price": 800,
        "discount_price": 720.0,
        "vector": np.random.random(128).tolist(),
    },
    {
        "id": 3,
        "name": "Headphones",
        "price": 150,
        "discount_price": 135.0,
        "vector": np.random.random(128).tolist(),
    },
    {
        "id": 4,
        "name": "Monitor",
        "price": 350,
        "discount_price": 315.0,
        "vector": np.random.random(128).tolist(),
    },
    {
        "id": 5,
        "name": "Keyboard",
        "price": 80,
        "discount_price": 72.0,
        "vector": np.random.random(128).tolist(),
    },
]
schema = pa.schema(
    {
        "id": pa.int64(),
        "name": pa.string(),
        "price": pa.int32(),
        "discount_price": pa.float64(),
        "vector": pa.list_(pa.float32(), 128),
    }
)

table = db.create_table(table_name, data, schema=schema, mode="overwrite")

# 1. Rename a column
table.alter_columns({"path": "discount_price", "rename": "sale_price"})

# 2. Change a column's data type
table.alter_columns({"path": "price", "data_type": pa.int64()})

# 3. Make a column nullable
table.alter_columns({"path": "name", "nullable": True})

# 4. Multiple changes at once
table.alter_columns(
    {
        "path": "sale_price",
        "rename": "final_price",
        "data_type": pa.float64(),
        "nullable": True,
    }
)
```
```Typescript Typescript
const tableName = "schema_evolution_alter_example";

const data = [
  {
    id: 1,
    name: "Laptop",
    price: 1200,
    discount_price: 1080.0,
    vector: Array.from({ length: 128 }, () => Math.random()),
  },
  {
    id: 2,
    name: "Smartphone",
    price: 800,
    discount_price: 720.0,
    vector: Array.from({ length: 128 }, () => Math.random()),
  },
  {
    id: 3,
    name: "Headphones",
    price: 150,
    discount_price: 135.0,
    vector: Array.from({ length: 128 }, () => Math.random()),
  },
  {
    id: 4,
    name: "Monitor",
    price: 350,
    discount_price: 315.0,
    vector: Array.from({ length: 128 }, () => Math.random()),
  },
  {
    id: 5,
    name: "Keyboard",
    price: 80,
    discount_price: 72.0,
    vector: Array.from({ length: 128 }, () => Math.random()),
  },
];

const table = await db.createTable(tableName, data, { mode: "overwrite" });

// 1. Rename a column
await table.alterColumns([
  {
    path: "discount_price",
    rename: "sale_price",
  },
]);

// 2. Make a column nullable
await table.alterColumns([
  {
    path: "name",
    nullable: true,
  },
]);

// 3. Multiple changes at once
console.log("\nMultiple changes to 'sale_price' column...");
await table.alterColumns([
  {
    path: "sale_price",
    rename: "final_price",
    nullable: true,
  },
]);
```
</CodeGroup>

<Warning>
  Changing data types requires rewriting the column data and may be resource-intensive for large tables.
  Renaming columns or changing nullability is more efficient as it only updates metadata.
</Warning>

## Dropping Columns

You can remove columns using the [`drop_columns`](https://lancedb.github.io/lancedb/python/python/#lancedb.table.Table.drop_columns)
 method in Python or [`dropColumns`] in TypeScript/JavaScript(https://lancedb.github.io/lancedb/js/classes/Table/#altercolumns).

<CodeGroup>
```Python Python
table_name = "schema_evolution_drop_example"

data = [
    {
        "id": 1,
        "name": "Laptop",
        "price": 1200.00,
        "temp_col1": "X",
        "temp_col2": 100,
        "vector": np.random.random(128).tolist(),
    },
    {
        "id": 2,
        "name": "Smartphone",
        "price": 800.00,
        "temp_col1": "Y",
        "temp_col2": 200,
        "vector": np.random.random(128).tolist(),
    },
    {
        "id": 3,
        "name": "Headphones",
        "price": 150.00,
        "temp_col1": "Z",
        "temp_col2": 300,
        "vector": np.random.random(128).tolist(),
    },
]

table = db.create_table(table_name, data, mode="overwrite")

# 1. Drop a single column
table.drop_columns(["temp_col1"])

# 2. Drop multiple columns
table.drop_columns(["temp_col2"])
```
```TypeScript TypeScript
const tableName = "schema_evolution_drop_example";

  const data = [
    {
      id: 1,
      name: "Laptop",
      price: 1200.0,
      temp_col1: "X",
      temp_col2: 100,
      vector: Array.from({ length: 128 }, () => Math.random()),
    },
    {
      id: 2,
      name: "Smartphone",
      price: 800.0,
      temp_col1: "Y",
      temp_col2: 200,
      vector: Array.from({ length: 128 }, () => Math.random()),
    },
    {
      id: 3,
      name: "Headphones",
      price: 150.0,
      temp_col1: "Z",
      temp_col2: 300,
      vector: Array.from({ length: 128 }, () => Math.random()),
    },
  ];

  const table = await db.createTable(tableName, data, { mode: "overwrite" });

  // 1. Drop a single column
  await table.dropColumns(["temp_col1"]);

  // 2. Drop multiple columns
  await table.dropColumns(["temp_col2"]);
```
</CodeGroup>

<Warning>
  Dropping columns cannot be undone. Make sure you have backups or are certain
  before removing columns.
</Warning>

## Vector Column Considerations

Vector columns (used for embeddings) have special considerations. When altering vector columns, you should ensure consistent dimensionality.

### Converting List to FixedSizeList

A common schema evolution task is converting a generic list column to a fixed-size list for performance:

```Python Python
vector_dim = 768  # Your embedding dimension
table.alter_columns(dict(path="embedding", data_type=pa.list_(pa.float32(), vector_dim)))
```
