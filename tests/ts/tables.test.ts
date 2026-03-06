// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors
import { expect, test } from "@jest/globals";
import * as arrow from "apache-arrow";
import * as lancedb from "@lancedb/lancedb";
import { withTempDirectory } from "./util.ts";

test("table creation snippets (async)", async () => {
  await withTempDirectory(async (databaseDir) => {
    const db = await lancedb.connect(databaseDir);

    // --8<-- [start:create_table_from_dicts]
    type Location = {
      vector: number[];
      lat: number;
      long: number;
    };

    const data: Location[] = [
      { vector: [1.1, 1.2], lat: 45.5, long: -122.7 },
      { vector: [0.2, 1.8], lat: 40.1, long: -74.1 },
    ];
    const table = await db.createTable("test_table", data, {
      mode: "overwrite",
    });
    // --8<-- [end:create_table_from_dicts]
    expect(await table.countRows()).toBe(2);

    // --8<-- [start:create_table_custom_schema]
    const customSchema = new arrow.Schema([
      new arrow.Field(
        "vector",
        new arrow.FixedSizeList(
          4,
          new arrow.Field("item", new arrow.Float32(), true),
        ),
      ),
      new arrow.Field("lat", new arrow.Float32()),
      new arrow.Field("long", new arrow.Float32()),
    ]);

    const customSchemaData = lancedb.makeArrowTable(
      [
        { vector: [1.1, 1.2, 1.3, 1.4], lat: 45.5, long: -122.7 },
        { vector: [0.2, 1.8, 0.4, 3.6], lat: 40.1, long: -74.1 },
      ],
      { schema: customSchema },
    );
    const customSchemaTable = await db.createTable(
      "my_table_custom_schema",
      customSchemaData,
      { mode: "overwrite" },
    );
    // --8<-- [end:create_table_custom_schema]
    expect(await customSchemaTable.countRows()).toBe(2);

    // --8<-- [start:create_table_from_arrow]
    const arrowSchema = new arrow.Schema([
      new arrow.Field(
        "vector",
        new arrow.FixedSizeList(
          16,
          new arrow.Field("item", new arrow.Float32(), true),
        ),
      ),
      new arrow.Field("text", new arrow.Utf8()),
    ]);
    const arrowData = lancedb.makeArrowTable(
      [
        { vector: Array(16).fill(0.1), text: "foo" },
        { vector: Array(16).fill(0.2), text: "bar" },
      ],
      { schema: arrowSchema },
    );
    const arrowTable = await db.createTable("f32_tbl", arrowData, {
      mode: "overwrite",
    });
    // --8<-- [end:create_table_from_arrow]
    expect(await arrowTable.countRows()).toBe(2);

    // --8<-- [start:create_table_from_iterator]
    const batchSchema = new arrow.Schema([
      new arrow.Field(
        "vector",
        new arrow.FixedSizeList(
          4,
          new arrow.Field("item", new arrow.Float32(), true),
        ),
      ),
      new arrow.Field("item", new arrow.Utf8()),
      new arrow.Field("price", new arrow.Float32()),
    ]);

    const tableForBatches = await db.createEmptyTable(
      "batched_table",
      batchSchema,
      {
        mode: "overwrite",
      },
    );

    const rows = Array.from({ length: 10 }, (_, i) => ({
      vector: [i + 0.1, i + 0.2, i + 0.3, i + 0.4],
      item: `item-${i + 1}`,
      price: (i + 1) * 10,
    }));

    const chunkSize = 2;
    for (let i = 0; i < rows.length; i += chunkSize) {
      const batch = lancedb.makeArrowTable(rows.slice(i, i + chunkSize), {
        schema: batchSchema,
      });
      await tableForBatches.add(batch);
    }
    // --8<-- [end:create_table_from_iterator]
    expect(await tableForBatches.countRows()).toBe(10);

    // --8<-- [start:open_existing_table]
    const openTableData = [{ vector: [1.1, 1.2], lat: 45.5, long: -122.7 }];
    await db.createTable("test_table_open", openTableData, {
      mode: "overwrite",
    });

    console.log(await db.tableNames());

    const openedTable = await db.openTable("test_table_open");
    // --8<-- [end:open_existing_table]
    expect(await openedTable.countRows()).toBe(1);

    // --8<-- [start:create_empty_table]
    const emptySchema = new arrow.Schema([
      new arrow.Field(
        "vector",
        new arrow.FixedSizeList(
          2,
          new arrow.Field("item", new arrow.Float32(), true),
        ),
      ),
      new arrow.Field("item", new arrow.Utf8()),
      new arrow.Field("price", new arrow.Float32()),
    ]);
    const emptyTable = await db.createEmptyTable(
      "test_empty_table",
      emptySchema,
      {
        mode: "overwrite",
      },
    );
    // --8<-- [end:create_empty_table]
    expect(await emptyTable.countRows()).toBe(0);

    // --8<-- [start:drop_table]
    await db.createTable("my_table", [{ vector: [1.1, 1.2], lat: 45.5 }], {
      mode: "overwrite",
    });

    await db.dropTable("my_table");
    // --8<-- [end:drop_table]
    expect(await db.tableNames()).not.toContain("my_table");
  });
});

test("schema evolution snippets (async)", async () => {
  await withTempDirectory(async (databaseDir) => {
    const db = await lancedb.connect(databaseDir);

    // --8<-- [start:schema_add_setup]
    const schemaAddData = [
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
    ];
    const schemaAddTable = await db.createTable(
      "schema_evolution_add_example",
      schemaAddData,
      { mode: "overwrite" },
    );
    // --8<-- [end:schema_add_setup]
    expect(await schemaAddTable.countRows()).toBe(3);

    // --8<-- [start:add_columns_calculated]
    // Add a discounted price column (10% discount)
    await schemaAddTable.addColumns([
      {
        name: "discounted_price",
        valueSql: "cast((price * 0.9) as float)",
      },
    ]);
    // --8<-- [end:add_columns_calculated]

    // --8<-- [start:add_columns_default_values]
    // Add a stock status column with default value
    await schemaAddTable.addColumns([
      {
        name: "in_stock",
        valueSql: "cast(true as boolean)",
      },
    ]);
    // --8<-- [end:add_columns_default_values]

    // --8<-- [start:add_columns_nullable]
    // Add a nullable timestamp column
    await schemaAddTable.addColumns([
      {
        name: "last_ordered",
        valueSql: "cast(NULL as timestamp)",
      },
    ]);
    // --8<-- [end:add_columns_nullable]

    // --8<-- [start:schema_alter_setup]
    const schemaAlter = new arrow.Schema([
      new arrow.Field("id", new arrow.Int64()),
      new arrow.Field("name", new arrow.Utf8()),
      new arrow.Field("price", new arrow.Int32()),
      new arrow.Field("discount_price", new arrow.Float64()),
      new arrow.Field(
        "vector",
        new arrow.FixedSizeList(
          128,
          new arrow.Field("item", new arrow.Float32(), true),
        ),
      ),
    ]);
    const schemaAlterData = lancedb.makeArrowTable(
      [
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
      ],
      { schema: schemaAlter },
    );
    const schemaAlterTable = await db.createTable(
      "schema_evolution_alter_example",
      schemaAlterData,
      { mode: "overwrite" },
    );
    // --8<-- [end:schema_alter_setup]
    expect(await schemaAlterTable.countRows()).toBe(2);

    // --8<-- [start:alter_columns_rename]
    // Rename discount_price to sale_price
    await schemaAlterTable.alterColumns([
      { path: "discount_price", rename: "sale_price" },
    ]);
    // --8<-- [end:alter_columns_rename]

    // --8<-- [start:alter_columns_data_type]
    // Change price from int32 to int64 for larger numbers
    await schemaAlterTable.alterColumns([
      { path: "price", dataType: new arrow.Int64() },
    ]);
    // --8<-- [end:alter_columns_data_type]

    // --8<-- [start:alter_columns_nullable]
    // Make the name column nullable
    await schemaAlterTable.alterColumns([{ path: "name", nullable: true }]);
    // --8<-- [end:alter_columns_nullable]

    // --8<-- [start:alter_columns_multiple]
    // Rename, change type, and make nullable in one operation
    await schemaAlterTable.alterColumns([
      {
        path: "sale_price",
        rename: "final_price",
        dataType: new arrow.Float64(),
        nullable: true,
      },
    ]);
    // --8<-- [end:alter_columns_multiple]

    // --8<-- [start:alter_columns_with_expression]
    // For custom transforms, create a new column from a SQL expression.
    const expressionTable = await db.createTable(
      "schema_evolution_expression_example",
      [{ id: 1, price_text: "$100" }],
      { mode: "overwrite" },
    );

    await expressionTable.addColumns([
      {
        name: "price_numeric",
        valueSql: "cast(replace(price_text, '$', '') as int)",
      },
    ]);
    await expressionTable.dropColumns(["price_text"]);
    await expressionTable.alterColumns([
      { path: "price_numeric", rename: "price" },
    ]);
    // --8<-- [end:alter_columns_with_expression]
    expect(await expressionTable.countRows()).toBe(1);

    // --8<-- [start:schema_drop_setup]
    const schemaDropData = [
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
    const schemaDropTable = await db.createTable(
      "schema_evolution_drop_example",
      schemaDropData,
      { mode: "overwrite" },
    );
    // --8<-- [end:schema_drop_setup]
    expect(await schemaDropTable.countRows()).toBe(3);

    // --8<-- [start:drop_columns_single]
    // Remove the first temporary column
    await schemaDropTable.dropColumns(["temp_col1"]);
    // --8<-- [end:drop_columns_single]

    // --8<-- [start:drop_columns_multiple]
    // Remove the second temporary column
    await schemaDropTable.dropColumns(["temp_col2"]);
    // --8<-- [end:drop_columns_multiple]

    // --8<-- [start:alter_vector_column]
    const oldDim = 384;
    const newDim = 1024;
    const vectorSchema = new arrow.Schema([
      new arrow.Field("id", new arrow.Int64()),
      new arrow.Field(
        "embedding",
        new arrow.FixedSizeList(
          oldDim,
          new arrow.Field("item", new arrow.Float16(), true),
        ),
        true,
      ),
    ]);
    const vectorData = lancedb.makeArrowTable(
      [{ id: 1, embedding: Array.from({ length: oldDim }, () => Math.random()) }],
      { schema: vectorSchema },
    );
    const vectorTable = await db.createTable("vector_alter_example", vectorData, {
      mode: "overwrite",
    });

    // Changing FixedSizeList dimensions (384 -> 1024) is not supported via alterColumns.
    // Use addColumns + dropColumns + alterColumns(rename) to replace the column.
    await vectorTable.addColumns([
      {
        name: "embedding_v2",
        valueSql: `arrow_cast(NULL, 'FixedSizeList(${newDim}, Float16)')`,
      },
    ]);
    await vectorTable.dropColumns(["embedding"]);
    await vectorTable.alterColumns([{ path: "embedding_v2", rename: "embedding" }]);
    // --8<-- [end:alter_vector_column]
    expect(await vectorTable.countRows()).toBe(1);
  });
});
