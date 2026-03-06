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
