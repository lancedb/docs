// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors
import { expect, test } from "@jest/globals";
import * as lancedb from "@lancedb/lancedb";
import { withTempDirectory } from "./util.ts";

test("quickstart example (async)", async () => {
  await withTempDirectory(async (databaseDir) => {
    const db = await lancedb.connect(databaseDir);

    // --8<-- [start:quickstart_create_table]
    const data = [
      { id: "1", text: "knight", vector: [0.9, 0.4, 0.8] },
      { id: "2", text: "ranger", vector: [0.8, 0.4, 0.7] },
      { id: "9", text: "priest", vector: [0.6, 0.2, 0.6] },
      { id: "4", text: "rogue", vector: [0.7, 0.4, 0.7] },
    ];
    let table = await db.createTable("adventurers", data, { mode: "overwrite" });
    // --8<-- [end:quickstart_create_table]
    expect(await table.countRows()).toBe(4);
    await db.dropTable("adventurers");

    // --8<-- [start:quickstart_create_table_no_overwrite]
    table = await db.createTable("adventurers", data);
    // --8<-- [end:quickstart_create_table_no_overwrite]
    expect(await table.countRows()).toBe(4);

    // --8<-- [start:quickstart_vector_search_1]
    // Let's search for vectors similar to "warrior"
    let queryVector = [0.8, 0.3, 0.8];

    let result = await table.search(queryVector).limit(2).toArray();
    console.table(result);
    // --8<-- [end:quickstart_vector_search_1]
    expect(result[0].text).toBe("knight");

    // --8<-- [start:quickstart_output_array]
    result = await table.search(queryVector).limit(2).toArray();
    console.table(result);
    // --8<-- [end:quickstart_output_array]
    expect(result[0].text).toBe("knight");

    // --8<-- [start:quickstart_open_table]
    table = await db.openTable("adventurers");
    // --8<-- [end:quickstart_open_table]

    // --8<-- [start:quickstart_add_data]
    const moreData = [
      { id: "7", text: "mage", vector: [0.6, 0.3, 0.4] },
      { id: "8", text: "bard", vector: [0.3, 0.8, 0.4] },
    ];

    // Add data to table
    await table.add(moreData);
    // --8<-- [end:quickstart_add_data]
    expect(await table.countRows()).toBe(6);

    // --8<-- [start:quickstart_vector_search_2]
    // Let's search for vectors similar to "wizard"
    queryVector = [0.7, 0.3, 0.5];

    const results = await table.search(queryVector).limit(2).toArray();
    console.table(results);
    // --8<-- [end:quickstart_vector_search_2]
    expect(results[0].text).toBe("mage");
  });
});
