// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors
import { expect, test } from "@jest/globals";
import * as path from "node:path";
import { withTempDirectory } from "./util.ts";
// --8<-- [start:connect]
import * as lancedb from "@lancedb/lancedb";

async function connectExample(uri: string) {
  const db = await lancedb.connect(uri);
  return db;
}
// --8<-- [end:connect]

test("connect to a local database", async () => {
  await withTempDirectory(async (tempDir) => {
    const uri = path.join(tempDir, "ex_lancedb");
    const db = await connectExample(uri);
    expect(db).toBeDefined();
  });
});

// --8<-- [start:connect_cloud]
const uri = "db://your-database-uri";
const apiKey = "your-api-key";
const region = "us-east-1";
// --8<-- [end:connect_cloud]

// --8<-- [start:connect_object_storage]
async function connectObjectStorageExample() {
  const uri = "s3://your-bucket/path";
  // You can also use "gs://your-bucket/path" or "az://your-container/path".
  const db = await lancedb.connect(uri);
  return db;
}
// --8<-- [end:connect_object_storage]

async function namespaceTableOpsExample(uri: string) {
  // --8<-- [start:namespace_table_ops]
  const db = await lancedb.connect(uri);
  const namespace = ["prod", "search"];

  await db.createTable(
    "users",
    [{ id: 1, name: "alice" }],
    namespace,
    { mode: "overwrite" },
  );

  const table = await db.openTable("users", namespace);
  const tableNames = await db.tableNames(namespace);

  await db.dropTable("users", namespace);
  // dropAllTables is namespace-aware as well:
  // await db.dropAllTables(namespace);
  // --8<-- [end:namespace_table_ops]
  return { table, tableNames };
}

void [uri, apiKey, region, connectObjectStorageExample, namespaceTableOpsExample];
