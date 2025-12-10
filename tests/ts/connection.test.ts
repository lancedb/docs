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

void [uri, apiKey, region];
