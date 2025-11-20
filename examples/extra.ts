// --8<-- [start:libraries]
import * as lancedb from "@lancedb/lancedb";
import * as arrow from "apache-arrow";
// --8<-- [end:libraries]

// --8<-- [start:install]
npm install @lancedb/lancedb
// --8<-- [end:install]

// --8<-- [start:install_preview]
npm install @lancedb/lancedb@preview
// --8<-- [end:install_preview]

// --8<-- [start:install_preview]
npm install @lancedb/lancedb@preview
// --8<-- [end:install_preview]

// --8<-- [start:connect_cloud]
const dbUri = process.env.LANCEDB_URI || 'db://your-database-uri';
const apiKey = process.env.LANCEDB_API_KEY;
const region = process.env.LANCEDB_REGION;
// --8<-- [end:connect_cloud]
