---
title: "Changelog"
hide:
  - toc
---
## April 2025

### Enhanced Performance and Improved Version Control

#### Features
* **Reduced Commit Conflicts**: Mitigated upsert-induced table conflicts through client/server-side retry mechanisms.
* **New SDK APIs**:
  - `table.tags.create/list/update/delete/checkout`: Enables semantic versioning through intuitive tagging instead of numeric versioning
  - `wait_for_index`: Ensures complete data indexing with configurable `wait_timeout`.
* **Performance Improvements**:
  - **Query Latency**: Eliminated full cache ring scans when nodes < replication factor.
  - **Full-Text Search**: Introduced configurable FTS index prewarming to accelerate search operations. `enterprise`
  - **Table Creation**: Leveraged cached database connections to reduce overhead—ideal for bulk table creation scenarios.
* **Session-Level Object Store Caching**: Shared connection pools via weak references to object stores. Example: 100 tables → 1 S3 connection pool.
* **IVF_PQ float64 support**: Expanded vector indexing to float64 datasets (previously limited to float16/32).
* **Full-text search on string arrays**: Extended full-text search to support string array columns for efficient multi-value keyword search.
* **UI - Enhanced Table Preview**:
  - One-click vector copying from cells
  - Full row expansion on click (no truncation)
  - Consistent right-panel binary data display
* **UI - Direct Page Navigation**:Added page selector for instant access to specific data pages in table preview.

#### Bug Fixes
* **Query Performance**:
  - Fixed hybrid search distance range filtering to properly respect lower and upper bounds. [\[lancedb#2356\]](https://github.com/lancedb/lancedb/pull/2356)
  - Enforced valid `k` parameters to prevent overflow. [\[lancedb#2354\]](https://github.com/lancedb/lancedb/pull/2354)
  - `BETWEEN clause`: Improved BETWEEN query handling to return 0 results when `start` > `end` instead of panicking. [\[lance#3706\]](https://github.com/lancedb/lance/pull/3706)
* **SDK/API Fixes**:
  - Fixed TypeScript SDK support for FTS advanced features (fuzzy search and boosting). [\[lancedb#2314\]](https://github.com/lancedb/lancedb/pull/2314)
  - Added scalar index support for small FixedSizeBinary columns (e.g., UUID). [\[lancedb#2297\]](https://github.com/lancedb/lancedb/pull/2297)
* **Index**:
  - Fixed IVF_PQ index on vector columns with NaN/INFs. [\[lance#3648\]](https://github.com/lancedb/lance/pull/3648)
  - Resolved GPU-based indexing crashes on non-contiguous arrays. [\[lance#3675\]](https://github.com/lancedb/lance/pull/3675) `enterprise`
  - Fixed B-tree index corruption during null remapping. [\[lance#3704\]](https://github.com/lancedb/lance/pull/3704)
* **Cloud Dashboard**: Optimized chart rendering for better visibility in the Cloud usage dashboard.


## March 2025

### Enhanced Full-Text Search and Advanced Query Debugging Features

#### Features
* **Enhanced Full-Text Search (FTS): Fuzzy Search & Boosting Now Available**: it improves user experience with 
resilient, typo-tolerant searches and can surface the most contextually relevant results faster.
* **New SDK APIs**: 
  - `explain_plan`: Diagnose query performance and debug unexpected results by inspecting the execution plan.
  - `analyze_plan`: Analyze query execution metrics to optimize performance and resource usage. Such metrics
   include execution time, number of rows processed, I/O stats, and more.
  - `restore`: Revert to a specific prior version of your dataset and modify it from a verified, stable state.
* **Scalar Indexing for Extended Data Types**: LanceDB now supports scalar 
indexing on UUID columns of FixedSizeBinary type.
* **Binary vector support in TypeScript SDK**: LanceDB's TypeScript SDK now 
natively supports binary vector indexing and querying with production-grade efficiency.
* **Support S3-compatible object store**: Extended LanceDB Enterprise deployment 
to work with S3-compatible object stores, such as Tigris, Minio and etc. `enterprise`

#### Bug fix 
* **Improved Merge insert performance**: Enhanced the merge-insert operation 
to reduce error rates during upsert operations, improving data reliability.[\[lance#3603\]](https://github.com/lancedb/lance/pull/3603).
* **Batch Ingestion Error**: Updated error codes for batch ingestion failures 
from 500 to 409 to accurately reflect resource conflict scenarios.
* **Cloud Signup Workflow Fix**: Resolved an issue where users encountered a 
blank page after organization creation during the Cloud signup process.
* **Table Preview Data Display Issue**: Fixed a bug causing the table preview 
page to show a generic "Something went wrong" error for datasets containing 
`datetime` columns, which previously prevented data inspection.
* **Rate Limit Error Clarity**: Added explicit error messaging for system rate 
limits (e.g., API keys per table), replacing vague notifications that 
confused users. 


## February 2025

### Multivector Search ready and Table data preview available in Cloud UI

#### Features
* **Multivector Search is now live**: documents can be stored 
as contextualized vector lists. Fast multi-vector queries are supported at scale, 
powered by our XTR optimization.
* **`Drop_index` added to SDK**: users can remove unused or outdated indexes from your tables.
* **Explore Your Data at a Glance**: preview sample data from 
any table with a single click. `lancedb-cloud`
* **Search by Project/Table in Cloud UI**: allow users to quickly locate the desired project/table.`lancedb-cloud`


#### Bug fix 
* **FTS stability fix**: Resolved a crash in Full Text Search (FTS) during flat-mode searches.
* **`prefilter` parameter enforcement**: Fixed a bug where the prefilter parameter was not honored in FTS queries.
* **Vector index bounds error**: Addressed an out-of-bounds indexing issue during vector index creation.
* **`distance_range()` compatibility**: Fixed errors when performing vector searches with `distance_range()` on unindexed rows.
* **Error messaging improvements**: Replaced generic HTTP 500 errors with detailed, actionable error messages for easier debugging.


## January 2025

### Support Hamming Distance and GPU based indexing ready

#### Features
  * **Support Hamming distance and binary vector**: Added `hamming` as a distance metric (joining `l2`, `cosine`, `dot`) 
  for binary vector similarity search.
  * **GPU-Accelerated IVF-PQ indexing**: Build IVF-PQ indexes 10x faster. `enterprise`
  * **AWS Graviton 4 & Google Axion build optimizations**: ARM64 SIMD acceleration cuts query costs.
  `enterprise`
  * **float16 Vector Index Supports**: reduce storage size while maintaining search quality.
  * **Self-Serve Cloud Onboarding**: new workflow-based UI guides users for smooth experience. `lancedb-cloud`

#### Bug fix
   * `list_indices` and `index_stats` now always fetch the latest version of the table by default unless a specific version is explicitly provided.
   * **Error message fix**: Improved clarity for cases where `create_index` is called to create a vector index on tables with fewer than 256 rows.
   * **TypeScript SDK fixes**: Resolved an issue where 1createTable()` failed to correctly save embeddings and for `mergeInsert` not utilizing saved embeddings. [lancedb#2065](https://github.com/lancedb/lancedb/pull/2065)
   * **Multi-vector schema inference**: Addressed an issue where the vector column could not be inferred for multi-vector indexes. [lancedb#2026](https://github.com/lancedb/lancedb/pull/2026)
   * **Hybrid search consistency**: Fixed a discrepancy where hybrid search returned distance values inconsistent with standalone vector search. [lancedb#2061](https://github.com/lancedb/lancedb/pull/2061)



## December 2024

### Performant SQL queries at scale and more cost-effective vector search

#### Features
  * **Run SQL with massive datasets**: added [Apache Arrow flight-SQL protocol](https://arrow.apache.org/docs/format/FlightSql.html) 
  to run SQL queries with billions of data and return in seconds. `enterprise`.
  * **Accelerate vector search**: added our Quantinized-IVF algorithm and other optimization techniques to improve QPS/core. `enterprise`
  * **Azure Stack Router Deployment**: route traffic efficiently to serve low query latency. `enterprise`
  * **Distance range filtering**. filter query results using `distance_range()` to return search results with a 
  lowerbound, upperbound or a range [\[lance#3326\]](https://github.com/lancedb/lance/pull/3326).
  * **Full-Text Search(FTS) indexing options**: configure tokenizers, stopword lists and more at FTS index creation. 

#### Bug fix
  * **Full-text search parameters**: Fixed an issue where full-text search index configurations were not applied correctly. [lancedb#1928](https://github.com/lancedb/lancedb/pull/1928)
  * **Float16 vector queries**: Addressed a bug preventing the use of lists of Float16 values in vector queries. [lancedb#1931](https://github.com/lancedb/lancedb/pull/1931)
  * **Versioned `checkout` API**: Resolved inconsistencies in the `checkout` method when specifying the `version` parameter. [lancedb#1988](https://github.com/lancedb/lancedb/pull/1988)
  * **Table recreation error**: Fixed an issue where dropping and recreating a table with the same name resulted in a table creation error.
