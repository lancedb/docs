---
title: Understanding Query Performance
---

# Query Optimization with LanceDB Enterprise

LanceDB provides two powerful tools for query analysis and optimization:

- `explain_plan`: Reveals the logical query plan before execution, 
helping you identify potential issues with query structure and index usage. 
This tool is useful for:
  - Verifying query optimization strategies
  - Validating index selection
  - Understanding query execution order
  - Detecting missing indices

- `analyze_plan`: Executes the query and provides detailed runtime metrics, 
including:
  - Operation duration (_elapsed_compute_)
  - Data processing statistics (_output_rows_, _bytes_read_)
  - Index effectiveness (_index_comparisons_, _indices_loaded_)
  - Resource utilization (_iops_, _requests_)

Together, these tools offer a comprehensive view of query performance, 
from planning to execution. Use `explain_plan` to verify your query 
structure and `analyze_plan` to measure and optimize actual performance.

## Reading the Execution Plan

To demonstrate query performance analysis, we'll use a table containing 1.2M rows sampled from the 
[Wikipedia dataset](https://huggingface.co/datasets/wikimedia/wikipedia). Initially, 
the table has no indices, allowing us to observe the impact of optimization.

Let's examine a vector search query that:
- Filters rows where `identifier` is between 0 and 1,000,000
- Returns the top 100 matches
- Projects specific columns: `chunk_index`, `title`, and `identifier`

<CodeGroup>
```Python Python
# explain_plan
query_explain_plan = (
    table.search(query_embed)
    .where("identifier > 0 AND identifier < 1000000")
    .select(["chunk_index", "title", "identifier"])
    .limit(100)
    .explain_plan(True)
)
```

```typescript TypeScript
// explain_plan
const explainPlan = await table
    .search(queryEmbed)
    .where("identifier > 0 AND identifier < 1000000")
    .select(["chunk_index", "title", "identifier"])
    .limit(100)
    .explainPlan(true);
```
</CodeGroup>

The execution plan reveals the sequence of operations performed to execute your query. 
Let's examine each component of the plan:
```python
ProjectionExec: expr=[chunk_index@4 as chunk_index, title@5 as title, identifier@1 as identifier, _distance@3 as _distance]
  RemoteTake: columns="vector, identifier, _rowid, _distance, chunk_index, title"
    CoalesceBatchesExec: target_batch_size=1024
      GlobalLimitExec: skip=0, fetch=100
        FilterExec: _distance@3 IS NOT NULL
          SortExec: TopK(fetch=100), expr=[_distance@3 ASC NULLS LAST], preserve_partitioning=[false]
            KNNVectorDistance: metric=l2
              FilterExec: identifier@1 > 0 AND identifier@1 < 1000000
                LanceScan: uri=***, projection=[vector, identifier], row_id=true, row_addr=false, ordered=false
```

1. **Base Layer (LanceScan)**
   - Initial data scan loading only specified columns to minimize I/O
   - Unordered scan enabling parallel processing
   ```python
   LanceScan: 
   - projection=[vector, identifier]
   - row_id=true, row_addr=false, ordered=false
   ```

2. **First Filter**
   - Apply requested filter on `identifier` column (will reduce the number of vectors that need
   KNN computation)
   ```python
   FilterExec: identifier@1 > 0 AND identifier@1 < 1000000   
   ```

3. **Vector Search**
   - Computes L2 (Euclidean) distances between query vector and all vectors that 
   passed the filter
   ```python
   KNNVectorDistance: metric=l2
   ```

4. **Results Processing**
   - Filters out null distance results, which occur when vectors are NULL or when using cosine 
   distance metric with zero vectors.
   - Sorts by distance and takes top 100 results
   - Processes in batches of 1024 for optimal memory usage
   ```python
   SortExec: TopK(fetch=100)
   - expr=[_distance@3 ASC NULLS LAST]
   - preserve_partitioning=[false]
   FilterExec: _distance@3 IS NOT NULL
   GlobalLimitExec: skip=0, fetch=100
   CoalesceBatchesExec: target_batch_size=1024
   ```

5. **Data Retrieval**
   - `RemoteTake` is a key component of Lance's I/O cache
   - Handles efficient data retrieval from remote storage locations
   - Fetches specific rows and columns (e.g., `chunk_index` and `title`) needed for the final output
   - Optimizes network bandwidth by only retrieving required data
   ```python
   RemoteTake: columns="vector, identifier, _rowid, _distance, chunk_index, title"
   ```

6. **Final Output**
   - Returns only requested columns and maintains column ordering
   ```python
   ProjectionExec: expr=[chunk_index@4 as chunk_index, title@5 as title, identifier@1 as identifier, _distance@3 as _distance]
   ```

This plan demonstrates a basic search without index optimizations: it performs a full 
scan and filter before vector search. 

## Performance Analysis

Let's use `analyze_plan` to run the query and analyze the query performance, 
which will help us identify potential bottlenecks:
<CodeGroup>
```Python Python
# analyze_plan
query_analyze_plan = (
    table.search(query_embed)
    .where("identifier > 0 AND identifier < 1000000")
    .select([ "chunk_index", "title", "identifier"])
    .limit(100)
    .analyze_plan()
)
```
```TypeScript TypeScript
// analyze_plan
const analyzePlan = await table
    .search(queryEmbed)
    .where("identifier > 0 AND identifier < 1000000")
    .select([ "chunk_index", "title", "identifier"])
    .limit(100)
    .analyzePlan();
```
</CodeGroup>
```python
ProjectionExec: expr=[chunk_index@4 as chunk_index, title@5 as title, identifier@1 as identifier, _distance@3 as _distance], metrics=[output_rows=100, elapsed_compute=1.424µs]
    RemoteTake: columns="vector, identifier, _rowid, _distance, chunk_index, title", metrics=[output_rows=100, elapsed_compute=175.53097ms, output_batches=1, remote_takes=100]
      CoalesceBatchesExec: target_batch_size=1024, metrics=[output_rows=100, elapsed_compute=2.748µs]
        GlobalLimitExec: skip=0, fetch=100, metrics=[output_rows=100, elapsed_compute=1.819µs]
          FilterExec: _distance@3 IS NOT NULL, metrics=[output_rows=100, elapsed_compute=10.275µs]
            SortExec: TopK(fetch=100), expr=[_distance@3 ASC NULLS LAST], preserve_partitioning=[false], metrics=[output_rows=100, elapsed_compute=39.259451ms, row_replacements=546]
              KNNVectorDistance: metric=l2, metrics=[output_rows=1099508, elapsed_compute=56.783526ms, output_batches=1076]
                FilterExec: identifier@1 > 0 AND identifier@1 < 1000000, metrics=[output_rows=1099508, elapsed_compute=17.136819ms]
                  LanceScan: uri=***, projection=[vector, identifier], row_id=true, row_addr=false, ordered=false, metrics=[output_rows=1200000, elapsed_compute=21.348178ms, bytes_read=1852931072, iops=78, requests=78]
```

The `analyze_plan` output reveals detailed performance metrics for each step of 
the query execution:

1. **Data Loading (LanceScan)**
   - Scanned 1,200,000 rows from the LanceDB table
   - Read 1.86GB of data in 78 I/O operations
   - Only loaded the necessary columns (`vector` and `identifier`) for the initial processing
   - The scan was unordered, allowing for parallel processing

2. **Filtering & Search**
   - Applied a prefilter condition (`identifier > 0 AND identifier < 1000000`) before vector search
   - This reduced the dataset from 1.2M to 1,099,508 rows, optimizing the subsequent KNN computation
   - The KNN search used L2 (Euclidean) distance metric
   - Vector comparisons were processed in 1076 batches for efficient memory & cache usage

3. **Results Processing**
   - The KNN results were sorted by distance (TopK with fetch=100)
   - Null distances were filtered out to ensure result quality
   - Batches were coalesced to a target size of 1024 rows for optimal memory usage
   - Additional columns (`chunk_index` and `title`) were fetched for the final results
   - The remote take operation fetched these additional columns for each of the 100 results
   - Final projection selected only the required columns for output


Key observations:
- The vector search operation is the primary performance bottleneck, requiring KNN computation across 1,099,508 vectors
- Significant I/O overhead with 1.86GB of data read through multiple I/O requests
- Query execution involves a full table scan due to lack of indices
- Substantial optimization potential through proper index implementation


Let's create the vector index on the `vector` column and the scalar index
on the `identifier` columns where the filter is applied. `explain_plan` and `analyze_plan` now show 
the following:

```python
ProjectionExec: expr=[chunk_index@3 as chunk_index, title@4 as title, identifier@2 as identifier, _distance@0 as _distance]
  RemoteTake: columns="_distance, _rowid, identifier, chunk_index, title"
    CoalesceBatchesExec: target_batch_size=1024
      GlobalLimitExec: skip=0, fetch=100
        SortExec: TopK(fetch=100), expr=[_distance@0 ASC NULLS LAST], preserve_partitioning=[false]
          ANNSubIndex: name=vector_idx, k=100, deltas=1
            ANNIvfPartition: uuid=83916fd5-fc45-4977-bad9-1f0737539bb9, nprobes=20, deltas=1
            ScalarIndexQuery: query=AND(identifier > 0,identifier < 1000000)
```
```python
ProjectionExec: expr=[chunk_index@3 as chunk_index, title@4 as title, identifier@2 as identifier, _distance@0 as _distance], metrics=[output_rows=100, elapsed_compute=1.488µs]
  RemoteTake: columns="_distance, _rowid, identifier, chunk_index, title", metrics=[output_rows=100, elapsed_compute=113.491859ms, output_batches=1, remote_takes=100]
    CoalesceBatchesExec: target_batch_size=1024, metrics=[output_rows=100, elapsed_compute=2.709µs]
      GlobalLimitExec: skip=0, fetch=100, metrics=[output_rows=100, elapsed_compute=2.5µs]
        SortExec: TopK(fetch=100), expr=[_distance@0 ASC NULLS LAST], preserve_partitioning=[false], metrics=[output_rows=100, elapsed_compute=72.322µs, row_replacements=153]
          ANNSubIndex: name=vector_idx, k=100, deltas=1, metrics=[output_rows=2000, elapsed_compute=111.849043ms, index_comparisons=25893, indices_loaded=0, output_batches=20, parts_loaded=20]
            ANNIvfPartition: uuid=83916fd5-fc45-4977-bad9-1f0737539bb9, nprobes=20, deltas=1, metrics=[]
            ScalarIndexQuery: query=AND(identifier > 0,identifier < 1000000), metrics=[output_rows=2, elapsed_compute=86.979354ms, index_comparisons=2301824, indices_loaded=2, output_batches=1, parts_loaded=562]
```

Let's break down this optimized query execution plan from bottom to top:
1. Scalar Index Query (Bottom Layer):
    - Performs range filter using scalar index, only 2 index files and 562 sclar index parts are loaded.
    - Made 2.3M index comparisons to find matches
```python
ScalarIndexQuery: query=AND(identifier > 0,identifier < 1000000)
metrics=[
    output_rows=2
    index_comparisons=2,301,824
    indices_loaded=2
    output_batches=1
    parts_loaded=562
    elapsed_compute=86.979354ms
]
```

2. Vector Search:
    - Uses IVF index with 20 probes, only needed to load 20 index parts
    - Made 25,893 vector comparisons
    - Found 2,000 matching vectors
```python
ANNSubIndex: name=vector_idx, k=100, deltas=1
metrics=[
    output_rows=2,000
    index_comparisons=25,893
    indices_loaded=0
    output_batches=20
    parts_loaded=20
    elapsed_compute=111.849043ms
]
```

3. Results Processing:
    - Sorts results by distance
    - Limits to top 100 results
    - Batches results into groups of 1024
```python
SortExec: TopK(fetch=100), expr=[_distance@0 ASC NULLS LAST], preserve_partitioning=[false], metrics=[output_rows=100, elapsed_compute=72.322µs, row_replacements=153]
GlobalLimitExec: skip=0, fetch=100, metrics=[output_rows=100, elapsed_compute=2.5µs]
CoalesceBatchesExec: target_batch_size=1024, metrics=[output_rows=100, elapsed_compute=2.709µs]
```

4. Data Fetching:
    - Consolidates into just 1 output batches, one remote take per row.
```python
RemoteTake: columns="_distance, _rowid, identifier, chunk_index, title", metrics=[output_rows=100, elapsed_compute=113.491859ms, output_batches=1, remote_takes=100]
```

5. Final Projection:
    - Returns only the specified columns: chunk_index, title, identifier, and distance

```python
ProjectionExec: expr=[chunk_index@3 as chunk_index, title@4 as title, identifier@2 as identifier, _distance@0 as _distance], metrics=[output_rows=100, elapsed_compute=1.488µs]
```
The plan demonstrates a combined scalar index filter + vector similarity search, 
optimized to first filter by `identifier` before performing the ANN search.

### Key Improvements:

1. **Initial Data Access**
```python
ScalarIndexQuery metrics:
- indices_loaded=2
- parts_loaded=562
- output_batches=1
```
   - Before optimization: Full table scan of 1.2M rows, reading 1.86GB of data
   - After optimization: With scalar index on the `identifier` column, only 2 indices (vector and scalar) and 562 scalar index parts were loaded
   - Benefit: Eliminated the need for any table scans to load the prefilter, significantly reducing I/O operations
    
2. **Vector Search Efficiency**
```python
ANNSubIndex:
- index_comparisons=25,893
- indices_loaded=0
- parts_loaded=20
- output_batches=20
```
   - Before optimization: L2 distance calculations performed on 1,099,508 vectors
   - After optimization: 
     - Reduced vector comparisons by 99.8% (from ~1.1M to 2K)
     - Decreased output batches from 1,076 to 20

3. **Data Retrieval Optimization**
```python
RemoteTake:
- remote_takes=100
- output_batches=1
```
   - RemoteTake operation remains the same.


## Performance Optimization Tips

1. **Index Implementation Guide**

When to Create Indices
- Columns used in WHERE clauses
- Vector columns for similarity searches
- Join columns used in `merge_insert`

Index Type Selection

| Data Type | Recommended Index | Use Case |
|-----------|------------------|----------|
| Vector | IVF_PQ/IVF_HNSW_SQ | Approximate nearest neighbor search |
| Scalar | B-Tree | Range queries and sorting |
| Categorical | Bitmap | Multi-value filters and set operations |
| `List` | Label_list | Multi-label classification and filtering |

More details on indexing can be found [here](../guides/build-index.md#vector-index). 
<Info>
    Use `table.index_stats()` to monitor index coverage. 
    A well-optimized table should have `num_unindexed_rows ~ 0`.
</Info>

2. **Query Plan Optimization**

Common Patterns and Fixes:

| Plan Pattern | Optimization |
|--------------|--------------|
| LanceScan with high _bytes_read_ or _iops_ | Add missing index |
|  | Use `select()` to limit returned columns |
|  | Check whether the dataset has been compacted |
| Multiple sequential filters | Reorder filter conditions |


<Note>
Regularly analyze your query plans to identify and address performance bottlenecks. 
The `analyze_plan` output provides detailed metrics to guide optimization efforts.
</Note>

## First Step: Use the Right Indices

For vector search performance:
- Create ANN index on your vector column(s) as described in the [index guide](../guides/build-index.md#vector-index)
- If you often filter by metadata, create [scalar indices](../guides/build-index.md#scalar-index) on those columns



