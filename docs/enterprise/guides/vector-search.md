---
title: Vector Search
---

# Vector Search with LanceDB Enterprise

We support lightning fast vector search on massive scale data. Following performance 
data shows search latency from a 1M dataset with warmed up cache. 
| Percentile | Latency|
|:-------------|:--------------:|
| P50         | 25ms         | 
| P90         | 26ms         |
| P99         | 35ms         |
| Max         | 49ms         |

Other than latency, users can also tune the following parameters for better search quality.
 - [nprobes](https://lancedb.github.io/lancedb/ann_indexes/#querying-an-ann-index): 
 the number of partitions to search (probe)
 - [refine factor](https://lancedb.github.io/lancedb/ann_indexes/#querying-an-ann-index): 
 a multiplier to control how many additional rows are taken during the refine step
 - [distance range](https://lancedb.github.io/lancedb/search/#search-with-distance-range): search for vectors within the distance range

LanceDB delivers exceptional vector search performance with 
metadata filtering.
Benchmark results demonstrate **65ms** query latency at scale, tested on a 
15-million vector dataset.This combination of fast vector 
search and precise metadata filtering enables efficient, 
accurate querying of large-scale datasets.

### Vector search with metadata _prefiltering_

<CodeGroup>
```Python Python
import lancedb
from datasets import load_dataset

# Connect to LanceDB
db = lancedb.connect(
  uri="db://your-project-slug",
  api_key="your-api-key",
  region="us-east-1"
)

# Load query vector from dataset
query_dataset = load_dataset("sunhaozhepy/ag_news_sbert_keywords_embeddings", split="test[5000:5001]")
print(f"Query keywords: {query_dataset[0]['keywords']}")
query_embed = query_dataset["keywords_embeddings"][0]

# Open table and perform search
table_name = "lancedb-cloud-quickstart"
table = db.open_table(table_name)

# Vector search with filters (pre-filtering is the default)
search_results = (
    table.search(query_embed)
    .where("label > 2")
    .select(["text", "keywords", "label"])
    .limit(5)
    .to_pandas()
)

print("Search results (with pre-filtering):")
print(search_results)
```
```typescript TypeScript
import * as lancedb from "@lancedb/lancedb";

// Connect to LanceDB
const db = await lancedb.connect({
  uri: "db://your-project-slug",
  apiKey: "your-api-key",
  region: "us-east-1"
});

// Generate a sample 768-dimension embedding vector (typical for BERT-based models)
// In real applications, you would get this from an embedding model
const dimensions = 768;
const queryEmbed = Array.from({ length: dimensions }, () => Math.random() * 2 - 1);

// Open table and perform search
const tableName = "lancedb-cloud-quickstart";
const table = await db.openTable(tableName);

// Vector search with filters (pre-filtering is the default)
const vectorResults = await table.search(queryEmbed)
  .where("label > 2")
  .select(["text", "keywords", "label"])
  .limit(5)
  .toArray();

console.log("Search results (with pre-filtering):");
console.log(vectorResults);
```
</CodeGroup>

### Vector search with metadata _postfiltering_

 By default, pre-filtering is performed to filter prior to vector search. 
 This can be useful to narrow down the search space of a very large dataset to 
 reduce query latency. Post-filtering is also an option that performs the filter 
 on the results returned by the vector search. You can use post-filtering as follows:
 <CodeGroup>
 ```Python Python
results_post_filtered = (
    table.search(query_embed)
    .where("label > 1", prefilter=False)
    .select(["text", "keywords", "label"])
    .limit(5)
    .to_pandas()
)

print("Vector search results with post-filter:")
print(results_post_filtered)
```
```TypeScript TypeScript
const vectorResultsWithPostFilter = await (table.search(queryEmbed) as VectorQuery)
  .where("label > 2")
  .postfilter()
  .select(["text", "keywords", "label"])
  .limit(5)
  .toArray();

console.log("Vector search results with post-filter:");
console.log(vectorResultsWithPostFilter);
```
</CodeGroup>

### Batch query
LanceDB can process multiple similarity search requests 
simultaneously in a single operation, rather than handling 
each query individually. 
<CodeGroup>
```Python Python
# Load a batch of query embeddings
query_dataset = load_dataset(
    "sunhaozhepy/ag_news_sbert_keywords_embeddings", split="test[5000:5005]"
)
query_embeds = query_dataset["keywords_embeddings"]
batch_results = table.search(query_embeds).limit(5).to_pandas()
print(batch_results)
```
```TypeScript TypeScript
 // Batch query
console.log("Performing batch vector search...");
const batchSize = 5;
const queryVectors = Array.from(
  { length: batchSize },
  () => Array.from(
    { length: dimensions },
    () => Math.random() * 2 - 1,
  ),
);
let batchQuery = table.search(queryVectors[0]) as VectorQuery;
for (let i = 1; i < batchSize; i++) {
  batchQuery = batchQuery.addQueryVector(queryVectors[i]);
}
const batchResults = await batchQuery
  .select(["text", "keywords", "label"])
  .limit(5)
  .toArray();
console.log("Batch vector search results:");
console.log(batchResults);
```
</CodeGroup>
<Info>
  When processing batch queries, the results include a `query_index` field 
  to explicitly associate each result set with its corresponding query in 
  the input batch. 
</Info>

### Other search options
**Fast search**

While vector indexing occurs asynchronously, newly added vectors are immediately 
searchable through a fallback brute-force search mechanism. This ensures zero 
latency between data insertion and searchability, though it may temporarily 
increase query response times. To optimize for speed over completeness, 
enable the `fast_search` flag in your query to skip searching unindexed data.
<CodeGroup>
```Python Python
# sync API
table.search(embedding, fast_search=True).limit(5).to_pandas()

# async API
await table.query().nearest_to(embedding).fast_search().limit(5).to_pandas()
```
```TypeScript TypeScript
await table
  .query()
  .nearestTo(embedding)
  .fastSearch()
  .limit(5)
  .toArray();
```
</CodeGroup>

**Bypass Vector Index**

The bypass vector index feature prioritizes search accuracy over query speed by performing 
an exhaustive search across all vectors. Instead of using the approximate nearest neighbor 
(ANN) index, it compares the query vector against every vector in the table directly. 

While this approach increases query latency, especially with large datasets, it provides 
exact, ground-truth results. This is particularly useful when:
- Evaluating ANN index quality
- Calculating recall metrics to tune `nprobes` parameter
- Verifying search accuracy for critical applications
- Benchmarking approximate vs exact search results

<CodeGroup>
```Python Python
# sync API
table.search(embedding).bypass_vector_index().limit(5).to_pandas()

# async API
await table.query().nearest_to(embedding).bypass_vector_index().limit(5).to_pandas()
```
```TypeScript TypeScript
await table
  .query()
  .nearestTo(embedding)
  .bypassVectorIndex()
  .limit(5)
  .toArray();
```
</CodeGroup>

<Tip>
  For system design and performance tuning:
  - Check our [benchmark page](../benchmark.md) for performance characteristics
  - For complex queries, see the [query performance tuning guide](./understand-query-perf.md)
  - When deciding on distance metrics, consider your vector type and search objectives
</Tip>