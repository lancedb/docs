---
title: Full Text Search
---

# Full Text Search with LanceDB Enterprise

The full-text search allows you to 
incorporate keyword-based search (based on BM25) in your retrieval solutions.
LanceDB can deliver **26ms** query latency for full-text search. 
Our benchmark tests have more details.

<CodeGroup>
```Python Python
import lancedb

# connect to LanceDB
db = lancedb.connect(
  uri="db://your-project-slug",
  api_key="your-api-key",
  region="us-east-1"
)

# let's use the table created from quickstart
table_name = "lancedb-cloud-quickstart"
table = db.open_table(table_name)

table.create_fts_index("text")
# Wait for FTS index to be ready
fts_index_name = f"{text_column}_idx"
wait_for_index(table, fts_index_name)

query_text = "football"
fts_results = table.search(query_text, query_type="fts").select(["text", "keywords", "label"]).limit(5).to_pandas()
```
```typescript Typescript
import * as lancedb from "@lancedb/lancedb"

const db = await lancedb.connect({
  uri: "db://your-project-slug",
  apiKey: "your-api-key",
  region: "us-east-1"
});

// Open table and perform search
const tableName = "lancedb-cloud-quickstart";
const table = await db.openTable(tableName);
const textColumn = "text";

// Create a full-text search index
await table.createIndex(textColumn, {
  config: Index.fts()
});

// Wait for the index to be ready
const ftsIndexName = textColumn + "_idx";
await waitForIndex(table, ftsIndexName);

const queryText = "football";
const ftsResults = await table.query()
  .fullTextSearch(queryText, {
    columns: [textColumn]
  })
  .select(["text", "keywords", "label"])
  .limit(5)
  .toArray();
```
</CodeGroup>
<Info>
  Newly added or modified records become searchable immediately. 
  The full-text search (FTS) index updates automatically in the background, 
  ensuring continuous search availability without blocking queries.
</Info>

## Advanced Search Features

### Fuzzy Search
Fuzzy search allows you to find matches even when the search terms contain typos or slight variations. 
LanceDB uses the classic [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) 
to find similar terms within a specified edit distance.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| fuzziness | int | 0 | Maximum edit distance allowed for each term. If not specified, automatically set based on term length: 0 for length ≤ 2, 1 for length ≤ 5, 2 for length > 5 |
| max_expansions | int | 50 | Maximum number of terms to consider for fuzzy matching. Higher values may improve recall but increase search time |

Let's create a sample table and build full-text search indices to demonstrate 
fuzzy search capabilities and relevance boosting features.
<CodeGroup>
```Python Python [expandable]
import lancedb

# connect to LanceDB
db = lancedb.connect(
  uri="db://your-project-slug",
  api_key="your-api-key",
  region="us-east-1"
)

table_name = "fts-fuzzy-boosting-test"
vectors = [np.random.randn(128) for _ in range(100)]
text_nouns = ("puppy", "car")
text2_nouns = ("rabbit", "girl", "monkey")
verbs = ("runs", "hits", "jumps", "drives", "barfs")
adv = ("crazily.", "dutifully.", "foolishly.", "merrily.", "occasionally.")
adj = ("adorable", "clueless", "dirty", "odd", "stupid")
text = [
    " ".join(
        [
            text_nouns[random.randrange(0, len(text_nouns))],
            verbs[random.randrange(0, 5)],
            adv[random.randrange(0, 5)],
            adj[random.randrange(0, 5)],
        ]
    )
    for _ in range(100)
]
text2 = [
    " ".join(
        [
            text2_nouns[random.randrange(0, len(text2_nouns))],
            verbs[random.randrange(0, 5)],
            adv[random.randrange(0, 5)],
            adj[random.randrange(0, 5)],
        ]
    )
    for _ in range(100)
]
count = [random.randint(1, 10000) for _ in range(100)]

table = db.create_table(
    table_name,
    data=pd.DataFrame(
        {
            "vector": vectors,
            "id": [i % 2 for i in range(100)],
            "text": text,
            "text2": text2,
            "count": count,
        }
    ),
    mode="overwrite"
)

table.create_fts_index("text")
wait_for_index(table, "text_idx")
table.create_fts_index("text2")
wait_for_index(table, "text2_idx")
```
```Typescript Typescript [expandable]
import * as lancedb from "@lancedb/lancedb"

const db = await lancedb.connect({
  uri: "db://your-project-slug",
  apiKey: "your-api-key",
  region: "us-east-1"
});

const tableName = "fts-fuzzy-boosting-test-ts";
  
// Generate sample data
const n = 100;
const vectors = Array.from({ length: n }, () => 
  Array.from({ length: 128 }, () => Math.random() * 2 - 1)
);

const textNouns = ["puppy", "car"];
const text2Nouns = ["rabbit", "girl", "monkey"];
const verbs = ["runs", "hits", "jumps", "drives", "barfs"];
const adverbs = ["crazily", "dutifully", "foolishly", "merrily", "occasionally"];
const adjectives = ["adorable", "clueless", "dirty", "odd", "stupid"];

const generateText = (nouns: string[]) => {
  const noun = nouns[Math.floor(Math.random() * nouns.length)];
  const verb = verbs[Math.floor(Math.random() * verbs.length)];
  const adv = adverbs[Math.floor(Math.random() * adverbs.length)];
  const adj = adjectives[Math.floor(Math.random() * adjectives.length)];
  return `${noun} ${verb} ${adv} ${adj}`;
};

const text = Array.from({ length: n }, () => generateText(textNouns));
const text2 = Array.from({ length: n }, () => generateText(text2Nouns));
const count = Array.from({ length: n }, () => Math.floor(Math.random() * 10000) + 1);

// Create table with data
const data = makeArrowTable(
  vectors.map((vector, i) => ({
    vector,
    id: i % 2,
    text: text[i],
    text2: text2[i],
    count: count[i],
  }))
);

const table = await db.createTable(tableName, data, { mode: "overwrite" });
console.log(`Created table: ${tableName}`);

// Create FTS indices
console.log("Creating FTS indices...");
await table.createIndex("text", { config: Index.fts() });
await waitForIndex(table, "text_idx");
await table.createIndex("text2", { config: Index.fts() });
await waitForIndex(table, "text2_idx");
```
</CodeGroup>


To demonstrate fuzzy search's ability to handle typos and misspellings, let's perform 
a search with a deliberately misspelled word. The search engine will attempt to match 
similar terms within the specified edit distance.

<CodeGroup>
```Python Python
from lancedb.query import MatchQuery

print("\n=== Match Query Examples ===")
# Basic match
print("\n1. Basic Match Query for 'crazily':")
basic_match_results = (
    table.search(MatchQuery("crazily", "text"), query_type="fts")
    .select(["id", "text"])
    .limit(100)
    .to_pandas()
)

# Fuzzy match (allows typos)
print("\n2. Fuzzy Match Query for 'crazi~1' (with typo):")
fuzzy_results = (
    table.search(MatchQuery("crazi~1", "text", fuzziness=2), query_type="fts")
    .select(["id", "text"])
    .limit(100)
    .to_pandas()
)
```
```typescript Typescript
import { MatchQuery } from "@lancedb/lancedb";

// Basic match
console.log("\n1. Basic Match Query for 'crazily':");
const basicMatchResults = await table.query()
  .fullTextSearch(new MatchQuery("crazily", "text"))
  .select(["id", "text"])
  .limit(100)
  .toArray();
console.log(basicMatchResults);

// Fuzzy match
console.log("\n2. Fuzzy Match Query for 'crazi~1' (with typo):");
const fuzzyResults = await table.query()
  .fullTextSearch(new MatchQuery("crazi~1", "text", {
    fuzziness: 2,
  }))  // fuzziness of 2
  .select(["id", "text"])
  .limit(100)
  .toArray();
console.log(fuzzyResults);
```
</CodeGroup>

### Phrase Match

Phrase matching enables you to search for exact sequences of words. Unlike regular text search 
which matches individual terms independently, phrase matching requires words to appear in the 
specified order with no intervening terms. This is particularly useful when:
- Searching for specific multi-word expressions
- Matching exact titles or quotes
- Finding precise word combinations in a specific order
<CodeGroup>
```Python Python
# Exact phrase match
from lancedb.query import PhraseQuery

print("\n1. Exact phrase match for 'puppy runs':")
phrase_results = (
    table.search(PhraseQuery("puppy runs", "text"), query_type="fts")
    .select(["id", "text"])
    .limit(100)
    .to_pandas()
)
```
```typescript Typescript
import { PhraseQuery } from "@lancedb/lancedb";

// Exact phrase match
console.log("\n1. Exact phrase match for 'puppy runs':");
const phraseResults = await table.query()
  .fullTextSearch(new PhraseQuery("puppy runs", "text"))
  .select(["id", "text"])
  .limit(100)
  .toArray();
```
</CodeGroup>

### Search with Boosting

Boosting allows you to control the relative importance of different search terms or fields 
in your queries. This feature is particularly useful when you need to:
- Prioritize matches in certain columns
- Promote specific terms while demoting others
- Fine-tune relevance scoring for better search results

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| positive | Query | required | The primary query terms to match and promote in results |
| negative | Query | required | Terms to demote in the search results |
| negative_boost | float | 0.5 | Multiplier for negative matches (lower values = stronger demotion) |

<CodeGroup>
```Python Python
from lancedb.query import MatchQuery, BoostQuery, MultiMatchQuery

# Boost data with 'runs' in text more than 'puppy' in text
print("\n2. Boosting data with 'runs' in text:")
boosting_results = (
  table.search(
      BoostQuery(
          MatchQuery("runs", "text"),
          MatchQuery("puppy", "text"),
          negative_boost=0.2,
      ),
      query_type="fts",
  )
  .select(["id", "text"])
  .limit(100)
  .to_pandas()
)

"""Test searching across multiple fields."""
print("\n=== Multi Match Query Examples ===")
# Search across both text and text2
print("\n1. Searching 'crazily' in both text and text2:")
multi_match_results = (
    table.search(MultiMatchQuery("crazily", ["text", "text2"]), query_type="fts")
    .select(["id", "text", "text2"])
    .limit(100)
    .to_pandas()
)

# Search with field boosting
print("\n2. Searching with boosted text2 field:")
multi_match_boosting_results = (
    table.search(
        MultiMatchQuery("crazily", ["text", "text2"], boosts=[1.0, 2.0]),
        query_type="fts",
    )
    .select(["id", "text", "text2"])
    .limit(100)
    .to_pandas()
)
```
```typescript Typescript
import { MatchQuery, BoostQuery, MultiMatchQuery } from "@lancedb/lancedb";

// Boosting Example
console.log("\n2. Boosting data with 'runs' in text:");
const boostingResults = await table.query()
  .fullTextSearch(new BoostQuery(new MatchQuery("runs", "text"), new MatchQuery("puppy", "text"), {
    negativeBoost: 0.2,
  }))
  .select(["id", "text"])
  .limit(100)
  .toArray();

// Multi Match Query Examples
console.log("\n=== Multi Match Query Examples ===");

// Search across both text fields
console.log("\n1. Searching 'crazily' in both text and text2:");
const multiMatchResults = await table.query()
  .fullTextSearch(new MultiMatchQuery("crazily", ["text", "text2"]))
  .select(["id", "text", "text2"])
  .limit(100)
  .toArray();

// Search with field boosting
console.log("\n2. Searching with boosted text2 field:");
const multiMatchBoostingResults = await table.query()
  .fullTextSearch(new MultiMatchQuery("crazily", ["text", "text2"], {
    boosts: [1.0, 2.0],
  }))
  .select(["id", "text", "text2"])
  .limit(100)
  .toArray();
```
</CodeGroup>

<Tip>
- Use fuzzy search when handling user input that may contain typos or variations
- Apply field boosting to prioritize matches in more important columns
- Combine fuzzy search with boosting for robust and precise search results

**Recommendations for optimal FTS performance:**

- Create full-text search indices on text columns that will be frequently searched
- For hybrid search combining text and vectors, see our [hybrid search guide](./hybrid-search.md)
- For performance benchmarks, check our [benchmark results](../benchmark.md)
- For complex queries, use SQL to combine FTS with other filter conditions
</Tip>

