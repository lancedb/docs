---
title: "Quickstart Guide"
description: "Effortlessly transform Open Source experience to Cloud/Enterprise"
---

# Quickstart Guide with LanceDB Cloud and Enterprise

Explore the full implementation in this Quickstart guide:

→ [Python Notebook](https://colab.research.google.com/github/lancedb/vectordb-recipes/blob/main/examples/saas_examples/python_notebook/LanceDB_Cloud_quickstart.ipynb)  | [TypeScript Example](https://github.com/lancedb/vectordb-recipes/tree/main/examples/saas_examples/ts_example/quickstart)

## Prerequisite:
Install the LanceDB SDK with your preferred language.

<CodeGroup>

```Python Python
# Install lancedb and the optional datasets package used in the quickstart example
pip install lancedb datasets
```

```typescript Typescript
npm install @lancedb/lancedb
```

</CodeGroup>
## 1. Connect to LanceDB Cloud/Enterprise
<Note>
  - For LanceDB Cloud users, the database URI (which starts with `db://`) and API key can both be 
  retrieved from the LanceDB Cloud UI. For step-by-step instructions, 
  refer to our [onboarding tutorial](https://app.storylane.io/share/pudefwx54tun).
  - For LanceDB Enterprise user, please contact [our team](mailto:contact@lancedb.com) 
  to obtain your database URI, API key and host_override URL.
</Note>

<CodeGroup>
```Python Python
import lancedb
import numpy as np
import pyarrow as pa
import os

# Connect to LanceDB Cloud/Enterprise
uri = "db://your-database-uri"
api_key = "your-api-key"
region = "us-east-1"

# (Optional) For LanceDB Enterprise, set the host override to your enterprise endpoint
host_override = os.environ.get("LANCEDB_HOST_OVERRIDE")

db = lancedb.connect(
  uri=uri,
  api_key=api_key,
  region=region,
  host_override=host_override
)
```

```typescript Typescript
import { connect, Index, Table } from '@lancedb/lancedb';
import { FixedSizeList, Field, Float32, Schema, Utf8 } from 'apache-arrow';

// Connect to LanceDB Cloud/Enterprise
const dbUri = process.env.LANCEDB_URI || 'db://your-database-uri';
const apiKey = process.env.LANCEDB_API_KEY;
const region = process.env.LANCEDB_REGION;

// (Optional) For LanceDB Enterprise, set the host override to your enterprise endpoint
const hostOverride = process.env.LANCEDB_HOST_OVERRIDE;

const db = await connect(dbUri, { 
  apiKey,
  region,
  hostOverride
});
```
</CodeGroup>

## 2. Load Dataset
<CodeGroup>

```Python Python
from datasets import load_dataset

# Load a sample dataset from HuggingFace with pre-computed embeddings
sample_dataset = load_dataset("sunhaozhepy/ag_news_sbert_keywords_embeddings", split="test[:1000]")
print(f"Loaded {len(sample_dataset)} samples")
print(f"Sample features: {sample_dataset.features}")
print(f"Column names: {sample_dataset.column_names}")

# Preview the first sample
print(sample_dataset[0])

# Get embedding dimension
vector_dim = len(sample_dataset[0]["keywords_embeddings"])
print(f"Embedding dimension: {vector_dim}")
```

```typescript Typescript [expandable]
const BATCH_SIZE = 100; // HF API default limit
const POLL_INTERVAL = 10000; // 10 seconds
const MAX_RETRIES = 5;
const INITIAL_RETRY_DELAY = 1000; // 1 second

interface Document {
    text: string;
    label: number;
    keywords: string[];
    embeddings?: number[];
    [key: string]: unknown;
}

interface HfDatasetResponse {
    rows: {
        row: {
            text: string;
            label: number;
            keywords: string[];
            keywords_embeddings?: number[];
        };
    }[];
}

/**
 * Loads documents from the Hugging Face dataset API in batches
 */
async function loadDataset(datasetName: string, split: string = 'train', targetSize: number = 1000, offset: number = 0): Promise<Document[]> {    
    try {
        console.log('Fetching dataset...');
        const batches = Math.ceil(targetSize / BATCH_SIZE);
        let allDocuments: Document[] = [];
        const hfToken = process.env.HF_TOKEN; // Optional Hugging Face token

        for (let i = 0; i < batches; i++) {
            const offset = i * BATCH_SIZE;
            const url = `https://datasets-server.huggingface.co/rows?dataset=${datasetName}&config=default&split=${split}&offset=${offset}&limit=${BATCH_SIZE}`;
            console.log(`Fetching batch ${i + 1}/${batches} from offset ${offset}...`);
            
            // Add retry logic with exponential backoff
            let retries = 0;
            let success = false;
            let data: HfDatasetResponse | null = null;

            while (!success && retries < MAX_RETRIES) {
                try {
                    const headers: HeadersInit = {
                        'Content-Type': 'application/json',
                    };
                    
                    // Add authorization header if token is available
                    if (hfToken) {
                        headers['Authorization'] = `Bearer ${hfToken}`;
                    }
                    
                    const fetchOptions = {
                        method: 'GET',
                        headers,
                        timeout: 30000, // 30 second timeout
                    };
                    
                    const response = await fetch(url, fetchOptions);
                    if (!response.ok) {
                        const errorText = await response.text();
                        console.error(`Error response (attempt ${retries + 1}):`, errorText);
                        throw new Error(`HTTP error! status: ${response.status}, body: ${errorText}`);
                    }
                    
                    data = JSON.parse(await response.text()) as HfDatasetResponse;
                    if (!data.rows) {
                        throw new Error('No rows found in response');
                    }
                    
                    success = true;
                } catch (error) {
                    retries++;
                    if (retries >= MAX_RETRIES) {
                        console.error(`Failed after ${MAX_RETRIES} retries:`, error);
                        throw error;
                    }
                    
                    const delay = INITIAL_RETRY_DELAY * Math.pow(2, retries - 1);
                    console.log(`Retry ${retries}/${MAX_RETRIES} after ${delay}ms...`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                }
            }
            
            // Ensure data is defined before using it
            if (!data || !data.rows) {
                throw new Error('No data received after retries');
            }
            
            console.log(`Received ${data.rows.length} rows in batch ${i + 1}`);
            const documents = data.rows.map(({ row }) => ({
                text: row.text,
                label: row.label,
                keywords: row.keywords,
                embeddings: row.keywords_embeddings
            }));
            allDocuments = allDocuments.concat(documents);
            
            if (data.rows.length < BATCH_SIZE) {
                console.log('Reached end of dataset');
                break;
            }
        }

        console.log(`Total documents loaded: ${allDocuments.length}`);
        return allDocuments;
    } catch (error) {
        console.error("Failed to load dataset:", error);
        throw error;
    }
}

// Load dataset
console.log('Loading AG News dataset...');
const datasetName = "sunhaozhepy/ag_news_sbert_keywords_embeddings";
const split = "test";
const targetSize = 1000;
const sampleData = await loadDataset(datasetName, split, targetSize);
console.log(`Loaded ${sampleData.length} examples from AG News dataset`);
```

</CodeGroup>

## 3. Create a table and ingest data

<CodeGroup>
```Python Python
import pyarrow as pa

# Create a table with the dataset
table_name = "lancedb-cloud-quickstart"
table = db.create_table(table_name, data=sample_dataset, mode="overwrite")

# Convert list to fixedsizelist on the vector column
table.alter_columns(dict(path="keywords_embeddings", data_type=pa.list_(pa.float32(), vector_dim)))
print(f"Table '{table_name}' created successfully")
```

```typescript Typescript
const tableName = "lancedb-cloud-quickstart";

const dataWithEmbeddings: Document[] = sampleData;
const firstDocWithEmbedding = dataWithEmbeddings.find((doc: Document) => 
    (doc.embeddings && Array.isArray(doc.embeddings) && doc.embeddings.length > 0));
    
if (!firstDocWithEmbedding || !firstDocWithEmbedding.embeddings || !Array.isArray(firstDocWithEmbedding.embeddings)) {
    throw new Error('No document with valid embeddings found in the dataset. Please check if keywords_embeddings field exists.');
}
const embeddingDimension = firstDocWithEmbedding.embeddings.length;

// Create schema
const schema = new Schema([
    new Field('text', new Utf8(), true),
    new Field('label', new Float32(), true),
    new Field('keywords', new Utf8(), true),
    new Field('embeddings', new FixedSizeList(embeddingDimension, new Field('item', new Float32(), true)), true)
]);

// Create table with data
const table = await db.createTable(tableName, dataWithEmbeddings, { 
    schema,
    mode: "overwrite" 
});
console.log('Successfully created table');
```
</CodeGroup>

## 4. Create a vector index
<CodeGroup>
```Python Python
from datetime import timedelta

# Create a vector index and wait for it to complete
table.create_index("cosine", vector_column_name="keywords_embeddings", wait_timeout=timedelta(seconds=120))
print(table.index_stats("keywords_embeddings_idx"))
```
```typescript Typescript
// Create a vector index
await table.createIndex("embeddings", {
  config: Index.ivfPq({
    distanceType: "cosine",
  }),
});

// Wait for the index to be ready
const indexName = "embeddings_idx";
await table.waitForIndex([indexName], 120);
console.log(await table.indexStats(indexName));
```
</CodeGroup>
<Tip>
  The `create_index`/`createIndex` operation executes asynchronously in LanceDB Cloud/Enterprise. To ensure the index is fully built,
  you can use the `wait_timeout` parameter or call `wait_for_index` on the table.
</Tip>

# 5. Perform a vector search
<CodeGroup>
``` python Python
query_dataset = load_dataset("sunhaozhepy/ag_news_sbert_keywords_embeddings", split="test[5000:5001]")
print(f"Query keywords: {query_dataset[0]['keywords']}")
query_embed = query_dataset["keywords_embeddings"][0]

# A vector search
result = (
    table.search(query_embed)
    .select(["text", "keywords", "label"])
    .limit(5)
    .to_pandas()
)
print("Search results:")
print(result)

# A vector search with a filter
filtered_result = (
    table.search(query_embed)
    .where("label > 2")
    .select(["text", "keywords", "label"])
    .limit(5)
    .to_pandas()
)
print("Filtered search results (label > 2):")
print(filtered_result)
```
```typescript Typescript
// Perform semantic search with a new query
const queryDocs = await loadDataset(datasetName, split, 1, targetSize);
if (queryDocs.length === 0) {
    throw new Error("Failed to load a query document");
}
const queryDoc = queryDocs[0];
if (!queryDoc.embeddings || !Array.isArray(queryDoc.embeddings)) {
    throw new Error("Query document doesn't have a valid embedding after processing");
}
const results = await table.search(queryDoc.embeddings)
    .limit(5)
    .select(['text','keywords','label'])
    .toArray();

console.log('Search Results:');
console.log(results);

// perform semantic search with a filter applied
const filteredResultsesults = await table.search(queryDoc.embeddings)
    .where("label > 2")
    .limit(5)
    .select(['text', 'keywords','label'])
    .toArray();

console.log('Search Results with filter:');
console.log(filteredResultsesults);
```
</CodeGroup>
## 6. Drop the table
<CodeGroup>
```Python Python
db.drop_table(table_name)
```

```typescript Typescript
await db.dropTable(tableName);
```
</CodeGroup>

## Next Steps

* Dive into the `Work with Data` section to unlock LanceDB's full potential.
* Explore our [python notebooks](https://github.com/lancedb/vectordb-recipes/tree/main/examples/saas_examples/python_notebook) and [typescript examples](https://github.com/lancedb/vectordb-recipes/tree/main/examples/saas_examples/ts_example) for real-world use cases.
