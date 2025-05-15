---
title: Benchmarking
---

# Benchmarking LanceDB Enterprise

LanceDB designed our architecture in a way to deliver **25ms** vector search latency.
Even with metadata filtering, our query latency remains as low as **50ms**.
It is important to note that we can support thousands of QPS with such query performance.

| Percentile | vector search | vector search w. filtering | full-text search |
|:----------|:--------------:|:--------------:|:--------------:|
| P50 | 25ms         |  30ms         | 26ms         |
| P90 | 26ms         |  39ms         | 37ms         |
| P99 | 35ms         |  50ms         | 42ms         |


## Dataset

We used two datasets for this benchmark test. The [dbpedia-entities-openai-1M](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M) 
for vector search, and a synthetic dataset for vector search with metadata filtering. 

| Name | # Vector | Vector Dimension |
|:-------------|:--------------:|:--------------:|
| dbpedia-entities-openai-1M         | 1,000,000         | 1536 |
| synthetic dataset         | 15,000,000 | 256 |

## Vector search

We ran vector queries with dbpedia-entities-openai-1M with warmed up cache. 
The query latency is as follows:

| Percentile | Latency|
|:-------------|:--------------:|
| P50         | 25ms         | 
| P90         | 26ms         |
| P99         | 35ms         |
| Max         | 49ms         |

## Full-text search

With the same dataset, and warmed-up cache, the full-text search performance is as follows:

| Percentile | Latency|
|:-------------|:--------------:|
| P50         | 26ms         | 
| P90         | 37ms         |
| P99         | 42ms         |
| Max         | 98ms         |

## Vector search with metadata filtering
We created a 15M-vector dataset with sufficient complexity to thoroughly test our complex metadata filtering capabilities. 
Such filtering can span a wide range on the scalar columns, _e.g. find Sci-fi movies since 1900_. 

With a warmed-up cache, the query performance using slightly more selective filters, 
**e.g. find Sci-fi movies between the year of 2000 and 2012**, is as follows: 

| Percentile | Latency|
|:-------------|:--------------:|
| P50         | 30ms         | 
| P90         | 39ms         |
| P99         | 50ms         |

The query performance using complex filters, **e.g. find Sci-fi movies since 1900**, is as follows:

| Percentile | Latency|
|:-------------|:--------------:|
| P50         | 65ms         | 
| P90         | 76ms         |
| P99         | 100ms        |

Note: Our benchmarking tests provide consistent, up-to-date performance evaluations of LanceDB.  
We regularly update and re-run these benchmarks to ensure the data remains accurate and relevant. 
