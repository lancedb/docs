# Hybrid Search

LanceDB supports both semantic and keyword-based search (also termed full-text search, or FTS). In real world applications, it is often useful to combine these two approaches to get the best best results. For example, you may want to search for a document that is semantically similar to a query document, but also contains a specific keyword. This is an example of *hybrid search*, a search algorithm that combines multiple search techniques.

## Hybrid search in LanceDB
You can perform hybrid search in LanceDB by combining the results of semantic and full-text search via a reranking algorithm of your choice. LanceDB provides multiple rerankers out of the box. However, you can always write a custom reranker if your use case need more sophisticated logic .

=== "Sync API"

    ```python
    --8<-- "python/python/tests/docs/test_search.py:import-os"
    --8<-- "python/python/tests/docs/test_search.py:import-openai"
    --8<-- "python/python/tests/docs/test_search.py:import-lancedb"
    --8<-- "python/python/tests/docs/test_search.py:import-embeddings"
    --8<-- "python/python/tests/docs/test_search.py:import-pydantic"
    --8<-- "python/python/tests/docs/test_search.py:import-lancedb-fts"
    --8<-- "python/python/tests/docs/test_search.py:import-openai-embeddings"
    --8<-- "python/python/tests/docs/test_search.py:class-Documents"
    --8<-- "python/python/tests/docs/test_search.py:basic_hybrid_search"
    ```
=== "Async API"

    ```python
    --8<-- "python/python/tests/docs/test_search.py:import-os"
    --8<-- "python/python/tests/docs/test_search.py:import-openai"
    --8<-- "python/python/tests/docs/test_search.py:import-lancedb"
    --8<-- "python/python/tests/docs/test_search.py:import-embeddings"
    --8<-- "python/python/tests/docs/test_search.py:import-pydantic"
    --8<-- "python/python/tests/docs/test_search.py:import-lancedb-fts"
    --8<-- "python/python/tests/docs/test_search.py:import-openai-embeddings"
    --8<-- "python/python/tests/docs/test_search.py:class-Documents"
    --8<-- "python/python/tests/docs/test_search.py:basic_hybrid_search_async"
    ```

!!! Note
    You can also pass the vector and text query manually. This is useful if you're not using the embedding API or if you're using a separate embedder service.
### Explicitly passing the vector and text query
=== "Sync API"

    ```python
    --8<-- "python/python/tests/docs/test_search.py:hybrid_search_pass_vector_text"
    ```
=== "Async API"

    ```python
    --8<-- "python/python/tests/docs/test_search.py:hybrid_search_pass_vector_text_async"
    ```

By default, LanceDB uses `RRFReranker()`, which uses reciprocal rank fusion score, to combine and rerank the results of semantic and full-text search. You can customize the hyperparameters as needed or write your own custom reranker. Here's how you can use any of the available rerankers:


### `rerank()` arguments
* `normalize`: `str`, default `"score"`:
    The method to normalize the scores. Can be "rank" or "score". If "rank", the scores are converted to ranks and then normalized. If "score", the scores are normalized directly.
* `reranker`: `Reranker`, default `RRF()`.
    The reranker to use. If not specified, the default reranker is used.


## Available Rerankers
LanceDB provides a number of rerankers out of the box. You can use any of these rerankers by passing them to the `rerank()` method. 
Go to [Rerankers]LINK(../guides/reranking/index.md) to learn more about using the available rerankers and implementing custom rerankers.

# Hybrid Search

Hybrid Search is a broad (often misused) term. It can mean anything from combining multiple methods for searching, to applying ranking methods to better sort the results. In this blog, we use the definition of "hybrid search" to mean using a combination of keyword-based and vector search.

## The challenge of (re)ranking search results
Once you have a group of the most relevant search results from multiple search sources, you'd likely standardize the score and rank them accordingly. This process can also be seen as another independent step: reranking.
There are two approaches for reranking search results from multiple sources.

* <b>Score-based</b>: Calculate final relevance scores based on a weighted linear combination of individual search algorithm scores. Example: Weighted linear combination of semantic search & keyword-based search results.

* <b>Relevance-based</b>: Discards the existing scores and calculates the relevance of each search result-query pair. Example: Cross Encoder models

Even though there are many strategies for reranking search results, none works for all cases. Moreover, evaluating them itself is a challenge. Also, reranking can be dataset or application specific so it's hard to generalize.

### Example evaluation of hybrid search with Reranking

Here's some evaluation numbers from an experiment comparing these rerankers on about 800 queries. It is modified version of an evaluation script from [llama-index](https://github.com/run-llama/finetune-embedding/blob/main/evaluate.ipynb) that measures hit-rate at top-k.

<b> With OpenAI ada2 embedding </b>

Vector Search baseline: `0.64`

| Reranker | Top-3 | Top-5 | Top-10 |
| --- | --- | --- | --- |
| Linear Combination | `0.73` | `0.74` | `0.85` |
| Cross Encoder | `0.71` | `0.70` | `0.77` |
| Cohere | `0.81` | `0.81` | `0.85` |
| ColBERT | `0.68` | `0.68` | `0.73` |

<p>
<img src="https://github.com/AyushExel/assets/assets/15766192/d57b1780-ef27-414c-a5c3-73bee7808a45">
</p>

<b> With OpenAI embedding-v3-small </b>

Vector Search baseline: `0.59`

| Reranker | Top-3 | Top-5 | Top-10 |
| --- | --- | --- | --- |
| Linear Combination | `0.68` | `0.70` | `0.84` |
| Cross Encoder | `0.72` | `0.72` | `0.79` |
| Cohere | `0.79` | `0.79` | `0.84` |
| ColBERT | `0.70` | `0.70` | `0.76` |

<p>
<img src="https://github.com/AyushExel/assets/assets/15766192/259adfd2-6ec6-4df6-a77d-1456598970dd">
</p>

### Conclusion

The results show that the reranking methods are able to improve the search results. However, the improvement is not consistent across all rerankers. The choice of reranker depends on the dataset and the application. It is also important to note that the reranking methods are not a replacement for the search methods. They are complementary and should be used together to get the best results. The speed to recall tradeoff is also an important factor to consider when choosing the reranker.
