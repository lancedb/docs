---
title: "Pandas and PyArrow"
sidebar_title: "Pandas & PyArrow"
weight: 1
---

Because Lance is built on top of [Apache Arrow](https://arrow.apache.org/),
LanceDB is tightly integrated with the Python data ecosystem, including [Pandas](https://pandas.pydata.org/)
and [pyarrow](https://arrow.apache.org/docs/python/index.html). The sequence of steps in a typical workflow with Pandas is shown below.

## Create dataset

Let's first import LanceDB:

{{< code language="python" source="examples/py/test_python.py" id="import-lancedb" />}}

Next, we'll import pandas

{{< code language="python" source="examples/py/test_python.py" id="import-pandas" />}}

#### Sync API

We'll first connect to LanceDB.

{{< code language="python" source="examples/py/test_python.py" id="connect_to_lancedb" />}}

We can create a LanceDB table directly from a Pandas DataFrame by passing it as the `data` parameter:

{{< code language="python" source="examples/py/test_python.py" id="create_table_pandas" />}}

Similar to the [pyarrow.write_dataset()](https://arrow.apache.org/docs/python/generated/pyarrow.dataset.write_dataset.html) method, LanceDB's
[`db.create_table()`](/docs/reference/python/#lancedb.db.DBConnection.create_table) accepts data in a variety of forms, including pyarrow datasets.

#### Async API

Connect to LanceDB:

{{< code language="python" source="examples/py/test_python.py" id="connect_to_lancedb_async" />}}

We can create a LanceDB table directly from a Pandas DataFrame by passing it as the `data` parameter:

{{< code language="python" source="examples/py/test_python.py" id="create_table_pandas_async" />}}

### Larger-than-memory data

If you have a dataset that is larger than memory, you can create a table with `Iterator[pyarrow.RecordBatch]` to lazily load the data:

{{< code language="python" source="examples/py/test_python.py" id="import-iterable" />}}
\
{{< code language="python" source="examples/py/test_python.py" id="import-pyarrow" />}}
\
{{< code language="python" source="examples/py/test_python.py" id="make_batches" />}}

You can then pass the `make_batches()` function to the `data` parameter, while specifying the pyarrow schema in the `create_table()` function.

#### Sync API

{{< code language="python" source="examples/py/test_python.py" id="create_table_iterable" />}}

#### Async API

{{< code language="python" source="examples/py/test_python.py" id="create_table_iterable_async" />}}

You will find detailed instructions of creating a LanceDB dataset in
[Getting Started](/docs/quickstart/basic-usage/#quick-start) and [API](/docs/reference)
sections.

## Vector search

We can now perform similarity search via the LanceDB Python API.

#### Sync API

{{< code language="python" source="examples/py/test_python.py" id="vector_search" />}}

#### Async API

{{< code language="python" source="examples/py/test_python.py" id="vector_search_async" />}}

This returns a Pandas DataFrame as follows:

```
    vector     item  price    _distance
0  [5.9, 26.5]  bar   20.0  14257.05957
```

If you have a simple filter, it's faster to provide a `where` clause to LanceDB's `search` method.
For more complex filters or aggregations, you can always resort to using the underlying `DataFrame` methods after performing a search.

#### Sync API

{{< code language="python" source="examples/py/test_python.py" id="vector_search_with_filter" />}}

#### Async API

{{< code language="python" source="examples/py/test_python.py" id="vector_search_with_filter_async" />}}
