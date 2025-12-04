---
title: "Polars"
sidebar_title: "Polars & Arrow"
weight: 2
---

LanceDB supports [Polars](https://github.com/pola-rs/polars), a blazingly fast DataFrame library for Python written in Rust. Just like in Pandas, the Polars integration is enabled by PyArrow under the hood. A deeper integration between Lance Tables and Polars DataFrames is in progress, but at the moment, you can read a Polars DataFrame into LanceDB and output the search results from a query to a Polars DataFrame.


## Create & Query LanceDB Table

### From Polars DataFrame

First, we connect to a LanceDB database.

{{< code language="python" source="examples/py/test_python.py" id="import-lancedb" />}}

Let's also import Polars:

{{< code language="python" source="examples/py/test_python.py" id="import-polars" />}}

#### Sync API

{{< code language="python" source="examples/py/test_python.py" id="connect_to_lancedb" />}}

We can then load a Polars `DataFrame` to LanceDB directly.

{{< code language="python" source="examples/py/test_python.py" id="create_table_polars" />}}

You can now perform similarity search via the LanceDB Python API.

{{< code language="python" source="examples/py/test_python.py" id="vector_search_polars" />}}

In addition to the selected columns, LanceDB also returns a vector
and also the `_distance` column which is the distance between the query
vector and the returned vector.

```
shape: (1, 4)
┌───────────────┬──────┬───────┬───────────┐
│ vector        ┆ item ┆ price ┆ _distance │
│ ---           ┆ ---  ┆ ---   ┆ ---       │
│ array[f32, 2] ┆ str  ┆ f64   ┆ f32       │
╞═══════════════╪══════╪═══════╪═══════════╡
│ [3.1, 4.1]    ┆ foo  ┆ 10.0  ┆ 0.0       │
└───────────────┴──────┴───────┴───────────┘
<class 'polars.dataframe.frame.DataFrame'>
```

Note that the type of the result from a table search is a Polars DataFrame.

#### Async API

Let's look at the same workflow, this time, using LanceDB's async Python API.

{{< code language="python" source="examples/py/test_python.py" id="connect_to_lancedb_async" />}}

We can then load a Polars `DataFrame` to LanceDB directly.

{{< code language="python" source="examples/py/test_python.py" id="create_table_polars_async" />}}

You can now perform similarity search via the LanceDB Python API.

{{< code language="python" source="examples/py/test_python.py" id="vector_search_polars_async" />}}

In addition to the selected columns, LanceDB also returns a vector
and also the `_distance` column which is the distance between the query
vector and the returned vector.

```
shape: (1, 4)
┌───────────────┬──────┬───────┬───────────┐
│ vector        ┆ item ┆ price ┆ _distance │
│ ---           ┆ ---  ┆ ---   ┆ ---       │
│ array[f32, 2] ┆ str  ┆ f64   ┆ f32       │
╞═══════════════╪══════╪═══════╪═══════════╡
│ [3.1, 4.1]    ┆ foo  ┆ 10.0  ┆ 0.0       │
└───────────────┴──────┴───────┴───────────┘
<class 'polars.dataframe.frame.DataFrame'>
```

Note that the type of the result from a table search is a Polars DataFrame.

### From Pydantic Models

Alternately, we can create an empty LanceDB Table using a Pydantic schema and populate it with a Polars DataFrame.

Let's first import Polars:

{{< code language="python" source="examples/py/test_python.py" id="import-polars" />}}

And then the necessary models from Pydantic:

{{< code language="python" source="examples/py/test_python.py" id="import-lancedb-pydantic" />}}

First, let's define a Pydantic model:

{{< code language="python" source="examples/py/test_python.py" id="class_Item" />}}

We can then create the table from the Pydantic model and add the Polars DataFrame to the Lance table
as follows:

{{< code language="python" source="examples/py/test_python.py" id="create_table_pydantic" />}}

The table can now be queried as usual.

{{< code language="python" source="examples/py/test_python.py" id="vector_search_polars" />}}

```
shape: (1, 4)
┌───────────────┬──────┬───────┬───────────┐
│ vector        ┆ item ┆ price ┆ _distance │
│ ---           ┆ ---  ┆ ---   ┆ ---       │
│ array[f32, 2] ┆ str  ┆ f64   ┆ f32       │
╞═══════════════╪══════╪═══════╪═══════════╡
│ [3.1, 4.1]    ┆ foo  ┆ 10.0  ┆ 0.02      │
└───────────────┴──────┴───────┴───────────┘
<class 'polars.dataframe.frame.DataFrame'>
```

This result is the same as the previous one, with a DataFrame returned.

## Dump Table to LazyFrame

As you iterate on your application, you'll likely need to work with the whole table's data pretty frequently, for which Polars provides a lazily-evaluated alternative to a DataFrame.
LanceDB tables can also be converted directly into a Polars `LazyFrame` for further processing.

{{< code language="python" source="examples/py/test_python.py" id="dump_table_lazyform" />}}

Unlike the search result from a query, we can see that the type of the result is a LazyFrame.

```
<class 'polars.lazyframe.frame.LazyFrame'>
```

We can now work with the LazyFrame as we would in Polars, and collect the first result.

{{< code language="python" source="examples/py/test_python.py" id="print_table_lazyform" />}}

```
shape: (1, 3)
┌───────────────┬──────┬───────┐
│ vector        ┆ item ┆ price │
│ ---           ┆ ---  ┆ ---   │
│ array[f32, 2] ┆ str  ┆ f64   │
╞═══════════════╪══════╪═══════╡
│ [3.1, 4.1]    ┆ foo  ┆ 10.0  │
└───────────────┴──────┴───────┘
```

The reason it's beneficial to not convert the LanceDB Table
to a regular Polars `DataFrame` is that the table can potentially be _way_ larger
than memory. Using a Polars `LazyFrame` allows us to work with such
larger-than-memory datasets by not loading it into memory all at once.
