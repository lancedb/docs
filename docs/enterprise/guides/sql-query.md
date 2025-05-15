---
title: Query Data with SQL
---
# SQL Queries with LanceDB Enterprise

> Note: This is a preview feature that is only available in LanceDB Enterprise.

Our solution includes an SQL endpoint that can be used for analytical queries and data exploration.
The SQL endpoint is designed to be compatible with the
[Arrow FlightSQL protocol](https://arrow.apache.org/docs/format/FlightSql.html), which allows you
to use any Arrow FlightSQL-compatible client to query your data.

### Installing a client

There are Flight SQL clients available for most languages and tools.  If you find that your
preferred language or tool is not listed here, please [reach out](mailto:contact@lancedb.com) to us and we can help you
find a solution.  The following examples demonstrate how to install the Python and TypeScript
clients.

<CodeGroup>
```bash Python
# The `flightsql-dbapi` package provides a Python DB API 2 interface to the
# LanceDB SQL endpoint. You can use it to connect to the SQL endpoint and
# execute queries directly and get back results in pyarrow format.

pip install flightsql-dbapi
```

```bash Typescript
# LanceDB maintains a TypeScript client for the Arrow FlightSQL protocol.
# You can use it to connect to the SQL endpoint and execute queries directly.
# Results are returned in Arrow format or as plain JS/TS objects.

npm install --save @lancedb/flightsql-client
```
</CodeGroup>

### Usage

LanceDB uses the powerful DataFusion query engine to execute SQL queries.  This means that
you can use a wide variety of SQL syntax and functions to query your data.  For more detailed
information on the SQL syntax and functions supported by DataFusion, please refer to the
[DataFusion documentation](https://datafusion.apache.org/user-guide/sql/index.html).

<CodeGroup>
```python Python
from flightsql import FlightSQLClient

client = FlightSQLClient(
    host="your-enterprise-endpoint",
    port=10025,
    insecure=True,
    token="DATABASE_TOKEN",
    metadata={"database": "your-project-slug"},
    features={"metadata-reflection": "true"},
)

def run_query(query: str):
    """Simple method to fully materialize query results"""
    info = client.execute(query)
    if len(info.endpoints) != 1:
        raise Error("Expected exactly one endpoint")
    ticket = info.endpoints[0].ticket
    reader = client.do_get(ticket)
    return reader.read_all()

print(run_query("SELECT * FROM flights WHERE origin = 'SFO'"))
```

```typescript Typescript
import { Client } from "@lancedb/flightsql-client";

const client = await Client.connect({
  host: "your-enterprise-endpoint:10025",
  username: "lancedb",
  password: "password",
});

const result = await client.query("SELECT * FROM flights WHERE origin = 'SFO'");

// Results are returned as plain JS/TS objects and we create an interface
// here for our expected structure so we can have strong typing.  This is
// optional but recommended.
interface FlightRecord {
    origin: string;
    destination: string;
}

const flights = (await result.collectToObjects()) as FlightRecord[];
console.log(flights);
```
</CodeGroup>

