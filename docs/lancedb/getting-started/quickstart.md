# Quick Start Guide

Welcome to LanceDB! This guide will help you get started with the basics of using LanceDB.

## Installation

Install LanceDB using pip:

```bash
pip install lancedb
```

## Basic Usage

Here's a simple example to get you started:

```python
import lancedb
import numpy as np

# Create a new database
db = lancedb.connect("./data")

# Create a table with some sample data
data = [
    {"id": 1, "vector": [1.1, 1.2, 1.3], "text": "Hello"},
    {"id": 2, "vector": [2.1, 2.2, 2.3], "text": "World"}
]

# Create the table
table = db.create_table("my_table", data=data)

# Search for similar vectors
query_vector = [1.1, 1.2, 1.3]
results = table.search(query_vector).limit(2).to_list()
```

## Key Features

- **Vector Search**: Perform fast similarity search on your data
- **Filtering**: Combine vector search with metadata filtering
- **Batch Operations**: Efficiently process large datasets
- **Persistence**: Data is automatically persisted to disk

## Next Steps

- Check out the [Installation Guide](installation.md) for detailed setup instructions
- Explore the [Core Concepts](../user-guide/core-concepts.md) to learn more about LanceDB's architecture
- Visit the [API Reference](../user-guide/api-reference.md) for detailed documentation

## Need Help?

- Join our [Discord community](https://discord.gg/lancedb)
- Check out our [GitHub repository](https://github.com/lancedb/lancedb)
- Open an issue if you find a bug or need help 