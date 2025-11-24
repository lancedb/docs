# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The LanceDB Authors

import shutil
from datetime import timedelta
from pathlib import Path

from lancedb.db import AsyncConnection, DBConnection
import lancedb
import pytest
import pytest_asyncio


@pytest.fixture(scope="session")
def example_dir(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("lancedb_examples")
    yield path
    shutil.rmtree(path, ignore_errors=True)


# Use an in-memory database for most tests.
@pytest.fixture
def mem_db() -> DBConnection:
    return lancedb.connect("memory://")


# Use a shared temporary directory when we need to inspect the database files.
@pytest.fixture
def tmp_db(example_dir) -> DBConnection:
    return lancedb.connect(str(example_dir / "sync_db"))


@pytest_asyncio.fixture
async def mem_db_async() -> AsyncConnection:
    return await lancedb.connect_async("memory://")


@pytest_asyncio.fixture
async def tmp_db_async(example_dir) -> AsyncConnection:
    return await lancedb.connect_async(
        str(example_dir / "async_db"), read_consistency_interval=timedelta(seconds=0)
    )
