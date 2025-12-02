# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The LanceDB Authors

from pathlib import Path
import shutil

def test_connection():
    # --8<-- [start:connect]
    import lancedb

    uri = "ex_lancedb"
    db = lancedb.connect(uri)
    # --8<-- [end:connect]
    assert db is not None
    shutil.rmtree(uri, ignore_errors=True)
    assert not Path(uri).exists()


# --8<-- [start:connect_cloud]
uri = "db://your-database-uri"
api_key = "your-api-key"
region = "us-east-1"
# --8<-- [end:connect_cloud]