# --8<-- [start:libraries]
import lancedb
import pandas as pd
import numpy as np
import pyarrow as pa
import os
# --8<-- [end:libraries]

# --8<-- [start:install]
pip install lancedb
# --8<-- [end:install]

# --8<-- [start:connect_oss]
import lancedb
import pandas as pd
import pyarrow as pa

uri = "data/sample-lancedb"
db = lancedb.connect(uri)
# --8<-- [end:connect_oss]

# --8<-- [start:connect_cloud]
uri = "db://your-database-uri"
api_key = "your-api-key"
region = "us-east-1"
# --8<-- [end:connect_cloud]

# --8<-- [start:install_preview]
pip install --pre --extra-index-url https://pypi.fury.io/lancedb/ lancedb
# --8<-- [end:install_preview]

