"""
Database can be either an sqlite file or a postgress server
"""

import os
from airsenal import TMPDIR

# Default connection string points to a local sqlite file in
# airsenal/data/data.db

DB_CONNECTION_STRING = "sqlite:///{}/data.db".format(TMPDIR)

# Check that we're not trying to set location for both sqlite and postgres
if "AIrsenalDBFile" in os.environ.keys() and "AIrsenalDBUri" in os.environ.keys():
    raise RuntimeError("Please choose only ONE of AIrsenalDBFile and AIrsenalDBUri")

# location of sqlite file overridden by an env var
if "AIrsenalDBFile" in os.environ.keys():
    DB_CONNECTION_STRING = "sqlite:///{}".format(os.environ["AIrsenalDBFile"])

# location of postgres server
if "AIrsenalDBUri" in os.environ.keys():
    DB_CONNECTION_STRING = "postgres://{}:{}@{}/airsenal".format(
        os.environ["AIrsenalDBUser"],
        os.environ["AIrsenalDBPassword"],
        os.environ["AIrsenalDBUri"],
    )
