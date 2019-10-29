"""
Database can be either an sqlite file or a postgress server
"""

import os
from .. import AIRSENAL_HOME

## Default connection string points to a local sqlite file in
## airsenal/data/data.db
DB_CONNECTION_STRING = "sqlite:///{}/data/data.db".format(AIRSENAL_HOME)

## Check that we're not trying to set location for both sqlite and postgres
if "AIrsenalDBFile" in os.environ.keys() and \
   "AIrsenalDBUri" in os.environ.keys():
    raise RuntimeError("Please choose only ONE of AIrsenalDBFile and AIrsenalDBUri")

## location of sqlite file overridden by an env var
if "AIrsenalDBFile" in os.environ.keys():
    DB_CONNECTION_STRING = "sqlite:///{}".format(os.environ["AIrsenalDBFile"])

## location of postgres server
if "AIrsenalDBUri" in os.environ.keys():
    DB_CONNECTION_STRING = "postgres://{}/airsenal?check_same_thread=False".format(os.environ["AIrsenalDBUri"])
