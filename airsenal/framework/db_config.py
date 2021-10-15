"""
Database can be either an sqlite file or a postgress server
"""
import os

from airsenal import TMPDIR

config_path = os.path.join(os.path.dirname(__file__), "..", "data")
AIrsenalDBFile_path = os.path.join(config_path, "AIrsenalDBFile")
AIrsenalDBUri_path = os.path.join(config_path, "AIrsenalDBUri")
AIrsenalDBUser_path = os.path.join(config_path, "AIrsenalDBUser")
AIrsenalDBPassword_path = os.path.join(config_path, "AIrsenalDBPassword")

# Check that we're not trying to set location for both sqlite and postgres
if ("AIrsenalDBFile" in os.environ.keys() or os.path.exists(AIrsenalDBFile_path)) and (
    "AIrsenalDBUri" in os.environ.keys() or os.path.exists(AIrsenalDBUri_path)
):
    raise RuntimeError("Please choose only ONE of AIrsenalDBFile and AIrsenalDBUri")

# sqlite database in a local file with path specified by:
# - AIrsenalDBFile environment variable
# - airsenal/data/AIrsenalDBFile file
# - platform-dependent temporary directory (default)
if "AIrsenalDBFile" in os.environ.keys():
    AIrsenalDBFile = os.environ["AIrsenalDBFile"]
elif os.path.exists(AIrsenalDBFile_path):
    AIrsenalDBFile = open(AIrsenalDBFile_path).read().strip()
else:
    AIrsenalDBFile = os.path.join(TMPDIR, "data.db")

DB_CONNECTION_STRING = "sqlite:///{}".format(AIrsenalDBFile)

# postgres database specified by: AIrsenalDBUri, AIrsenalDBUser, AIrsenalDBPassword
# defined either as:
# - environment variables
# - Files in airsenal/data/
if "AIrsenalDBUri" in os.environ.keys() or os.path.exists(AIrsenalDBUri_path):
    if "AIrsenalDBUser" in os.environ.keys():
        AIrsenalDBUser = os.environ["AIrsenalDBUser"]
    elif os.path.exists(AIrsenalDBUser_path):
        AIrsenalDBUser = open(AIrsenalDBUser_path).read().strip()
    else:
        raise RuntimeError(
            "AIrsenalDBUser must be defined when using a postgres database"
        )

    if "AIrsenalDBUser" in os.environ.keys():
        AIrsenalDBPassword = os.environ["AIrsenalDBPassword"]
    elif os.path.exists(AIrsenalDBPassword_path):
        AIrsenalDBPassword = open(AIrsenalDBPassword_path).read().strip()
    else:
        raise RuntimeError(
            "AIrsenalDBPassword must be defined when using a postgres database"
        )

    if "AIrsenalDBUri" in os.environ.keys():
        AIrsenalDBUri = os.environ["AIrsenalDBUri"]
    elif os.path.exists(AIrsenalDBUri_path):
        AIrsenalDBUri = open(AIrsenalDBUri_path).read().strip()
    else:
        raise RuntimeError(
            "AIrsenalDBUri must be defined when using a postgres database"
        )

    DB_CONNECTION_STRING = "postgres://{}:{}@{}/airsenal".format(
        AIrsenalDBUser,
        AIrsenalDBPassword,
        AIrsenalDBUri,
    )
