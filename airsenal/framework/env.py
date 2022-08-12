"""
Database can be either an sqlite file or a postgress server
"""
import os
from pathlib import Path

from platformdirs import user_data_dir

AIRSENAL_ENV_KEYS = [
    "FPL_TEAM_ID",
    "FPL_LOGIN",
    "FPL_PASSWORD",
    "FPL_LEAGUE_ID",
    "AIRSENAL_DB_FILE",
    "AIRSENAL_DB_URI",
    "AIRSENAL_DB_USER",
    "AIRSENAL_DB_PASSWORD",
    "DISCORD_WEBHOOK",
]

# Cross-platform data directory
if "AIRSENAL_HOME" in os.environ.keys():
    AIRSENAL_HOME = os.environ["AIRSENAL_HOME"]
else:
    AIRSENAL_HOME = Path(user_data_dir("airsenal"))
os.makedirs(AIRSENAL_HOME, exist_ok=True)


def check_valid_key(func):
    """decorator to pre-check whether we are using a valid AIrsenal key in env
    get/save/del functions"""

    def wrapper(key, *args, **kwargs):
        if key not in AIRSENAL_ENV_KEYS:
            raise KeyError(f"{key} is not a known AIrsenal environment variable")
        return func(key, *args, **kwargs)

    return wrapper


@check_valid_key
def get_env(key, default=None):
    if key in os.environ.keys():
        return os.environ[key]
    if os.path.exists(AIRSENAL_HOME / key):
        with open(AIRSENAL_HOME / key) as f:
            return f.read().strip()
    return default


@check_valid_key
def save_env(key, value):
    with open(AIRSENAL_HOME / key, "w") as f:
        f.write(value)


@check_valid_key
def delete_env(key):
    if os.path.exists(AIRSENAL_HOME / key):
        os.remove(AIRSENAL_HOME / key)
    if key in os.environ.keys():
        os.unsetenv(key)
        os.environ.pop(key)


if get_env("AIRSENAL_DB_FILE") and get_env("AIRSENAL_DB_URI"):
    raise RuntimeError("Please choose only ONE of AIRSENAL_DB_FILE and AIRSENAL_DB_URI")

# sqlite database in a local file with path specified by AIRSENAL_DB_FILE,
# or AIRSENAL_HOME / data.db by default
DB_CONNECTION_STRING = (
    f"sqlite:///{get_env('AIRSENAL_DB_FILE', default=AIRSENAL_HOME / 'data.db')}"
)

# postgres database specified by: AIRSENAL_DB{_URI, _USER, _PASSWORD}
if get_env("AIRSENAL_DB_URI"):
    keys = ["AIRSENAL_DB_URI", "AIRSENAL_DB_USER", "AIRSENAL_DB_PASSWORD"]
    params = {}
    for k in keys:
        if value := get_env(k):
            params[k] = value
        else:
            raise KeyError(f"{k} must be defined when using a postgres database")

    DB_CONNECTION_STRING = (
        f"postgres://{params['AIRSENAL_DB_USER']}:"
        f"{params['AIRSENAL_DB_PASSWORD']}@{params['AIRSENAL_DB_URI']}/airsenal"
    )
