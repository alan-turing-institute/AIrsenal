"""
Database can be either an sqlite file or a postgress server
"""
import os
from pathlib import Path

from platformdirs import user_data_dir

AIRSENAL_ENV_KEYS = {  # dict of name then function to  convert str to correct type
    "FPL_TEAM_ID": int,
    "FPL_LOGIN": str,
    "FPL_PASSWORD": str,
    "FPL_LEAGUE_ID": int,
    "AIRSENAL_DB_FILE": str,
    "AIRSENAL_DB_URI": str,
    "AIRSENAL_DB_USER": str,
    "AIRSENAL_DB_PASSWORD": str,
    "DISCORD_WEBHOOK": str,
}

# Cross-platform data directory
if "AIRSENAL_HOME" in os.environ.keys():
    AIRSENAL_HOME = Path(os.environ["AIRSENAL_HOME"])
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
        return AIRSENAL_ENV_KEYS[key](os.environ[key])
    if os.path.exists(AIRSENAL_HOME / key):
        with open(AIRSENAL_HOME / key) as f:
            return AIRSENAL_ENV_KEYS[key](f.read().strip())
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
