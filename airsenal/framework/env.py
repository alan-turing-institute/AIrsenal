"""Module to handle AIrsenal environment variables and storage."""

import os
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from platformdirs import user_data_dir

# Cross-platform data directory
if "AIRSENAL_HOME" in os.environ:
    AIRSENAL_HOME = Path(os.environ["AIRSENAL_HOME"])
else:
    AIRSENAL_HOME = Path(user_data_dir("airsenal"))
os.makedirs(AIRSENAL_HOME, exist_ok=True)


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


def check_valid_key(func):
    """decorator to pre-check whether we are using a valid AIrsenal key in env
    get/save/del functions"""

    def wrapper(key, *args, **kwargs):
        if key not in AIRSENAL_ENV_KEYS:
            msg = f"{key} is not a known AIrsenal environment variable"
            raise KeyError(msg)
        return func(key, *args, **kwargs)

    return wrapper


@check_valid_key
def save_env(key, value):
    with open(AIRSENAL_HOME / key, "w") as f:
        f.write(value)


@check_valid_key
def delete_env(key):
    if os.path.exists(AIRSENAL_HOME / key):
        os.remove(AIRSENAL_HOME / key)
    if key in os.environ:
        os.unsetenv(key)
        os.environ.pop(key)


T = TypeVar("T")


@check_valid_key
def get_env(key: str, return_type: Callable[[str], T]) -> T | None:
    if key in os.environ:
        return return_type(os.environ[key])
    if os.path.exists(AIRSENAL_HOME / key):
        with open(AIRSENAL_HOME / key) as f:
            return return_type(f.read().strip())
    return None


try:
    FPL_TEAM_ID = get_env("FPL_TEAM_ID", int)
    FPL_LEAGUE_ID = get_env("FPL_LEAGUE_ID", int)
except ValueError as e:
    msg = (
        "FPL_TEAM_ID and FPL_LEAGUE_ID must be valid integers if set. "
        "Please check your environment variables/files."
    )
    raise ValueError(msg) from e

FPL_LOGIN = get_env("FPL_LOGIN", str)
FPL_PASSWORD = get_env("FPL_PASSWORD", str)
AIRSENAL_DB_FILE = get_env("AIRSENAL_DB_FILE", str)
AIRSENAL_DB_URI = get_env("AIRSENAL_DB_URI", str)
AIRSENAL_DB_USER = get_env("AIRSENAL_DB_USER", str)
AIRSENAL_DB_PASSWORD = get_env("AIRSENAL_DB_PASSWORD", str)
DISCORD_WEBHOOK = get_env("DISCORD_WEBHOOK", str)
