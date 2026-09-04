"""Module to handle AIrsenal environment variables and storage."""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Concatenate

from platformdirs import user_data_dir

# Cross-platform data directory. Resolved at import (it cannot change within a
# process) but *not* created: Call `airsenal_home()` when you are about to write.
if "AIRSENAL_HOME" in os.environ:
    AIRSENAL_HOME = Path(os.environ["AIRSENAL_HOME"])
else:
    AIRSENAL_HOME = Path(user_data_dir("airsenal"))


def airsenal_home() -> Path:
    """The directory AIrsenal keeps its own files in, created on first use."""
    AIRSENAL_HOME.mkdir(parents=True, exist_ok=True)
    return AIRSENAL_HOME


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

# The subset of the above that is a credential rather than a setting. `airsenal env
# get` with no argument dumps every configured value but redacts these
SECRET_ENV_KEYS = frozenset(
    {
        "FPL_PASSWORD",
        "AIRSENAL_DB_PASSWORD",
        # anyone holding the URL can post to the channel, so it is a credential
        "DISCORD_WEBHOOK",
    }
)


def check_valid_key[**P, R](
    func: Callable[Concatenate[str, P], R],
) -> Callable[Concatenate[str, P], R]:
    """Reject an unrecognised AIrsenal setting name before the wrapped call runs."""

    def wrapper(key: str, /, *args: P.args, **kwargs: P.kwargs) -> R:
        if key not in AIRSENAL_ENV_KEYS:
            msg = f"{key} is not a known AIrsenal environment variable"
            raise KeyError(msg)
        return func(key, *args, **kwargs)

    # functools.wraps would erase the Concatenate signature mypy needs here
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


@check_valid_key
def save_env(key: str, value: str) -> None:
    with open(airsenal_home() / key, "w") as f:
        f.write(value)


@check_valid_key
def delete_env(key: str) -> None:
    if os.path.exists(AIRSENAL_HOME / key):
        os.remove(AIRSENAL_HOME / key)
    if key in os.environ:
        os.unsetenv(key)
        os.environ.pop(key)


@check_valid_key
def get_env[T](key: str, return_type: Callable[[str], T]) -> T | None:
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
# Resolved once here for the callers that only read them at start-up. Anything
# that has to see an `airsenal env set` made in the same process - the database
# connection string, the Discord webhook - calls `get_env` instead.
AIRSENAL_DB_FILE = get_env("AIRSENAL_DB_FILE", str)
AIRSENAL_DB_URI = get_env("AIRSENAL_DB_URI", str)
AIRSENAL_DB_USER = get_env("AIRSENAL_DB_USER", str)
AIRSENAL_DB_PASSWORD = get_env("AIRSENAL_DB_PASSWORD", str)
