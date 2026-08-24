"""Resolving which database to talk to."""

from airsenal.core.env import (
    AIRSENAL_DB_FILE,
    AIRSENAL_DB_PASSWORD,
    AIRSENAL_DB_URI,
    AIRSENAL_DB_USER,
    AIRSENAL_HOME,
)


def get_connection_string() -> str:
    if AIRSENAL_DB_FILE and AIRSENAL_DB_URI:
        msg = "Please choose only ONE of AIRSENAL_DB_FILE and AIRSENAL_DB_URI"
        raise RuntimeError(msg)

    # postgres database specified by: AIRSENAL_DB{_URI, _USER, _PASSWORD}
    if AIRSENAL_DB_URI:
        if AIRSENAL_DB_PASSWORD is None:
            msg = "AIRSENAL_DB_PASSWORD must be defined when using a postgres database"
            raise KeyError(msg)
        if AIRSENAL_DB_USER is None:
            msg = "AIRSENAL_DB_USER must be defined when using a postgres database"
            raise KeyError(msg)

        return (
            f"postgresql://{AIRSENAL_DB_USER}:"
            f"{AIRSENAL_DB_PASSWORD}@{AIRSENAL_DB_URI}/airsenal"
        )

    # sqlite database in a local file with path specified by AIRSENAL_DB_FILE,
    # or AIRSENAL_HOME / data.db by default
    if not AIRSENAL_DB_FILE:
        return f"sqlite:///{AIRSENAL_HOME / 'data.db'}"
    return f"sqlite:///{AIRSENAL_DB_FILE}"
