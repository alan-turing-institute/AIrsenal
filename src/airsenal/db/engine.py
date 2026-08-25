"""
Resolving which database to talk to.

The settings are read through the `env` module rather than copied out of it, so
that a value replaced after import - by `airsenal env set`, or by the test
suite pointing at a temporary directory - is the one used.
"""

from airsenal.core import env


def get_connection_string() -> str:
    """
    The SQLAlchemy URL for the configured database.

    `AIRSENAL_DB_FILE` and `AIRSENAL_DB_URI` are mutually exclusive. `_URI`
    selects postgres and requires `AIRSENAL_DB_USER` and `AIRSENAL_DB_PASSWORD`
    with it; otherwise the database is SQLite, at `AIRSENAL_DB_FILE` or at
    `$AIRSENAL_HOME/data.db`.
    """
    if env.AIRSENAL_DB_FILE and env.AIRSENAL_DB_URI:
        msg = "Please choose only ONE of AIRSENAL_DB_FILE and AIRSENAL_DB_URI"
        raise RuntimeError(msg)

    # postgres database specified by: AIRSENAL_DB{_URI, _USER, _PASSWORD}
    if env.AIRSENAL_DB_URI:
        if env.AIRSENAL_DB_PASSWORD is None:
            msg = "AIRSENAL_DB_PASSWORD must be defined when using a postgres database"
            raise KeyError(msg)
        if env.AIRSENAL_DB_USER is None:
            msg = "AIRSENAL_DB_USER must be defined when using a postgres database"
            raise KeyError(msg)

        return (
            f"postgresql://{env.AIRSENAL_DB_USER}:"
            f"{env.AIRSENAL_DB_PASSWORD}@{env.AIRSENAL_DB_URI}/airsenal"
        )

    # sqlite database in a local file with path specified by AIRSENAL_DB_FILE,
    # or AIRSENAL_HOME / data.db by default
    if not env.AIRSENAL_DB_FILE:
        return f"sqlite:///{env.airsenal_home() / 'data.db'}"
    return f"sqlite:///{env.AIRSENAL_DB_FILE}"
