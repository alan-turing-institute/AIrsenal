"""
What AIrsenal thinks it is: version, home directory and which database it is
pointed at.

Named for what it reports rather than for the environment variables it happens
to read - core/env.py reads the environment, cli/env.py is the command, and
three modules called env.py told you nothing about which was which.
"""

from airsenal import __version__
from airsenal.core.env import (
    AIRSENAL_ENV_KEYS,
    AIRSENAL_HOME,
    get_env,
)
from airsenal.core.logging import get_logger
from airsenal.db.engine import get_connection_string

logger = get_logger(__name__)


def redact_db_password(conn_str: str) -> str:
    # Only redact for postgresql connection strings
    if conn_str.startswith("postgresql://"):
        # Format: postgresql://user:password@host/dbname
        # Find the user:password part
        prefix = "postgresql://"
        rest = conn_str[len(prefix) :]
        if "@" in rest:
            creds, host_db = rest.split("@", 1)
            if ":" in creds:
                user, _ = creds.split(":", 1)
                return f"{prefix}{user}:***@{host_db}"
    return conn_str


def print_env() -> None:
    logger.info("AIRSENAL_VERSION: %s", __version__)
    logger.info("AIRSENAL_HOME: %s", AIRSENAL_HOME)
    conn_str = get_connection_string()
    logger.info("DB_CONNECTION_STRING: %s", redact_db_password(conn_str))
    for k in AIRSENAL_ENV_KEYS:
        if value := get_env(k, str):
            logger.info("%s: %s", k, value)
