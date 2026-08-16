from airsenal import __version__
from airsenal.framework.env import (
    AIRSENAL_ENV_KEYS,
    AIRSENAL_HOME,
    get_env,
)
from airsenal.framework.output import print
from airsenal.framework.schema import get_connection_string


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


def print_env():
    print(f"AIRSENAL_VERSION: {__version__}")
    print(f"AIRSENAL_HOME: {AIRSENAL_HOME}")
    conn_str = get_connection_string()
    print(f"DB_CONNECTION_STRING: {redact_db_password(conn_str)}")
    for k in AIRSENAL_ENV_KEYS:
        if value := get_env(k, str):
            print(f"{k}: {value}")
