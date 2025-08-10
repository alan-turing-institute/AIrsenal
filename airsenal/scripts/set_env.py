import argparse

from airsenal import __version__
from airsenal.framework.env import (
    AIRSENAL_ENV_KEYS,
    AIRSENAL_HOME,
    delete_env,
    get_env,
    save_env,
)
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


def main():
    parser = argparse.ArgumentParser(
        description="Show and set AIrsenal environment variables"
    )

    parser.add_argument(
        "cmd",
        choices=["get", "set", "del", "names"],
        help="whether to set/get/delete environment variables, or show all valid names",
    )

    parser.add_argument(
        "-k", "--key", help="name of environment variable", required=False
    )

    parser.add_argument(
        "-v", "--value", help="value to assign to environment variable", required=False
    )

    args = parser.parse_args()
    if args.cmd == "get":
        if args.value:
            msg = "value should not be given if getting variables"
            raise ValueError(msg)
        if args.key:
            print(f"{args.key}: {get_env(args.key, str)}")
        else:
            print_env()
    if args.cmd == "set":
        if args.key and args.value:
            save_env(args.key, args.value)
        else:
            msg = "key and value must not be given if getting variables"
            raise ValueError(msg)
    if args.cmd == "del":
        if not args.key:
            msg = "key must be given if deleting variables"
            raise ValueError(msg)
        if args.value:
            msg = "value should not be given if deleting variables"
            raise ValueError(msg)
        delete_env(args.key)

    if args.cmd == "names":
        if args.value or args.key:
            msg = "value should not be given if deleting variables"
            raise ValueError(msg)
        print(AIRSENAL_ENV_KEYS)


if __name__ == "__main__":
    main()
