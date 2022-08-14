import argparse

from airsenal import __version__
from airsenal.framework.env import (
    AIRSENAL_ENV_KEYS,
    AIRSENAL_HOME,
    delete_env,
    get_env,
    save_env,
)
from airsenal.framework.schema import session


def print_env():
    print(f"AIRSENAL_VERSION: {__version__}")
    print(f"AIRSENAL_HOME: {AIRSENAL_HOME}")
    print(f"DB_CONNECTION_STRING: {session.bind.url}")
    for k in AIRSENAL_ENV_KEYS:
        if value := get_env(k):
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
            raise ValueError("value should not be given if getting variables")
        if args.key:
            print(f"{args.key}: {get_env(args.key)}")
        else:
            print_env()
    if args.cmd == "set":
        if args.key and args.value:
            save_env(args.key, args.value)
        else:
            raise ValueError("key and value must not be given if getting variables")
    if args.cmd == "del":
        if not args.key:
            raise ValueError("key must be given if deleting variables")
        if args.value:
            raise ValueError("value should not be given if deleting variables")
        delete_env(args.key)

    if args.cmd == "names":
        if args.value or args.key:
            raise ValueError("value should not be given if deleting variables")
        print(AIRSENAL_ENV_KEYS)


if __name__ == "__main__":
    main()
