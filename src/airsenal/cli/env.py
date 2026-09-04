"""Commands for AIrsenal environment configuration."""

from typing import Annotated

import typer

from airsenal import __version__
from airsenal.core.env import (
    AIRSENAL_ENV_KEYS,
    AIRSENAL_HOME,
    SECRET_ENV_KEYS,
    delete_env,
    get_env,
    save_env,
)
from airsenal.core.logging import get_logger
from airsenal.db.engine import get_connection_string

logger = get_logger(__name__)

app = typer.Typer(
    no_args_is_help=True, help="Configure AIrsenal environment variables."
)


def redact_db_password(conn_str: str) -> str:
    """
    Replace the password in a connection string with `***`.

    Only postgres URLs that parse as `postgresql://user:password@host/db` are redacted.
    Other connection strings are left unchanged and may contain secrets.
    """
    if conn_str.startswith("postgresql://"):
        # Format: postgresql://user:password@host/dbname
        prefix = "postgresql://"
        rest = conn_str[len(prefix) :]
        if "@" in rest:
            creds, host_db = rest.split("@", 1)
            if ":" in creds:
                user, _ = creds.split(":", 1)
                return f"{prefix}{user}:***@{host_db}"
    return conn_str


def print_env() -> None:
    """
    Show what AIrsenal is configured with, without printing any credential.

    Values named in `SECRET_ENV_KEYS` are reported as set or not rather than
    echoed. `airsenal env get FPL_PASSWORD` still shows one secret when it is asked for
    by name.
    """
    logger.info("AIRSENAL_VERSION: %s", __version__)
    logger.info("AIRSENAL_HOME: %s", AIRSENAL_HOME)
    conn_str = get_connection_string()
    logger.info("DB_CONNECTION_STRING: %s", redact_db_password(conn_str))
    for k in AIRSENAL_ENV_KEYS:
        if value := get_env(k, str):
            logger.info("%s: %s", k, "***" if k in SECRET_ENV_KEYS else value)


@app.command()
def get(
    key: Annotated[
        str | None, typer.Argument(help="Environment variable name.")
    ] = None,
) -> None:
    """Show one environment value or all configured values."""
    if key:
        typer.echo(f"{key}: {get_env(key, str)}")
    else:
        print_env()


@app.command()
def set(
    key: Annotated[str, typer.Argument(help="Environment variable name.")],
    value: Annotated[str, typer.Argument(help="Environment variable value.")],
) -> None:
    """Save an environment value."""
    save_env(key, value)


@app.command("delete")
def delete(
    key: Annotated[str, typer.Argument(help="Environment variable name.")],
) -> None:
    """Delete an environment value."""
    delete_env(key)


@app.command()
def names() -> None:
    """List valid environment variable names."""
    for key in AIRSENAL_ENV_KEYS:
        typer.echo(key)
