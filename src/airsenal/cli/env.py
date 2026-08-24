"""Commands for AIrsenal environment configuration."""

from typing import Annotated

import typer

from airsenal.core.env import AIRSENAL_ENV_KEYS, delete_env, get_env, save_env
from airsenal.reporting.diagnostics import print_env

app = typer.Typer(
    no_args_is_help=True, help="Configure AIrsenal environment variables."
)


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
