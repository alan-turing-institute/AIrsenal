"""Commands for AIrsenal environment configuration."""

import typer

from airsenal.framework.env import AIRSENAL_ENV_KEYS, delete_env, get_env, save_env
from airsenal.scripts.set_env import print_env

app = typer.Typer(no_args_is_help=True)


@app.command()
def get(
    key: str | None = typer.Option(None, help="Environment variable name."),
) -> None:
    """Show one environment value or all configured values."""
    if key:
        typer.echo(f"{key}: {get_env(key, str)}")
    else:
        print_env()


@app.command()
def set(
    key: str = typer.Option(..., help="Environment variable name."),
    value: str = typer.Option(..., help="Environment variable value."),
) -> None:
    """Save an environment value."""
    save_env(key, value)


@app.command("delete")
def delete(key: str = typer.Option(..., help="Environment variable name.")) -> None:
    """Delete an environment value."""
    delete_env(key)


@app.command()
def names() -> None:
    """List valid environment variable names."""
    typer.echo(AIRSENAL_ENV_KEYS)
