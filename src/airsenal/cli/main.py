"""Root command-line application."""

import logging
import sys
from typing import Annotated

import typer

import airsenal
from airsenal.cli.apply import app as apply_app
from airsenal.cli.db import app as db_app
from airsenal.cli.dump import app as dump_app
from airsenal.cli.env import app as env_app
from airsenal.cli.optimize import app as optimize_app
from airsenal.cli.plot import plot
from airsenal.cli.predict import predict
from airsenal.cli.replay import replay
from airsenal.cli.run import run
from airsenal.core.logging import configure_logging
from airsenal.core.registry import ConfigError

app = typer.Typer(
    no_args_is_help=True,
    help=(
        "AIrsenal: A package for using Machine learning to pick a Fantasy Premier "
        "League team.\n\nHomepage: https://github.com/alan-turing-institute/AIrsenal\n"
        f"Version: {airsenal.__version__}"
    ),
)


@app.callback()
def main(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show debug-level output.")
    ] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Only show warnings and errors.")
    ] = False,
) -> None:
    """AIrsenal command-line interface."""
    if verbose and quiet:
        msg = "--verbose and --quiet cannot be used together."
        raise typer.BadParameter(msg)
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    configure_logging(level)


app.command()(run)
app.add_typer(db_app, name="db")
app.command()(predict)
app.add_typer(optimize_app, name="optimize")
app.add_typer(env_app, name="env")
app.add_typer(apply_app, name="apply")
app.add_typer(dump_app, name="dump")
app.command()(replay)
app.command()(plot)


def main_cli() -> None:
    """
    Entry point: report an unusable model or option as a bad option.

    Without this, `--set-player nope=1` exits with a full traceback, which reads
    as a crash rather than as "you typed something I do not recognise".
    """
    try:
        app()
    except ConfigError as e:
        typer.secho(f"Error: {e}", err=True, fg=typer.colors.RED)
        sys.exit(2)
