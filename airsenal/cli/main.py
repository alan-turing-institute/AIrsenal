"""Root command-line application."""

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

app = typer.Typer(
    no_args_is_help=True,
    help=(
        "AIrsenal: A package for using Machine learning to pick a Fantasy Premier "
        "League team.\n\nHomepage: https://github.com/alan-turing-institute/AIrsenal\n"
        f"Version: {airsenal.__version__}"
    ),
)
app.command()(run)
app.add_typer(db_app, name="db")
app.command()(predict)
app.add_typer(optimize_app, name="optimize")
app.add_typer(env_app, name="env")
app.add_typer(apply_app, name="apply")
app.add_typer(dump_app, name="dump")
app.command()(replay)
app.command()(plot)
