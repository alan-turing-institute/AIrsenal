"""Commands for archiving AIrsenal and FPL data."""

from typing import Annotated

import typer

from airsenal.core.season import CURRENT_SEASON
from airsenal.export.absences import main as save_absences
from airsenal.export.api_dump import main as dump_api
from airsenal.export.attributes import main as save_attributes
from airsenal.export.db_dump import main as dump_db
from airsenal.fetch.transfermarkt import scrape_transfermarkt

app = typer.Typer(
    no_args_is_help=True, help="Archive AIrsenal and FPL data for the current season."
)


@app.command()
def api() -> None:
    """Save FPL API data and related sources for the current season."""
    dump_api()


@app.command()
def db() -> None:
    """Save database tables as CSV files."""
    dump_db()


@app.command()
def transfermarkt(
    season: Annotated[
        str, typer.Option(help="Season in the form 2526.")
    ] = CURRENT_SEASON,
) -> None:
    """Save Transfermarkt absence data."""
    scrape_transfermarkt([season])


@app.command()
def absences() -> None:
    """Save expected player absences for the current season."""
    save_absences()


@app.command()
def attributes() -> None:
    """Append the current player attributes to the packaged history file."""
    save_attributes()
