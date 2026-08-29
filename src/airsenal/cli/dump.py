"""Commands for archiving AIrsenal and FPL data."""

import typer

from airsenal.cli import options
from airsenal.export.absences import save_expected_absences
from airsenal.export.api_dump import dump_api
from airsenal.export.attributes import save_attributes
from airsenal.export.db_dump import dump_db
from airsenal.remote.transfermarkt import scrape_transfermarkt

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
    season: options.Season = options.DEFAULT_SEASON,
) -> None:
    """Save Transfermarkt absence data."""
    scrape_transfermarkt([season])


@app.command()
def absences() -> None:
    """Save expected player absences for the current season."""
    save_expected_absences()


@app.command()
def attributes() -> None:
    """Append the current player attributes to the packaged history file."""
    save_attributes()
