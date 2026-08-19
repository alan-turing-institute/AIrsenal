"""Commands for archiving AIrsenal and FPL data."""

from typing import Annotated

import typer

from airsenal.framework.season import CURRENT_SEASON
from airsenal.scripts.dump_api import main as dump_api
from airsenal.scripts.dump_db_contents import main as dump_db
from airsenal.scripts.save_expected_absences import main as save_absences
from airsenal.scripts.scrape_transfermarkt import scrape_transfermarkt

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
