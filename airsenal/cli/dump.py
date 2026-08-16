"""Commands for archiving AIrsenal and FPL data."""

import typer

from airsenal.framework.season import CURRENT_SEASON
from airsenal.scripts.dump_api import main as dump_api
from airsenal.scripts.dump_db_contents import main as dump_db
from airsenal.scripts.save_expected_absences import main as save_absences
from airsenal.scripts.scrape_transfermarkt import scrape_transfermarkt

app = typer.Typer(no_args_is_help=True)


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
    season: str = typer.Option(CURRENT_SEASON, help="Season in the form 2526."),
    verbose: bool = typer.Option(False, help="Print detailed scraping output."),
) -> None:
    """Save Transfermarkt absence data."""
    scrape_transfermarkt([season], verbose=verbose)


@app.command()
def absences() -> None:
    """Save expected player absences for the current season."""
    save_absences()
