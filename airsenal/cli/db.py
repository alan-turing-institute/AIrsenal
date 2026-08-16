"""Commands for creating and updating the AIrsenal database."""

import typer

from airsenal.framework.season import CURRENT_SEASON
from airsenal.scripts.data_sanity_checks import run_all_checks
from airsenal.scripts.fill_db_init import create_database
from airsenal.scripts.update_db import update_database

app = typer.Typer(no_args_is_help=True)


@app.command()
def create(
    fpl_team_id: int | None = typer.Option(None, help="FPL team ID."),
    clean: bool = typer.Option(False, help="Delete and recreate an existing database."),
    n_previous: int = typer.Option(
        3, min=1, help="Number of previous seasons to include."
    ),
    no_current_season: bool = typer.Option(
        False, help="Exclude the current season from the database."
    ),
) -> None:
    """Create the AIrsenal database."""
    create_database(
        fpl_team_id=fpl_team_id,
        clean=clean,
        n_previous=n_previous,
        no_current_season=no_current_season,
    )


@app.command()
def update(
    season: str = typer.Option(CURRENT_SEASON, help="Season in the form 2526."),
    noattr: bool = typer.Option(False, help="Do not update player attributes."),
    fpl_team_id: int | None = typer.Option(None, help="FPL team ID."),
) -> None:
    """Update the AIrsenal database from current FPL data."""
    update_database(season=season, noattr=noattr, fpl_team_id=fpl_team_id)


@app.command()
def check() -> None:
    """Run database data-sanity checks."""
    run_all_checks()
