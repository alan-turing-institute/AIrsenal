"""Commands for creating and updating the AIrsenal database."""

from typing import Annotated

import typer

from airsenal.domain.season import CURRENT_SEASON
from airsenal.ingest.checks import run_all_checks
from airsenal.ingest.init_db import create_database
from airsenal.ingest.update import update_database

app = typer.Typer(no_args_is_help=True, help="Create and update the AIrsenal database.")


@app.command()
def create(
    fpl_team_id: Annotated[int | None, typer.Option(help="FPL team ID.")] = None,
    clean: Annotated[
        bool, typer.Option(help="Delete and recreate an existing database.")
    ] = False,
    n_previous: Annotated[
        int, typer.Option(min=1, help="Number of previous seasons to include.")
    ] = 3,
    no_current_season: Annotated[
        bool, typer.Option(help="Exclude the current season from the database.")
    ] = False,
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
    season: Annotated[
        str, typer.Option(help="Season in the form 2526.")
    ] = CURRENT_SEASON,
    noattr: Annotated[
        bool, typer.Option(help="Do not update player attributes.")
    ] = False,
    fpl_team_id: Annotated[int | None, typer.Option(help="FPL team ID.")] = None,
) -> None:
    """Update the AIrsenal database from current FPL data."""
    update_database(season=season, noattr=noattr, fpl_team_id=fpl_team_id)


@app.command()
def check() -> None:
    """Run database data-sanity checks."""
    run_all_checks()
