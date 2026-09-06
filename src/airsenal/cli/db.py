"""Commands for creating and updating the AIrsenal database."""

from typing import Annotated

import typer

from airsenal.cli import options
from airsenal.game.season import CURRENT_SEASON
from airsenal.ingest.checks import run_all_checks
from airsenal.ingest.init_db import create_database
from airsenal.ingest.update import update_database
from airsenal.pipeline.settings import DEFAULT_N_PREVIOUS

app = typer.Typer(no_args_is_help=True, help="Create and update the AIrsenal database.")


@app.command()
def create(
    fpl_team_id: options.FplTeamId = None,
    clean: options.Clean = False,
    n_previous: options.NPrevious = DEFAULT_N_PREVIOUS,
    current_season: options.CurrentSeason = True,
) -> None:
    """Create the AIrsenal database."""
    create_database(
        fpl_team_id=fpl_team_id,
        clean=clean,
        n_previous=n_previous,
        include_current_season=current_season,
    )


@app.command()
def update(
    attributes: Annotated[bool, typer.Option(help="Update player attributes.")] = True,
    fpl_team_id: options.FplTeamId = None,
) -> None:
    """Update the AIrsenal database from current FPL data."""
    # No --season: the FPL API only serves the current one, so it was an option
    # whose every other value silently filed this season's data under another.
    update_database(
        season=CURRENT_SEASON, attributes=attributes, fpl_team_id=fpl_team_id
    )


@app.command()
def check() -> None:
    """Run database data-sanity checks."""
    run_all_checks()
