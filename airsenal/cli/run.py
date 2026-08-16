"""Commands for running the full AIrsenal pipeline."""

import typer

from airsenal.framework.bpl_interface import DEFAULT_TEAM_EPSILON
from airsenal.scripts.airsenal_run_pipeline import run_pipeline


def run(
    num_thread: int | None = typer.Option(
        None, help="Number of threads to use for the pipeline run."
    ),
    weeks_ahead: int = typer.Option(
        3, help="Number of gameweeks to include in the pipeline run."
    ),
    fpl_team_id: int | None = typer.Option(
        None, help="FPL team ID for the pipeline run."
    ),
    clean: bool = typer.Option(
        False, help="Delete and recreate the AIrsenal database."
    ),
    apply_transfers: bool = typer.Option(
        False, help="Apply suggested transfers and set the lineup through the API."
    ),
    wildcard_week: int = typer.Option(
        -1, help="Wildcard week; use 0 to consider any gameweek."
    ),
    free_hit_week: int = typer.Option(
        -1, help="Free hit week; use 0 to consider any gameweek."
    ),
    triple_captain_week: int = typer.Option(
        -1, help="Triple captain week; use 0 to consider any gameweek."
    ),
    bench_boost_week: int = typer.Option(
        -1, help="Bench boost week; use 0 to consider any gameweek."
    ),
    n_previous: int = typer.Option(
        3, help="Number of previous seasons to include when creating the database."
    ),
    no_current_season: bool = typer.Option(
        False, help="Exclude the current season when creating the database."
    ),
    team_model: str = typer.Option(
        "extended", help="Team model to fit: extended or neutral."
    ),
    epsilon: float = typer.Option(
        DEFAULT_TEAM_EPSILON,
        help="Exponential time-weighting downweight factor.",
    ),
    max_transfers: int = typer.Option(
        2, min=0, max=5, help="Maximum transfers to consider per gameweek."
    ),
    max_hit: int = typer.Option(
        8, min=0, help="Maximum points to spend on additional transfers."
    ),
    allow_unused: bool = typer.Option(
        False, help="Include strategies that waste free transfers."
    ),
    save_absences: bool = typer.Option(
        False, help="Save expected absences to a CSV file."
    ),
) -> None:
    """Run the full AIrsenal pipeline."""
    run_pipeline(
        num_thread=num_thread,
        weeks_ahead=weeks_ahead,
        fpl_team_id=fpl_team_id,
        clean=clean,
        apply_transfers=apply_transfers,
        wildcard_week=wildcard_week,
        free_hit_week=free_hit_week,
        triple_captain_week=triple_captain_week,
        bench_boost_week=bench_boost_week,
        n_previous=n_previous,
        no_current_season=no_current_season,
        team_model=team_model,
        epsilon=epsilon,
        max_transfers=max_transfers,
        max_hit=max_hit,
        allow_unused=allow_unused,
        save_absences=save_absences,
    )
