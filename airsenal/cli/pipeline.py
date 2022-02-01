#!/usr/bin/env python3
import click

from airsenal.scripts.airsenal_run_pipeline import run_pipeline


@click.command()
@click.option(
    "--num-threads",
    type=int,
    help="No. of threads to use for pipeline run",
)
@click.option(
    "--num-weeks", type=int, default=3, help="Number of weeks to use for pipeline run"
)
@click.option(
    "--fpl-team-id",
    type=int,
    required=False,
    help="FPL team id for pipeline run",
)
@click.option(
    "--clean",
    is_flag=True,
    help="If set, delete and recreate the AIrsenal database",
)
@click.option(
    "--apply-transfers",
    is_flag=True,
    help="If set, go ahead and make the transfers via the API.",
)
@click.option(
    "--wildcard-week",
    type=int,
    help=(
        "If set to 0, consider playing wildcard in any gameweek. "
        "If set to a specific gameweek, it'll be played for that particular gameweek."
    ),
)
@click.option(
    "--free-hit-week",
    type=int,
    help="Play free hit in the specified week. Choose 0 for 'any week'.",
)
@click.option(
    "--triple-captain-week",
    type=int,
    help="Play triple captain in the specified week. Choose 0 for 'any week'.",
)
@click.option(
    "--bench-boost-week",
    type=int,
    help="Play bench_boost in the specified week. Choose 0 for 'any week'.",
)
def run(
    num_threads: int,
    num_weeks: int,
    fpl_team_id: int,
    clean: bool,
    apply_transfers: bool,
    wildcard_week: bool,
    free_hit_week: bool,
    triple_captain_week: bool,
    bench_boost_week: bool,
):
    """
    Run the complete `airsenal` pipeline.

    This runs all the steps of the pipeline, starting from setting up a
    database with information for players, teams, fixtures, and results.
    If the database already existed, then it updates the database to the
    latest info. The predictions are then run, to get a score estimate
    for every player, which are stored, and then optimized to choose the
    best squad.
    """
    run_pipeline(
        num_threads,
        num_weeks,
        fpl_team_id,
        clean,
        apply_transfers,
        wildcard_week,
        free_hit_week,
        triple_captain_week,
        bench_boost_week,
    )
