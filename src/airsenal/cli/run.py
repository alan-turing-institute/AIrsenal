"""Commands for running the full AIrsenal pipeline."""

import multiprocessing
from typing import Annotated

import typer

from airsenal.cli.options import parse_options
from airsenal.optimization.config import ChipWeeks
from airsenal.optimization.moves import TransferConstraints
from airsenal.optimization.transfer_optimizers import (
    TRANSFER_OPTIMIZERS,
    TreeSearchConfig,
)
from airsenal.pipeline import AIrsenalPipeline, DatabaseSettings, PipelineSettings
from airsenal.prediction.registry import (
    DEFAULT_PLAYER_MODEL,
    DEFAULT_TEAM_MODEL,
    PLAYER_MODELS,
    TEAM_MODELS,
)


def run(
    num_thread: Annotated[
        int | None, typer.Option(help="Number of threads to use for the pipeline run.")
    ] = None,
    n_gameweeks: Annotated[
        int,
        typer.Option(
            "--weeks-ahead", help="Number of gameweeks to include in the pipeline run."
        ),
    ] = 3,
    fpl_team_id: Annotated[
        int | None, typer.Option(help="FPL team ID for the pipeline run.")
    ] = None,
    clean: Annotated[
        bool, typer.Option(help="Delete and recreate the AIrsenal database.")
    ] = False,
    apply_transfers: Annotated[
        bool,
        typer.Option(
            help="Apply suggested transfers and set the lineup through the API."
        ),
    ] = False,
    wildcard_week: Annotated[
        int, typer.Option(help="Wildcard week; use 0 to consider any gameweek.")
    ] = -1,
    free_hit_week: Annotated[
        int, typer.Option(help="Free hit week; use 0 to consider any gameweek.")
    ] = -1,
    triple_captain_week: Annotated[
        int, typer.Option(help="Triple captain week; use 0 to consider any gameweek.")
    ] = -1,
    bench_boost_week: Annotated[
        int, typer.Option(help="Bench boost week; use 0 to consider any gameweek.")
    ] = -1,
    n_previous: Annotated[
        int,
        typer.Option(
            help="Number of previous seasons to include when creating the database."
        ),
    ] = 3,
    no_current_season: Annotated[
        bool,
        typer.Option(help="Exclude the current season when creating the database."),
    ] = False,
    team_model: Annotated[
        str, typer.Option(help=f"Team model: {', '.join(TEAM_MODELS.names())}.")
    ] = DEFAULT_TEAM_MODEL,
    epsilon: Annotated[
        float | None,
        typer.Option(
            help=(
                "Exponential time-weighting downweight factor. "
                "Defaults to the team model's own value."
            )
        ),
    ] = None,
    max_transfers: Annotated[
        int,
        typer.Option(min=0, max=5, help="Maximum transfers to consider per gameweek."),
    ] = 2,
    max_hit: Annotated[
        int,
        typer.Option(min=0, help="Maximum points to spend on additional transfers."),
    ] = 8,
    allow_unused: Annotated[
        bool, typer.Option(help="Include strategies that waste free transfers.")
    ] = False,
    save_absences: Annotated[
        bool, typer.Option(help="Save expected absences to a CSV file.")
    ] = False,
    player_model: Annotated[
        str,
        typer.Option(help=f"Player model: {', '.join(PLAYER_MODELS.names())}."),
    ] = DEFAULT_PLAYER_MODEL,
    set_player: Annotated[
        list[str] | None,
        typer.Option("--set-player", help="Player model option as key=value."),
    ] = None,
    set_team: Annotated[
        list[str] | None,
        typer.Option("--set-team", help="Team model option as key=value."),
    ] = None,
) -> None:
    """Run the full AIrsenal pipeline."""
    # the pipeline has always used every core unless told otherwise
    threads = num_thread or multiprocessing.cpu_count()
    AIrsenalPipeline.from_names(
        team_model=team_model,
        player_model=player_model,
        epsilon=epsilon,
        team_options=parse_options(set_team),
        player_options=parse_options(set_player),
        transfer_optimizer=TRANSFER_OPTIMIZERS.create(
            "tree_search",
            TreeSearchConfig(num_thread=threads),
        ),
        constraints=TransferConstraints(
            max_total_hit=max_hit,
            allow_unused_transfers=allow_unused,
            max_opt_transfers=max_transfers,
        ),
        settings=PipelineSettings(
            fpl_team_id=fpl_team_id,
            n_gameweeks=n_gameweeks,
            chips=ChipWeeks(
                wildcard=wildcard_week,
                free_hit=free_hit_week,
                triple_captain=triple_captain_week,
                bench_boost=bench_boost_week,
            ),
            database=DatabaseSettings(
                clean=clean,
                n_previous=n_previous,
                include_current_season=not no_current_season,
            ),
            apply_transfers=apply_transfers,
            save_absences=save_absences,
        ),
    ).run()
