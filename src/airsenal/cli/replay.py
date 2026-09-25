"""Commands for replaying historical seasons."""

from pathlib import Path
from typing import Annotated

import typer

from airsenal.cli import options
from airsenal.cli._components import (
    chip_gameweeks,
    squad_scoring,
    transfer_constraints,
)
from airsenal.optimization.protocols import (
    DEFAULT_MAX_OPT_TRANSFERS,
    DEFAULT_MAX_TOTAL_HIT,
)
from airsenal.optimization.squad_optimizers import (
    DEFAULT_SQUAD_OPTIMIZER,
    build_squad_optimizer,
)
from airsenal.optimization.transfer_optimizers import (
    DEFAULT_TRANSFER_OPTIMIZER,
    build_transfer_optimizer,
)
from airsenal.pipeline import (
    AIrsenalPipeline,
    PipelineSettings,
    ReplaySettings,
    run_replays,
)
from airsenal.pipeline.settings import DEFAULT_N_GAMEWEEKS
from airsenal.prediction.minutes_models import DEFAULT_MINUTES_MODEL
from airsenal.prediction.player_models import DEFAULT_PLAYER_MODEL
from airsenal.prediction.points_models import (
    DEFAULT_POINTS_MODEL,
    build_points_model,
)
from airsenal.prediction.team_models import DEFAULT_TEAM_MODEL

OutputDir = Annotated[
    Path | None,
    typer.Option(
        help="Directory to write results to. Defaults to the current directory.",
        rich_help_panel=options.OUTPUT,
    ),
]

TagPrefix = Annotated[
    str,
    typer.Option(
        help="Prefix for the result tag and filename. Defaults to a timestamped one.",
        rich_help_panel=options.OUTPUT,
    ),
]


def replay(
    season: options.Season,
    gameweek_start: Annotated[
        int, typer.Option(min=1, help="First gameweek to replay.")
    ] = 1,
    gameweek_end: Annotated[
        int | None, typer.Option(help="Last gameweek to replay.")
    ] = None,
    n_gameweeks: options.NGameweeks = DEFAULT_N_GAMEWEEKS,
    fpl_team_id: options.FplTeamId = None,
    resume: Annotated[
        bool, typer.Option(help="Resume an existing replay team.")
    ] = False,
    loop: Annotated[
        int, typer.Option(help="Replay count; -1 repeats indefinitely.")
    ] = 1,
    num_thread: options.NumThread = None,
    num_iterations: options.NumIterations = None,
    num_generations: options.NumGenerations = None,
    population_size: options.PopulationSize = None,
    num_free_transfers: options.NumFreeTransfers = None,
    player_model: options.PlayerModel = DEFAULT_PLAYER_MODEL,
    team_model: options.TeamModel = DEFAULT_TEAM_MODEL,
    minutes_model: options.MinutesModel = DEFAULT_MINUTES_MODEL,
    points_model: options.PointsModel = DEFAULT_POINTS_MODEL,
    epsilon: options.Epsilon = None,
    transfer_optimizer: options.TransferOptimizer = DEFAULT_TRANSFER_OPTIMIZER,
    squad_optimizer: options.SquadOptimizer = DEFAULT_SQUAD_OPTIMIZER,
    max_transfers: options.MaxTransfers = DEFAULT_MAX_OPT_TRANSFERS,
    max_hit: options.MaxHit = DEFAULT_MAX_TOTAL_HIT,
    allow_unused: options.AllowUnused = False,
    wildcard_gameweek: options.WildcardGameweek = -1,
    free_hit_gameweek: options.FreeHitGameweek = -1,
    triple_captain_gameweek: options.TripleCaptainGameweek = -1,
    bench_boost_gameweek: options.BenchBoostGameweek = -1,
    chip_heuristic: options.ChipHeuristic = False,
    subs: options.Subs = True,
    output_dir: OutputDir = None,
    tag_prefix: TagPrefix = "",
) -> None:
    """Replay a historical FPL season."""
    run_replays(
        AIrsenalPipeline(
            points_model=build_points_model(
                points_model,
                team_model=team_model,
                player_model=player_model,
                minutes_model=minutes_model,
                epsilon=epsilon,
            ),
            transfer_optimizer=build_transfer_optimizer(
                transfer_optimizer,
                num_thread=num_thread,
                num_iterations=num_iterations,
            ),
            squad_optimizer=build_squad_optimizer(
                squad_optimizer,
                num_generations=num_generations,
                population_size=population_size,
            ),
            constraints=transfer_constraints(max_hit, allow_unused, max_transfers),
            scoring=squad_scoring(subs),
            settings=PipelineSettings(
                fpl_team_id=fpl_team_id,
                n_gameweeks=n_gameweeks,
                num_free_transfers=num_free_transfers,
                season=season,
                chips=chip_gameweeks(
                    wildcard_gameweek,
                    free_hit_gameweek,
                    triple_captain_gameweek,
                    bench_boost_gameweek,
                    heuristic=chip_heuristic,
                ),
                # replay never touches the real entry or the live API
                refresh_database=False,
                apply_transfers=False,
            ),
        ),
        ReplaySettings(
            gameweek_start=gameweek_start,
            gameweek_end=gameweek_end,
            tag_prefix=tag_prefix,
            loop=loop,
            resume=resume,
            output_dir=output_dir,
        ),
    )
