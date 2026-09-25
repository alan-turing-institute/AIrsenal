"""Commands for running the full AIrsenal pipeline."""

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
from airsenal.pipeline import AIrsenalPipeline, DatabaseSettings, PipelineSettings
from airsenal.pipeline.settings import (
    DEFAULT_N_GAMEWEEKS,
    DEFAULT_N_PREVIOUS,
    StaleDatabase,
)
from airsenal.prediction.minutes_models import DEFAULT_MINUTES_MODEL
from airsenal.prediction.player_models import DEFAULT_PLAYER_MODEL
from airsenal.prediction.points_models import (
    DEFAULT_POINTS_MODEL,
    build_points_model,
)
from airsenal.prediction.team_models import DEFAULT_TEAM_MODEL

RefreshDatabase = Annotated[
    bool,
    typer.Option(
        help=(
            "Fetch new data before predicting. Off, the run works from what the "
            "database already holds - re-optimise without re-fetching."
        ),
        rich_help_panel=options.DATABASE,
    ),
]

OnStale = Annotated[
    StaleDatabase,
    typer.Option(
        help=(
            "What to do if the database could not be brought up to date. "
            "'abort' is the one for an unattended run."
        ),
        rich_help_panel=options.DATABASE,
    ),
]

ApplyTransfers = Annotated[
    bool,
    typer.Option(
        help="Apply the suggested transfers and lineup through the FPL API.",
        rich_help_panel=options.OUTPUT,
    ),
]


def run(
    fpl_team_id: options.FplTeamId = None,
    n_gameweeks: options.NGameweeks = DEFAULT_N_GAMEWEEKS,
    gameweek_start: options.GameweekStart = None,
    # --- database ---
    clean: options.Clean = False,
    n_previous: options.NPrevious = DEFAULT_N_PREVIOUS,
    current_season: options.CurrentSeason = True,
    refresh_database: RefreshDatabase = True,
    on_stale: OnStale = StaleDatabase.ASK,
    # --- prediction ---
    player_model: options.PlayerModel = DEFAULT_PLAYER_MODEL,
    team_model: options.TeamModel = DEFAULT_TEAM_MODEL,
    minutes_model: options.MinutesModel = DEFAULT_MINUTES_MODEL,
    points_model: options.PointsModel = DEFAULT_POINTS_MODEL,
    epsilon: options.Epsilon = None,
    # --- optimisation ---
    transfer_optimizer: options.TransferOptimizer = DEFAULT_TRANSFER_OPTIMIZER,
    squad_optimizer: options.SquadOptimizer = DEFAULT_SQUAD_OPTIMIZER,
    num_thread: options.NumThread = None,
    max_transfers: options.MaxTransfers = DEFAULT_MAX_OPT_TRANSFERS,
    max_hit: options.MaxHit = DEFAULT_MAX_TOTAL_HIT,
    allow_unused: options.AllowUnused = False,
    subs: options.Subs = True,
    wildcard_gameweek: options.WildcardGameweek = -1,
    free_hit_gameweek: options.FreeHitGameweek = -1,
    triple_captain_gameweek: options.TripleCaptainGameweek = -1,
    bench_boost_gameweek: options.BenchBoostGameweek = -1,
    chip_heuristic: options.ChipHeuristic = False,
    # --- output ---
    apply_transfers: ApplyTransfers = False,
    yes: options.Yes = False,
) -> None:
    """Run the full AIrsenal pipeline."""
    AIrsenalPipeline(
        points_model=build_points_model(
            points_model,
            team_model=team_model,
            player_model=player_model,
            minutes_model=minutes_model,
            epsilon=epsilon,
        ),
        transfer_optimizer=build_transfer_optimizer(
            transfer_optimizer, num_thread=num_thread
        ),
        squad_optimizer=build_squad_optimizer(squad_optimizer),
        constraints=transfer_constraints(max_hit, allow_unused, max_transfers),
        scoring=squad_scoring(subs),
        settings=PipelineSettings(
            fpl_team_id=fpl_team_id,
            n_gameweeks=n_gameweeks,
            gameweek_start=gameweek_start,
            chips=chip_gameweeks(
                wildcard_gameweek,
                free_hit_gameweek,
                triple_captain_gameweek,
                bench_boost_gameweek,
                heuristic=chip_heuristic,
            ),
            database=DatabaseSettings(
                clean=clean,
                n_previous=n_previous,
                include_current_season=current_season,
            ),
            refresh_database=refresh_database,
            on_stale_database=on_stale,
            apply_transfers=apply_transfers,
            skip_confirmation=yes,
        ),
    ).run()
