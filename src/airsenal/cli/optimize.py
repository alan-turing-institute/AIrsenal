"""Commands for optimizing transfers and squads."""

from pathlib import Path
from typing import Annotated

import typer

from airsenal.cli import options
from airsenal.cli._components import (
    chip_gameweeks,
    squad_scoring,
    transfer_constraints,
)
from airsenal.core.logging import get_logger
from airsenal.db.queries.tags import get_latest_prediction_tag
from airsenal.game.season import CURRENT_SEASON
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
from airsenal.pipeline import AIrsenalPipeline, PipelineSettings
from airsenal.pipeline.settings import DEFAULT_N_GAMEWEEKS
from airsenal.remote.fpl_api import require_fpl_team_id

logger = get_logger(__name__)

app = typer.Typer(
    no_args_is_help=True, help="Optimize transfers or full squads for your FPL team."
)

Budget = Annotated[
    int,
    typer.Option(
        min=0, help="Budget in 0.1 million units.", rich_help_panel=options.OPTIMISATION
    ),
]

ZeroPointsPlayers = Annotated[
    bool,
    typer.Option(
        help="Consider players predicted to score nothing.",
        rich_help_panel=options.OPTIMISATION,
    ),
]

Profile = Annotated[
    bool,
    typer.Option(
        help="Profile the search's execution time.", rich_help_panel=options.OUTPUT
    ),
]

SavePlans = Annotated[
    Path | None,
    typer.Option(
        help="Directory to write every plan considered to, as JSON.",
        rich_help_panel=options.OUTPUT,
    ),
]

# An internal persistence mode, not a user concept: `replay` sets it, and a
# person running `optimize` has no reason to.
IsReplay = Annotated[
    bool,
    typer.Option(help="Store suggestions as replay transactions.", hidden=True),
]


@app.command()
def transfers(
    n_gameweeks: options.OptionalNGameweeks = None,
    gameweek_start: options.GameweekStart = None,
    gameweek_end: options.GameweekEnd = None,
    tag: options.Tag = None,
    season: options.Season = CURRENT_SEASON,
    fpl_team_id: options.FplTeamId = None,
    # --- chips ---
    wildcard_gameweek: options.WildcardGameweek = -1,
    free_hit_gameweek: options.FreeHitGameweek = -1,
    triple_captain_gameweek: options.TripleCaptainGameweek = -1,
    bench_boost_gameweek: options.BenchBoostGameweek = -1,
    # --- optimisation ---
    transfer_optimizer: options.TransferOptimizer = DEFAULT_TRANSFER_OPTIMIZER,
    squad_optimizer: options.SquadOptimizer = DEFAULT_SQUAD_OPTIMIZER,
    num_free_transfers: options.NumFreeTransfers = None,
    max_hit: options.MaxHit = DEFAULT_MAX_TOTAL_HIT,
    max_transfers: options.MaxTransfers = DEFAULT_MAX_OPT_TRANSFERS,
    allow_unused: options.AllowUnused = False,
    subs: options.Subs = True,
    num_iterations: options.NumIterations = None,
    num_thread: options.NumThread = None,
    # --- output ---
    profile: Profile = False,
    save_plans: SavePlans = None,
    is_replay: IsReplay = False,
) -> None:
    """Optimize a transfer plan."""
    _check_gameweek_args(gameweek_start, gameweek_end)
    _optimize(
        AIrsenalPipeline(
            transfer_optimizer=build_transfer_optimizer(
                transfer_optimizer,
                num_thread=num_thread,
                num_iterations=num_iterations,
                profile=profile,
            ),
            squad_optimizer=build_squad_optimizer(squad_optimizer),
            constraints=transfer_constraints(max_hit, allow_unused, max_transfers),
            scoring=squad_scoring(subs),
            settings=PipelineSettings(
                fpl_team_id=fpl_team_id,
                season=season,
                n_gameweeks=n_gameweeks or DEFAULT_N_GAMEWEEKS,
                gameweek_start=gameweek_start,
                gameweek_end=gameweek_end,
                chips=chip_gameweeks(
                    wildcard_gameweek,
                    free_hit_gameweek,
                    triple_captain_gameweek,
                    bench_boost_gameweek,
                ),
                num_free_transfers=num_free_transfers,
                save_plans=save_plans,
                refresh_database=False,
            ),
        ),
        tag,
        is_replay,
    )


@app.command()
def squad(
    n_gameweeks: options.NGameweeks = DEFAULT_N_GAMEWEEKS,
    gameweek_start: options.GameweekStart = None,
    season: options.OptionalSeason = None,
    fpl_team_id: options.FplTeamId = None,
    # --- optimisation ---
    squad_optimizer: options.SquadOptimizer = DEFAULT_SQUAD_OPTIMIZER,
    budget: Budget = 1000,
    num_generations: options.NumGenerations = None,
    population_size: options.PopulationSize = None,
    subs: options.Subs = True,
    zero_points_players: ZeroPointsPlayers = False,
    # --- output ---
    is_replay: IsReplay = False,
) -> None:
    """Optimize an initial squad."""
    season = season or CURRENT_SEASON
    if gameweek_start is None and season != CURRENT_SEASON:
        # a past season has no next gameweek to start from, so start at the beginning
        gameweek_start = 1
    _optimize(
        AIrsenalPipeline(
            squad_optimizer=build_squad_optimizer(
                squad_optimizer,
                num_generations=num_generations,
                population_size=population_size,
            ),
            scoring=squad_scoring(subs, budget),
            settings=PipelineSettings(
                fpl_team_id=fpl_team_id,
                season=season,
                n_gameweeks=n_gameweeks,
                gameweek_start=gameweek_start,
                # this command exists to build from scratch, so it does not ask
                # the API whether the entry has started
                new_squad=True,
                remove_zero_points_players=not zero_points_players,
                refresh_database=False,
            ),
        ),
        None,
        is_replay,
    )


def _check_gameweek_args(gameweek_start: int | None, gameweek_end: int | None) -> None:
    """
    A window is given as a length or as both ends, never as one end.

    `get_gameweeks_array` already rejects a length alongside either end, so that
    check is not repeated here.
    """
    if (gameweek_start is None) != (gameweek_end is None):
        msg = "Need to specify both --gameweek-start and --gameweek-end"
        raise typer.BadParameter(msg)


def _optimize(pipeline: AIrsenalPipeline, tag: str | None, is_replay: bool) -> None:
    """Resolve the window and the tag, then hand both to the pipeline."""
    season = pipeline.settings.season
    fpl_team_id = require_fpl_team_id(pipeline.settings.fpl_team_id)
    gameweeks = pipeline.gameweeks()
    pipeline.optimize(
        gameweeks,
        tag or get_latest_prediction_tag(season=season),
        fpl_team_id,
        is_replay=is_replay,
    )
