"""Commands for optimizing transfers and squads."""

from pathlib import Path

import typer

from airsenal.cli import options
from airsenal.core.logging import get_logger
from airsenal.db.queries.tags import get_latest_prediction_tag
from airsenal.game.season import CURRENT_SEASON
from airsenal.optimization.moves import ChipWeeks
from airsenal.optimization.protocols import (
    DEFAULT_MAX_OPT_TRANSFERS,
    DEFAULT_MAX_TOTAL_HIT,
    TransferConstraints,
)
from airsenal.optimization.squad_optimizers import (
    DEFAULT_SQUAD_OPTIMIZER,
    build_squad_optimizer,
)
from airsenal.optimization.squad_score import SquadScoringConfig
from airsenal.optimization.transfer_optimizers import (
    DEFAULT_TRANSFER_OPTIMIZER,
    build_transfer_optimizer,
)
from airsenal.pipeline import AIrsenalPipeline, PipelineSettings
from airsenal.pipeline.settings import DEFAULT_N_GAMEWEEKS
from airsenal.remote.fpl_api import require_fpl_team_id
from airsenal.squad.squad import SubWeights

logger = get_logger(__name__)

app = typer.Typer(
    no_args_is_help=True, help="Optimize transfers or full squads for your FPL team."
)


@app.command()
def transfers(
    n_gameweeks: options.OptionalWeeksAhead = None,
    gameweek_start: options.GameweekStart = None,
    gameweek_end: options.GameweekEnd = None,
    tag: options.Tag = None,
    season: options.Season = CURRENT_SEASON,
    fpl_team_id: options.FplTeamId = None,
    # --- chips ---
    wildcard_week: options.WildcardWeek = -1,
    free_hit_week: options.FreeHitWeek = -1,
    triple_captain_week: options.TripleCaptainWeek = -1,
    bench_boost_week: options.BenchBoostWeek = -1,
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
    profile: options.Profile = False,
    save_plans: options.SavePlans = None,
    is_replay: options.IsReplay = False,
) -> None:
    """Optimize a transfer plan."""
    _run_transfer_optimization(
        n_gameweeks=n_gameweeks,
        gameweek_start=gameweek_start,
        gameweek_end=gameweek_end,
        tag=tag,
        chips=ChipWeeks(
            wildcard=wildcard_week,
            free_hit=free_hit_week,
            triple_captain=triple_captain_week,
            bench_boost=bench_boost_week,
        ),
        num_free_transfers=num_free_transfers,
        max_hit=max_hit,
        allow_unused=allow_unused,
        max_transfers=max_transfers,
        subs=subs,
        num_iterations=num_iterations,
        num_thread=num_thread,
        transfer_optimizer=transfer_optimizer,
        squad_optimizer=squad_optimizer,
        season=season,
        profile=profile,
        fpl_team_id=fpl_team_id,
        is_replay=is_replay,
        save_plans=save_plans,
    )


@app.command()
def squad(
    n_gameweeks: options.SquadWeeksAhead = DEFAULT_N_GAMEWEEKS,
    gameweek_start: options.GameweekStart = None,
    season: options.OptionalSeason = None,
    fpl_team_id: options.FplTeamId = None,
    # --- optimisation ---
    squad_optimizer: options.SquadOptimizer = DEFAULT_SQUAD_OPTIMIZER,
    budget: options.Budget = 1000,
    num_generations: options.NumGenerations = None,
    population_size: options.PopulationSize = None,
    subs: options.Subs = True,
    zero_points_players: options.ZeroPointsPlayers = False,
    # --- output ---
    is_replay: options.IsReplay = False,
) -> None:
    """Optimize an initial squad."""
    _run_squad_optimization(
        budget=budget,
        season=season,
        gameweek_start=gameweek_start,
        n_gameweeks=n_gameweeks,
        num_generations=num_generations,
        population_size=population_size,
        squad_optimizer=squad_optimizer,
        subs=subs,
        zero_points_players=zero_points_players,
        fpl_team_id=fpl_team_id,
        is_replay=is_replay,
    )


# --------------------------- turning flags into a run -----------------------


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


def _run_transfer_optimization(
    *,
    n_gameweeks: int | None,
    gameweek_start: int | None,
    gameweek_end: int | None,
    tag: str | None,
    chips: ChipWeeks,
    num_free_transfers: int | None,
    max_hit: int,
    allow_unused: bool,
    max_transfers: int,
    subs: bool,
    num_iterations: int | None,
    num_thread: int | None,
    transfer_optimizer: str,
    squad_optimizer: str,
    season: str,
    profile: bool,
    fpl_team_id: int | None,
    is_replay: bool,
    save_plans: Path | None = None,
) -> None:
    """Run transfer optimization for a gameweek range."""
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
            constraints=TransferConstraints(
                max_total_hit=max_hit,
                allow_unused_transfers=allow_unused,
                max_opt_transfers=max_transfers,
            ),
            scoring=SquadScoringConfig(
                sub_weights=SubWeights() if subs else SubWeights.none()
            ),
            settings=PipelineSettings(
                fpl_team_id=fpl_team_id,
                season=season,
                n_gameweeks=n_gameweeks or DEFAULT_N_GAMEWEEKS,
                gameweek_start=gameweek_start,
                gameweek_end=gameweek_end,
                chips=chips,
                num_free_transfers=num_free_transfers,
                save_plans=save_plans,
                refresh_database=False,
            ),
        ),
        tag,
        is_replay,
    )


def _run_squad_optimization(
    *,
    budget: int,
    season: str | None,
    gameweek_start: int | None,
    n_gameweeks: int,
    num_generations: int | None,
    population_size: int | None,
    squad_optimizer: str,
    subs: bool,
    zero_points_players: bool,
    fpl_team_id: int | None,
    is_replay: bool,
) -> None:
    """Generate an initial squad using prediction data."""
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
            scoring=SquadScoringConfig(
                sub_weights=SubWeights() if subs else SubWeights.none(),
                budget=budget,
            ),
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
