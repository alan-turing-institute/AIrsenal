"""Commands for optimizing transfers and squads."""

from pathlib import Path

import typer

from airsenal.cli import options
from airsenal.core.concurrency import set_multiprocessing_start_method
from airsenal.core.logging import get_logger
from airsenal.core.season import CURRENT_SEASON
from airsenal.db.queries.gameweeks import (
    get_gameweeks_array,
    get_max_gameweek,
    next_gameweek,
)
from airsenal.db.queries.tags import check_tag_valid, get_latest_prediction_tag
from airsenal.optimization.moves import ChipWeeks
from airsenal.optimization.protocols import (
    TransferConstraints,
)
from airsenal.optimization.run_squad import fill_initial_squad
from airsenal.optimization.run_transfers import run_optimization
from airsenal.optimization.squad_optimizers import (
    DEFAULT_SQUAD_OPTIMIZER,
    build_squad_optimizer,
)
from airsenal.optimization.squad_score import SquadScoringConfig, SubWeights
from airsenal.optimization.transfer_optimizers import (
    DEFAULT_TRANSFER_OPTIMIZER,
    build_transfer_optimizer,
)
from airsenal.remote.fpl_api import require_fpl_team_id

logger = get_logger(__name__)

app = typer.Typer(
    no_args_is_help=True, help="Optimize transfers or full squads for your FPL team."
)


@app.command()
def transfers(
    n_gameweeks: options.OptionalWeeksAhead = None,
    gameweek_start: options.GameweekStart = None,
    gameweek_end: options.GameweekEnd = None,
    season: options.Season = options.DEFAULT_SEASON,
    tag: options.Tag = None,
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
    max_hit: options.MaxHit = options.DEFAULT_MAX_HIT,
    max_transfers: options.MaxTransfers = options.DEFAULT_MAX_TRANSFERS,
    allow_unused: options.AllowUnused = False,
    subs: options.Subs = True,
    num_iterations: options.NumIterations = options.DEFAULT_NUM_ITERATIONS,
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
    n_gameweeks: options.SquadWeeksAhead = options.DEFAULT_N_GAMEWEEKS,
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


# --------------------------- turning flags into components ------------------


def _check_gameweek_args(gameweek_start: int | None, gameweek_end: int | None) -> None:
    """
    A window is given as a length or as both ends, never as one end.

    `get_gameweeks_array` already rejects a length alongside either end, so that
    check is not repeated here.
    """
    if (gameweek_start is None) != (gameweek_end is None):
        msg = "Need to specify both --gameweek-start and --gameweek-end"
        raise typer.BadParameter(msg)


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
    num_iterations: int,
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
    gameweeks = get_gameweeks_array(
        n_gameweeks=n_gameweeks,
        gameweek_start=gameweek_start,
        gameweek_end=gameweek_end,
        season=season,
    )
    tag = tag or get_latest_prediction_tag(season=season)

    if not check_tag_valid(tag, gameweeks, season=season):
        msg = (
            "The database has no predictions covering all the requested "
            "gameweeks. Run `airsenal predict` first, for the same gameweeks "
            "and season."
        )
        raise typer.BadParameter(msg)

    set_multiprocessing_start_method()

    run_optimization(
        gameweeks,
        tag,
        season=season,
        fpl_team_id=fpl_team_id,
        chips=chips,
        num_free_transfers=num_free_transfers,
        constraints=TransferConstraints(
            max_total_hit=max_hit,
            allow_unused_transfers=allow_unused,
            max_opt_transfers=max_transfers,
        ),
        optimizer=build_transfer_optimizer(
            transfer_optimizer,
            num_thread=num_thread,
            num_iterations=num_iterations,
            profile=profile,
        ),
        squad_optimizer=build_squad_optimizer(squad_optimizer),
        scoring=SquadScoringConfig(
            sub_weights=SubWeights() if subs else SubWeights.none()
        ),
        save_plans=save_plans,
        is_replay=is_replay,
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
    if gameweek_start:
        resolved_gameweek_start = gameweek_start
    elif season == CURRENT_SEASON:
        resolved_gameweek_start = next_gameweek()
    else:
        resolved_gameweek_start = 1
    gameweeks = list(
        range(
            resolved_gameweek_start,
            min(
                get_max_gameweek(season) + 1,
                resolved_gameweek_start + n_gameweeks,
            ),
        )
    )
    tag = get_latest_prediction_tag(season)
    if not check_tag_valid(tag, gameweeks, season=season):
        msg = (
            "The database has no predictions covering all the requested "
            "gameweeks. Run `airsenal predict` first, for the same gameweeks "
            "and season."
        )
        raise typer.BadParameter(msg)
    fpl_team_id = require_fpl_team_id(fpl_team_id)

    fill_initial_squad(
        tag=tag,
        gameweeks=gameweeks,
        season=season,
        fpl_team_id=fpl_team_id,
        optimizer=build_squad_optimizer(
            squad_optimizer,
            num_generations=num_generations,
            population_size=population_size,
        ),
        scoring=SquadScoringConfig(
            sub_weights=SubWeights() if subs else SubWeights.none(),
            budget=budget,
        ),
        remove_zero=not zero_points_players,
        is_replay=is_replay,
    )
