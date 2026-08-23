"""Commands for optimizing transfers and squads."""

from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from airsenal.core.concurrency import set_multiprocessing_start_method
from airsenal.core.logging import get_logger
from airsenal.core.season import CURRENT_SEASON
from airsenal.db.queries.gameweeks import (
    get_gameweeks_array,
    get_max_gameweek,
    next_gameweek,
)
from airsenal.db.queries.tags import check_tag_valid, get_latest_prediction_tag
from airsenal.fetch.fpl_api import require_fpl_team_id
from airsenal.optimization.config import (
    ChipWeeks,
    GeneticAlgorithmConfig,
    SquadScoringConfig,
    SubWeights,
)
from airsenal.optimization.moves import TransferConstraints
from airsenal.optimization.run_squad import fill_initial_squad
from airsenal.optimization.run_transfers import run_optimization
from airsenal.optimization.squad_optimizers import (
    GeneticSquadOptimizer,
    genetic_optimizer,
)
from airsenal.optimization.transfer_optimizers import (
    TreeSearchConfig,
    TreeSearchOptimizer,
)

logger = get_logger(__name__)

app = typer.Typer(
    no_args_is_help=True, help="Optimize transfers or full squads for your FPL team."
)


@app.command()
def transfers(
    n_gameweeks: Annotated[
        int | None,
        # The flag name is public, so it is pinned rather than derived from the
        # parameter, which was renamed for consistency with everything else.
        typer.Option("--weeks-ahead", help="Number of gameweeks to optimize."),
    ] = None,
    gameweek_start: Annotated[
        int | None, typer.Option(help="First gameweek to optimize.")
    ] = None,
    gameweek_end: Annotated[
        int | None, typer.Option(help="Last gameweek to optimize.")
    ] = None,
    tag: Annotated[
        str | None, typer.Option(help="Prediction tag; defaults to the latest.")
    ] = None,
    wildcard_week: Annotated[
        int, typer.Option(help="Wildcard week; use 0 for any week.")
    ] = -1,
    free_hit_week: Annotated[
        int, typer.Option(help="Free hit week; use 0 for any week.")
    ] = -1,
    triple_captain_week: Annotated[
        int, typer.Option(help="Triple captain week; use 0 for any week.")
    ] = -1,
    bench_boost_week: Annotated[
        int, typer.Option(help="Bench boost week; use 0 for any week.")
    ] = -1,
    num_free_transfers: Annotated[
        int | None, typer.Option(min=0, max=5, help="Free transfers available.")
    ] = None,
    max_hit: Annotated[
        int, typer.Option(min=0, help="Maximum points to spend on transfers.")
    ] = 8,
    allow_unused: Annotated[
        bool, typer.Option(help="Allow plans that waste free transfers.")
    ] = False,
    max_transfers: Annotated[
        int, typer.Option(min=0, help="Maximum transfers per gameweek.")
    ] = 2,
    num_iterations: Annotated[
        int, typer.Option(min=1, help="Wildcard/free-hit optimization iterations.")
    ] = 100,
    num_thread: Annotated[
        int, typer.Option(min=1, help="Worker processes to use.")
    ] = 4,
    season: Annotated[
        str, typer.Option(help="Season in the form 2526.")
    ] = CURRENT_SEASON,
    profile: Annotated[
        bool, typer.Option(help="Profile the search's execution time.")
    ] = False,
    fpl_team_id: Annotated[int | None, typer.Option(help="FPL team ID.")] = None,
    is_replay: Annotated[
        bool, typer.Option(help="Store suggestions as replay transactions.")
    ] = False,
    save_plans: Annotated[
        Path | None,
        typer.Option(help="Directory to write every plan considered to, as JSON."),
    ] = None,
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
        num_iterations=num_iterations,
        num_thread=num_thread,
        season=season,
        profile=profile,
        fpl_team_id=fpl_team_id,
        is_replay=is_replay,
        save_plans=save_plans,
    )


@app.command()
def squad(
    budget: Annotated[
        int, typer.Option(min=0, help="Budget in 0.1 million units.")
    ] = 1000,
    season: Annotated[str | None, typer.Option(help="Season in the form 2526.")] = None,
    gameweek_start: Annotated[
        int | None, typer.Option(help="Starting gameweek.")
    ] = None,
    n_gameweeks: Annotated[
        int,
        # --num-gameweeks is what this command has always been called; it keeps
        # working, but --weeks-ahead is what every other command uses.
        typer.Option(
            "--weeks-ahead",
            "--num-gameweeks",
            min=1,
            help="Number of gameweeks to optimize.",
        ),
    ] = 3,
    num_generations: Annotated[
        int | None,
        typer.Option(min=1, help="Genetic algorithm generations."),
    ] = None,
    population_size: Annotated[
        int | None,
        typer.Option(min=1, help="Candidate squads per generation."),
    ] = None,
    no_subs: Annotated[
        bool, typer.Option(help="Exclude substitute-point contributions.")
    ] = False,
    include_zero: Annotated[
        bool, typer.Option(help="Include zero-point players.")
    ] = False,
    fpl_team_id: Annotated[int | None, typer.Option(help="FPL team ID.")] = None,
    is_replay: Annotated[
        bool, typer.Option(help="Store suggestions as replay transactions.")
    ] = False,
) -> None:
    """Optimize an initial squad."""
    _run_squad_optimization(
        budget=budget,
        season=season,
        gameweek_start=gameweek_start,
        n_gameweeks=n_gameweeks,
        num_generations=num_generations,
        population_size=population_size,
        no_subs=no_subs,
        include_zero=include_zero,
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
    num_iterations: int,
    num_thread: int,
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
        chip_gameweeks=chips.as_dict(),
        num_free_transfers=num_free_transfers,
        constraints=TransferConstraints(
            max_total_hit=max_hit,
            allow_unused_transfers=allow_unused,
            max_opt_transfers=max_transfers,
        ),
        optimizer=TreeSearchOptimizer(
            TreeSearchConfig(
                num_thread=num_thread,
                num_iterations=num_iterations,
                profile=profile,
            )
        ),
        # the from-scratch fallback sizes its search from the same effort knob
        squad_optimizer=genetic_optimizer(num_iterations),
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
    no_subs: bool,
    include_zero: bool,
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
    remove_zero = not include_zero
    fpl_team_id = require_fpl_team_id(fpl_team_id)

    # --population-size and --generations are the two knobs people reach for; the
    # rest of the GA's defaults live in GeneticAlgorithmConfig only. They used to be
    # restated in the CLI signature, here, in fill_initial_squad and in
    # make_new_squad.
    ga_config = GeneticAlgorithmConfig()
    if num_generations is not None:
        ga_config = replace(ga_config, generations=num_generations)
    if population_size is not None:
        ga_config = replace(ga_config, population_size=population_size)

    fill_initial_squad(
        tag=tag,
        gameweeks=gameweeks,
        season=season,
        fpl_team_id=fpl_team_id,
        optimizer=GeneticSquadOptimizer(ga_config),
        scoring=SquadScoringConfig(
            sub_weights=SubWeights.none() if no_subs else SubWeights(),
            budget=budget,
        ),
        remove_zero=remove_zero,
        is_replay=is_replay,
    )
