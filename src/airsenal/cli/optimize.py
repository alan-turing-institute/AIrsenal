"""Commands for optimizing transfers and squads."""

from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from airsenal.core.concurrency import set_multiprocessing_start_method
from airsenal.core.logging import get_logger
from airsenal.core.registry import lookup
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
from airsenal.optimization.protocols import SquadOptimizer, TransferOptimizer
from airsenal.optimization.run_squad import fill_initial_squad
from airsenal.optimization.run_transfers import run_optimization
from airsenal.optimization.squad_optimizers import (
    DEFAULT_SQUAD_OPTIMIZER,
    SQUAD_OPTIMIZERS,
    GeneticSquadOptimizer,
)
from airsenal.optimization.transfer_optimizers import (
    DEFAULT_TRANSFER_OPTIMIZER,
    TRANSFER_OPTIMIZERS,
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
    subs: Annotated[
        bool, typer.Option(help="Count substitutes' predicted points.")
    ] = True,
    num_iterations: Annotated[
        int, typer.Option(min=1, help="Wildcard/free-hit optimization iterations.")
    ] = 100,
    num_thread: Annotated[
        int, typer.Option(min=1, help="Worker processes to use.")
    ] = 4,
    transfer_optimizer: Annotated[
        str,
        typer.Option(
            help=f"Transfer search: {', '.join(sorted(TRANSFER_OPTIMIZERS))}."
        ),
    ] = DEFAULT_TRANSFER_OPTIMIZER,
    squad_optimizer: Annotated[
        str,
        typer.Option(
            help=(
                "Whole-squad optimizer used by a wildcard or free hit: "
                f"{', '.join(sorted(SQUAD_OPTIMIZERS))}."
            )
        ),
    ] = DEFAULT_SQUAD_OPTIMIZER,
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
    squad_optimizer: Annotated[
        str,
        typer.Option(help=f"Squad optimizer: {', '.join(sorted(SQUAD_OPTIMIZERS))}."),
    ] = DEFAULT_SQUAD_OPTIMIZER,
    subs: Annotated[
        bool, typer.Option(help="Count substitutes' predicted points.")
    ] = True,
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
        squad_optimizer=squad_optimizer,
        subs=subs,
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


def transfer_optimizer_named(
    name: str,
    *,
    num_thread: int | None = None,
    num_iterations: int | None = None,
    profile: bool = False,
) -> TransferOptimizer:
    """
    The named transfer search, configured from the flags that pre-date the table.

    `--num-thread`, `--num-iterations` and `--profile` are the tree search's own
    settings, so they only reach the tree search; any other optimizer named here
    starts from its own defaults. Finer configuration means constructing the
    component in Python, which is what the protocols are for.
    """
    if name != DEFAULT_TRANSFER_OPTIMIZER:
        return lookup(TRANSFER_OPTIMIZERS, name, "transfer optimizer")()
    config = TreeSearchConfig(profile=profile)
    if num_thread is not None:
        config = replace(config, num_thread=num_thread)
    if num_iterations is not None:
        config = replace(config, num_iterations=num_iterations)
    return TreeSearchOptimizer(config)


def squad_optimizer_named(
    name: str,
    *,
    num_generations: int | None = None,
    population_size: int | None = None,
) -> SquadOptimizer:
    """
    The named whole-squad optimizer, sized by the two flags that pre-date the table.

    `--num-generations` and `--population-size` describe a genetic algorithm, so
    like the tree search's flags they only reach the one component they are about.
    The rest of the GA's defaults live in `GeneticAlgorithmConfig` and nowhere
    else; they used to be restated in the CLI signature, here, in
    `fill_initial_squad` and in `make_new_squad`.
    """
    if name != DEFAULT_SQUAD_OPTIMIZER:
        return lookup(SQUAD_OPTIMIZERS, name, "squad optimizer")()
    config = GeneticAlgorithmConfig()
    if num_generations is not None:
        config = replace(config, generations=num_generations)
    if population_size is not None:
        config = replace(config, population_size=population_size)
    return GeneticSquadOptimizer(config)


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
    num_thread: int,
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
        chip_gameweeks=chips.as_dict(),
        num_free_transfers=num_free_transfers,
        constraints=TransferConstraints(
            max_total_hit=max_hit,
            allow_unused_transfers=allow_unused,
            max_opt_transfers=max_transfers,
        ),
        optimizer=transfer_optimizer_named(
            transfer_optimizer,
            num_thread=num_thread,
            num_iterations=num_iterations,
            profile=profile,
        ),
        squad_optimizer=squad_optimizer_named(squad_optimizer),
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

    fill_initial_squad(
        tag=tag,
        gameweeks=gameweeks,
        season=season,
        fpl_team_id=fpl_team_id,
        optimizer=squad_optimizer_named(
            squad_optimizer,
            num_generations=num_generations,
            population_size=population_size,
        ),
        scoring=SquadScoringConfig(
            sub_weights=SubWeights() if subs else SubWeights.none(),
            budget=budget,
        ),
        remove_zero=remove_zero,
        is_replay=is_replay,
    )
