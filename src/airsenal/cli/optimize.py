"""Commands for optimizing transfers and squads."""

from dataclasses import fields
from pathlib import Path
from typing import Annotated

import typer

from airsenal.cli.options import parse_options
from airsenal.core.season import CURRENT_SEASON
from airsenal.optimization.config import GeneticAlgorithmConfig
from airsenal.optimization.run_squad import run_squad_optimization
from airsenal.optimization.run_transfers import run_transfer_optimization

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
        bool, typer.Option(help="Allow strategies that waste free transfers.")
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
        bool, typer.Option(help="Profile strategy execution time.")
    ] = False,
    fpl_team_id: Annotated[int | None, typer.Option(help="FPL team ID.")] = None,
    is_replay: Annotated[
        bool, typer.Option(help="Store suggestions as replay transactions.")
    ] = False,
    save_strategies: Annotated[
        Path | None,
        typer.Option(help="Directory to write every strategy considered to, as JSON."),
    ] = None,
) -> None:
    """Optimize a transfer strategy."""
    run_transfer_optimization(
        n_gameweeks,
        gameweek_start,
        gameweek_end,
        tag,
        wildcard_week,
        free_hit_week,
        triple_captain_week,
        bench_boost_week,
        num_free_transfers,
        max_hit,
        allow_unused,
        max_transfers,
        num_iterations,
        num_thread,
        season,
        profile,
        fpl_team_id,
        is_replay,
        save_strategies,
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
    set_ga: Annotated[
        list[str] | None,
        typer.Option(
            "--set-ga",
            help=(
                "Genetic algorithm option as key=value. Repeatable. Available: "
                + ", ".join(f.name for f in fields(GeneticAlgorithmConfig))
            ),
        ),
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
    run_squad_optimization(
        budget,
        season,
        gameweek_start,
        n_gameweeks,
        num_generations,
        population_size,
        parse_options(set_ga),
        no_subs,
        include_zero,
        fpl_team_id,
        is_replay,
    )
