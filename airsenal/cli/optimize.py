"""Commands for optimizing transfers and squads."""

import typer

from airsenal.framework.season import CURRENT_SEASON
from airsenal.scripts.fill_transfersuggestion_table import run_transfer_optimization
from airsenal.scripts.squad_builder import run_squad_optimization

app = typer.Typer(no_args_is_help=True)


@app.command()
def transfers(
    weeks_ahead: int | None = typer.Option(
        None, help="Number of gameweeks to optimize."
    ),
    gameweek_start: int | None = typer.Option(None, help="First gameweek to optimize."),
    gameweek_end: int | None = typer.Option(None, help="Last gameweek to optimize."),
    tag: str | None = typer.Option(
        None, help="Prediction tag; defaults to the latest."
    ),
    wildcard_week: int = typer.Option(-1, help="Wildcard week; use 0 for any week."),
    free_hit_week: int = typer.Option(-1, help="Free hit week; use 0 for any week."),
    triple_captain_week: int = typer.Option(
        -1, help="Triple captain week; use 0 for any week."
    ),
    bench_boost_week: int = typer.Option(
        -1, help="Bench boost week; use 0 for any week."
    ),
    num_free_transfers: int | None = typer.Option(
        None, min=0, max=5, help="Free transfers available."
    ),
    max_hit: int = typer.Option(8, min=0, help="Maximum points to spend on transfers."),
    allow_unused: bool = typer.Option(
        False, help="Allow strategies that waste free transfers."
    ),
    max_transfers: int = typer.Option(2, min=0, help="Maximum transfers per gameweek."),
    num_iterations: int = typer.Option(
        100, min=1, help="Wildcard/free-hit optimization iterations."
    ),
    num_thread: int = typer.Option(4, min=1, help="Worker processes to use."),
    season: str = typer.Option(CURRENT_SEASON, help="Season in the form 2526."),
    profile: bool = typer.Option(False, help="Profile strategy execution time."),
    fpl_team_id: int | None = typer.Option(None, help="FPL team ID."),
    is_replay: bool = typer.Option(
        False, help="Store suggestions as replay transactions."
    ),
) -> None:
    """Optimize a transfer strategy."""
    run_transfer_optimization(
        weeks_ahead,
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
    )


@app.command()
def squad(
    budget: int = typer.Option(1000, min=0, help="Budget in 0.1 million units."),
    season: str | None = typer.Option(None, help="Season in the form 2526."),
    gameweek_start: int | None = typer.Option(None, help="Starting gameweek."),
    num_gameweeks: int = typer.Option(
        3, min=1, help="Number of gameweeks to optimize."
    ),
    num_generations: int = typer.Option(
        100, min=1, help="Genetic algorithm generations."
    ),
    population_size: int = typer.Option(
        100, min=1, help="Candidate squads per generation."
    ),
    crossover_prob: float = typer.Option(
        0.7, min=0, max=1, help="Crossover probability."
    ),
    mutation_prob: float = typer.Option(
        0.3, min=0, max=1, help="Mutation probability."
    ),
    crossover_indpb: float = typer.Option(
        0.5, min=0, max=1, help="Per-attribute crossover probability."
    ),
    mutation_indpb: float = typer.Option(
        0.1, min=0, max=1, help="Per-attribute mutation probability."
    ),
    tournament_size: int = typer.Option(3, min=1, help="Tournament selection size."),
    no_subs: bool = typer.Option(False, help="Exclude substitute-point contributions."),
    include_zero: bool = typer.Option(False, help="Include zero-point players."),
    fpl_team_id: int | None = typer.Option(None, help="FPL team ID."),
    is_replay: bool = typer.Option(
        False, help="Store suggestions as replay transactions."
    ),
) -> None:
    """Optimize an initial squad."""
    run_squad_optimization(
        budget,
        season,
        gameweek_start,
        num_gameweeks,
        num_generations,
        population_size,
        crossover_prob,
        mutation_prob,
        crossover_indpb,
        mutation_indpb,
        tournament_size,
        no_subs,
        include_zero,
        fpl_team_id,
        is_replay,
    )
