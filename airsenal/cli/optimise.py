#!/usr/bin/env python3
import click

from airsenal.cli.utils import NotRequiredIf
from airsenal.framework.utils import CURRENT_SEASON
from airsenal.scripts.fill_transfersuggestion_table import optimise_transfers
from airsenal.scripts.squad_builder import optimise_squad


@click.command()
@click.option(
    "--transfers", is_flag=True, help="If set, run optimization for transfers."
)
@click.option(
    "--squad",
    is_flag=True,
    help="If set, run optimization for entire squad."
    " This should be used before the season starts or in cases like wildcards.",
)
@click.option(
    "--num-gameweeks",
    default=3,
    type=int,
    help="Number of weeks to run optimisation for (default 3).",
)
@click.option(
    "--gameweek-start", type=int, help="First gameweek to start looking from."
)
@click.option(
    "--gameweek-end",
    type=int,
    help="Last gameweek to look at. If not given, 3 weeks are used.",
    cls=NotRequiredIf,
    not_required_if="squad",
)
@click.option("--season", default=CURRENT_SEASON, help="Season, in format '1819'.")
@click.option(
    "--tag",
    type=str,
    help="Specify a string identifying prediction set.",
    cls=NotRequiredIf,
    not_required_if="squad",
)
@click.option(
    "--wildcard-week",
    default=-1,
    type=int,
    help="Play wildcard in the specified week. Choose 0 for 'any week'.",
    cls=NotRequiredIf,
    not_required_if="squad",
)
@click.option(
    "--free-hit-week",
    default=-1,
    type=int,
    help="Play free hit in the specified week. Choose 0 for 'any week'.",
    cls=NotRequiredIf,
    not_required_if="squad",
)
@click.option(
    "--triple-captain-week",
    default=-1,
    type=int,
    help="Play triple captain in the specified week. Choose 0 for 'any week'.",
    cls=NotRequiredIf,
    not_required_if="squad",
)
@click.option(
    "--bench-boost-week",
    default=-1,
    type=int,
    help="Play `bench boost` in the specified week. Choose 0 for 'any week'.",
    cls=NotRequiredIf,
    not_required_if="squad",
)
@click.option(
    "--num-free-transfers",
    type=int,
    help="Number of free transfers available.",
    cls=NotRequiredIf,
    not_required_if="squad",
)
@click.option(
    "--max-hit",
    default=8,
    type=int,
    help="Maximum number of points to spend on additional transfers.",
    cls=NotRequiredIf,
    not_required_if="squad",
)
@click.option(
    "--allow-unused",
    is_flag=True,
    help="If set, includes strategies that waste free transfers.",
    cls=NotRequiredIf,
    not_required_if="squad",
)
@click.option(
    "--num-iterations",
    default=100,
    type=int,
    help="If `--transfers`: Number of iterations to use for "
    + "Wildcard/Free Hit optimization. If `--squad`: Number of iterations for "
    + "normal algorithms.",
)
@click.option(
    "--num-threads",
    default=4,
    type=int,
    help="Number of threads to use for optimization (default 4).",
    cls=NotRequiredIf,
    not_required_if="squad",
)
@click.option(
    "--profile",
    is_flag=True,
    help="For developers: Profile strategy execution time.",
    cls=NotRequiredIf,
    not_required_if="squad",
)
@click.option(
    "--budget",
    default=1000,
    type=int,
    help="Squad Budget (in 0.1 millions).",
    cls=NotRequiredIf,
    not_required_if="transfers",
)
@click.option(
    "--algorithm",
    default="genetic",
    help="Optimization algorithm to use ['normal'/'genetic']",
    cls=NotRequiredIf,
    not_required_if="transfers",
)
@click.option(
    "--num-generations",
    default=100,
    type=int,
    help="Number of generations used in genetic algorithms.",
    cls=NotRequiredIf,
    not_required_if="transfers",
)
@click.option(
    "--population-size",
    default=100,
    type=int,
    help="Number of candidate solutions per generation (genetic algorithms only).",
    cls=NotRequiredIf,
    not_required_if="transfers",
)
@click.option(
    "--no-subs",
    is_flag=True,
    help="If set, don't include players with zero predicted points.",
    cls=NotRequiredIf,
    not_required_if="transfers",
)
@click.option(
    "--include-zero",
    is_flag=True,
    help="If set, include players with zero predicted points (genetic only).",
    cls=NotRequiredIf,
    not_required_if="transfers",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Print details on optimisation progress.",
    cls=NotRequiredIf,
    not_required_if="transfers",
)
@click.option("--fpl-team-id", type=int, required=False, help="FPL Team ID to use.")
def optimise(
    transfers: bool,
    squad: bool,
    num_gameweeks: int,
    gameweek_start: int,
    gameweek_end: int,
    season: int,
    tag: int,
    wildcard_week: int,
    free_hit_week: int,
    triple_captain_week: int,
    bench_boost_week: int,
    num_free_transfers: int,
    max_hit: int,
    allow_unused: bool,
    num_iterations: int,
    num_threads: int,
    profile: bool,
    budget: int,
    algorithm: str,
    num_generations: int,
    population_size: int,
    no_subs: bool,
    include_zero: bool,
    verbose: bool,
    fpl_team_id: int,
):
    """
    Run optimization to suggest teams. This can be run in two modes,
    `--transfers` and `--squad`. `--transfers` optimises the transfers for the
    given weeks and `--squad` optimises the entire squad. This is run in
    preseason or in cases where the entire squad can be overhauled, eg.
    wildcard usage.
    """
    if transfers:
        if num_gameweeks and (gameweek_start or gameweek_end):
            raise RuntimeError(
                "Please only specify num_gameweeks OR gameweek_start/end"
            )
        elif (gameweek_start and not gameweek_end) or (
            gameweek_end and not gameweek_start
        ):
            raise RuntimeError("Need to specify both gameweek_start and gameweek_end")
        if num_free_transfers and num_free_transfers not in range(1, 3):
            raise RuntimeError("Number of free transfers must be 1 or 2")
        optimise_transfers(
            num_gameweeks,
            gameweek_start,
            gameweek_end,
            season,
            tag,
            wildcard_week,
            free_hit_week,
            triple_captain_week,
            bench_boost_week,
            num_free_transfers,
            max_hit,
            allow_unused,
            num_iterations,
            num_threads,
            profile,
            fpl_team_id,
        )
    elif squad:
        optimise_squad(
            season,
            gameweek_start,
            num_gameweeks,
            budget,
            algorithm,
            num_iterations,
            num_generations,
            population_size,
            no_subs,
            include_zero,
            verbose,
            fpl_team_id,
        )
