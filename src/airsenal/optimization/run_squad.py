import sys

from rich.panel import Panel
from rich.text import Text

from airsenal.core.console import console, price_str, table
from airsenal.core.enums import Position
from airsenal.core.logging import get_logger
from airsenal.db.queries.gameweeks import get_max_gameweek, next_gameweek
from airsenal.db.queries.tags import get_latest_prediction_tag
from airsenal.domain.season import CURRENT_SEASON
from airsenal.fetch.fpl_api import get_fetcher
from airsenal.optimization.config import GeneticAlgorithmConfig, SubWeights
from airsenal.optimization.squad_ga import make_new_squad
from airsenal.optimization.utils import (
    DEFAULT_SUB_WEIGHTS,
    check_tag_valid,
    fill_initial_suggestion_table,
    fill_initial_transaction_table,
    get_discounted_squad_score,
)
from airsenal.reporting.squad_view import formation_table
from airsenal.squad.squad import Squad

logger = get_logger(__name__)

positions = list(Position.front_to_back())  # front-to-back


def fill_initial_squad(
    tag: str,
    gw_range: list[int],
    season: str,
    fpl_team_id: int,
    budget: int = 1000,
    remove_zero: bool = True,
    sub_weights: dict = DEFAULT_SUB_WEIGHTS,
    num_generations: int = 100,
    population_size: int = 100,
    crossover_prob: float = 0.7,
    mutation_prob: float = 0.3,
    crossover_indpb: float = 0.5,
    mutation_indpb: float = 0.1,
    tournament_size: int = 3,
    verbose: bool = True,
    is_replay: bool = False,  # for replaying seasons
    chip_gameweeks: dict[str, int] | None = None,
) -> Squad:
    with console.status("Optimising full squad..."):
        best_squad = make_new_squad(
            gw_range,
            tag,
            budget=budget,
            season=season,
            remove_zero=remove_zero,
            sub_weights=sub_weights,
            ga_config=GeneticAlgorithmConfig(
                population_size=population_size,
                generations=num_generations,
                crossover_prob=crossover_prob,
                mutation_prob=mutation_prob,
                crossover_indpb=crossover_indpb,
                mutation_indpb=mutation_indpb,
                tournament_size=tournament_size,
                verbose=verbose,
            ),
            verbose=verbose,
        )

    if best_squad is None:
        msg = (
            "best_squad is None: make_new_squad failed to generate a valid team or "
            "something went wrong with the squad expected points calculation."
        )
        raise RuntimeError(msg)

    gw_start = gw_range[0]
    optimised_score = get_discounted_squad_score(
        best_squad,
        gw_range,
        tag,
        gw_start,
        sub_weights=sub_weights,
    )

    chip_gameweeks = chip_gameweeks or {}

    summary = Text()
    summary.append(
        f"Gameweeks: {min(gw_range)}-{max(gw_range)}\n"
        if min(gw_range) != max(gw_range)
        else f"Gameweek: {min(gw_range)}\n",
        style="bold",
    )
    summary.append(f"Team ID: {fpl_team_id}\n")
    summary.append(f"Optimised Score: {optimised_score:.1f}pts\n", style="bold green")
    console.print(Panel(summary, title="Optimisation Result", expand=False))

    strategy_table = table(
        "Gameweek",
        "Transfers",
        "Chip",
        "Points Hit",
        "Predicted Score",
        title="Strategy",
    )
    for gw in gw_range:
        bench_boost = chip_gameweeks.get("bench_boost") == gw
        triple_captain = chip_gameweeks.get("triple_captain") == gw
        chip = (
            "bench_boost"
            if bench_boost
            else "triple_captain"
            if triple_captain
            else "-"
        )
        pred_pts = best_squad.get_expected_points(
            gw, tag, bench_boost=bench_boost, triple_captain=triple_captain
        )
        strategy_table.add_row(
            str(gw),
            str(len(best_squad.players)) if gw == gw_start else "0",
            chip,
            "0pts",
            f"{pred_pts:.1f}pts",
        )
    console.print(strategy_table)

    transfer_table = table(
        "Player In",
        "Pos",
        "Team",
        "Purchase Price",
        title="Transfers",
    )
    for player in sorted(
        best_squad.players, key=lambda player: positions.index(player.position)
    ):
        transfer_table.add_row(
            str(player),
            player.position,
            player.team,
            price_str(player.purchase_price),
        )
    console.print(transfer_table)

    console.print(
        formation_table(
            best_squad,
            tag,
            gw_start,
            bench_boost=chip_gameweeks.get("bench_boost") == gw_start,
            triple_captain=chip_gameweeks.get("triple_captain") == gw_start,
        )
    )

    fill_initial_suggestion_table(
        best_squad,
        fpl_team_id,
        tag,
        season=season,
        gameweek=gw_start,
    )
    if is_replay:
        # if simulating a previous season also add suggestions to transaction table
        # to imitate applying transfers
        fill_initial_transaction_table(
            best_squad,
            fpl_team_id,
            tag,
            season=season,
            gameweek=gw_start,
        )
    return best_squad


def run_squad_optimization(
    budget: int,
    season: str | None,
    gameweek_start: int | None,
    num_gameweeks: int,
    num_generations: int,
    population_size: int,
    crossover_prob: float,
    mutation_prob: float,
    crossover_indpb: float,
    mutation_indpb: float,
    tournament_size: int,
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
    gw_range = list(
        range(
            resolved_gameweek_start,
            min(
                get_max_gameweek(season) + 1,
                resolved_gameweek_start + num_gameweeks,
            ),
        )
    )
    tag = get_latest_prediction_tag(season)
    if not check_tag_valid(tag, gw_range, season=season):
        logger.error(
            "Database does not contain predictions for all the specified "
            "optimsation gameweeks.\nPlease run 'airsenal_run_prediction' first "
            "with the same input gameweeks and season you specified here."
        )
        sys.exit(1)
    remove_zero = not include_zero
    fpl_team_id = fpl_team_id or get_fetcher().FPL_TEAM_ID
    sub_weights = (SubWeights.none() if no_subs else SubWeights()).as_dict()

    fill_initial_squad(
        tag=tag,
        gw_range=gw_range,
        season=season,
        fpl_team_id=fpl_team_id,
        budget=budget,
        remove_zero=remove_zero,
        sub_weights=sub_weights,
        num_generations=num_generations,
        population_size=population_size,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        crossover_indpb=crossover_indpb,
        mutation_indpb=mutation_indpb,
        tournament_size=tournament_size,
        verbose=True,
        is_replay=is_replay,
    )
