import sys
from dataclasses import replace

from rich.panel import Panel
from rich.text import Text

from airsenal.core.console import console, price_str, progress_bar, table
from airsenal.core.enums import Position
from airsenal.core.logging import get_logger
from airsenal.core.season import CURRENT_SEASON
from airsenal.db.queries.gameweeks import get_max_gameweek, next_gameweek
from airsenal.db.queries.tags import check_tag_valid, get_latest_prediction_tag
from airsenal.fetch.fpl_api import require_fpl_team_id
from airsenal.optimization.config import (
    GeneticAlgorithmConfig,
    SquadScoringConfig,
    SubWeights,
)
from airsenal.optimization.persist import (
    fill_initial_suggestion_table,
    fill_initial_transaction_table,
)
from airsenal.optimization.protocols import (
    SquadOptimizer,
    SquadRequest,
    progress_total,
)
from airsenal.optimization.squad_optimizers import GeneticSquadOptimizer
from airsenal.optimization.squad_score import get_discounted_squad_score
from airsenal.reporting.squad_view import formation_table
from airsenal.squad.squad import Squad

logger = get_logger(__name__)

# str rather than Position: a squad player's position is a plain string
positions: list[str] = list(Position.front_to_back())  # front-to-back


def fill_initial_squad(
    tag: str,
    gameweeks: list[int],
    season: str,
    fpl_team_id: int,
    optimizer: SquadOptimizer | None = None,
    scoring: SquadScoringConfig | None = None,
    remove_zero: bool = True,
    is_replay: bool = False,  # for replaying seasons
    chip_gameweeks: dict[str, int] | None = None,
) -> Squad:
    if optimizer is None:
        optimizer = GeneticSquadOptimizer()
    scoring = scoring if scoring is not None else SquadScoringConfig()
    sub_weights = scoring.sub_weights.as_dict()
    with progress_bar(transient=True) as progress:
        # the optimizer says how many steps it will take, so the bar cannot drift
        # away from what actually happens; the best score so far is the part worth
        # watching, so it goes in the description
        task = progress.add_task(
            "Optimising full squad", total=progress_total(optimizer)
        )

        def report_generation(best_score: float) -> None:
            progress.update(
                task,
                advance=1,
                description=f"Optimising full squad (best {best_score:.1f}pts)",
            )

        best_squad = optimizer.optimize(
            SquadRequest(
                gameweeks=gameweeks,
                tag=tag,
                season=season,
                scoring=scoring,
                remove_zero=remove_zero,
                progress=report_generation,
            )
        )

    gw_start = gameweeks[0]
    optimised_score = get_discounted_squad_score(
        best_squad,
        gameweeks,
        tag,
        gw_start,
        sub_weights=sub_weights,
    )

    chip_gameweeks = chip_gameweeks or {}

    summary = Text()
    summary.append(
        f"Gameweeks: {min(gameweeks)}-{max(gameweeks)}\n"
        if min(gameweeks) != max(gameweeks)
        else f"Gameweek: {min(gameweeks)}\n",
        style="bold",
    )
    summary.append(f"Team ID: {fpl_team_id}\n")
    summary.append(f"Optimised Score: {optimised_score:.1f}pts\n", style="bold green")
    console.print(Panel(summary, title="Optimisation Result", expand=False))

    plan_table = table(
        "Gameweek",
        "Transfers",
        "Chip",
        "Points Hit",
        "Predicted Score",
        title="Plan",
    )
    for gw in gameweeks:
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
        plan_table.add_row(
            str(gw),
            str(len(best_squad.players)) if gw == gw_start else "0",
            chip,
            "0pts",
            f"{pred_pts:.1f}pts",
        )
    console.print(plan_table)

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
        logger.error(
            "Database does not contain predictions for all the specified "
            "optimsation gameweeks.\nPlease run 'airsenal_run_prediction' first "
            "with the same input gameweeks and season you specified here."
        )
        sys.exit(1)
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
