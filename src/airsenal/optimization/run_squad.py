"""Running a from-scratch squad build: everything around the optimizer itself."""

from airsenal.core.console import console, progress_bar
from airsenal.core.enums import Chip
from airsenal.core.logging import get_logger
from airsenal.optimization.moves import ChipWeeks
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
from airsenal.optimization.squad_score import (
    SquadScoringConfig,
    get_discounted_squad_score,
)
from airsenal.reporting.optimization import (
    GameweekRow,
    print_plan_table,
    print_result_panel,
    print_squad_table,
)
from airsenal.reporting.squad_view import formation_table
from airsenal.squad.squad import Squad

logger = get_logger(__name__)


def _chip_label(chips: ChipWeeks, gameweek: int) -> str | None:
    """
    The chip this gameweek's row should show.

    Only the two that leave the squad alone: a wildcard or free hit is what a
    from-scratch build already is, so naming it in the table says nothing.
    """
    chip = chips.chip_in(gameweek)
    return str(chip) if chip in (Chip.BENCH_BOOST, Chip.TRIPLE_CAPTAIN) else None


def fill_initial_squad(
    tag: str,
    gameweeks: list[int],
    season: str,
    fpl_team_id: int,
    optimizer: SquadOptimizer | None = None,
    scoring: SquadScoringConfig | None = None,
    remove_zero: bool = True,
    is_replay: bool = False,  # for replaying seasons
    chips: ChipWeeks | None = None,
) -> Squad:
    if optimizer is None:
        optimizer = GeneticSquadOptimizer()
    scoring = scoring if scoring is not None else SquadScoringConfig()
    sub_weights = scoring.sub_weights
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

    chips = chips if chips is not None else ChipWeeks()

    print_result_panel(
        gameweeks=gameweeks,
        fpl_team_id=fpl_team_id,
        optimised_score=optimised_score,
    )
    print_plan_table(
        [
            GameweekRow(
                gameweek=gw,
                # every player is new in the first gameweek, and kept after that
                transfers=str(len(best_squad.players)) if gw == gw_start else "0",
                chip=_chip_label(chips, gw),
                points_hit=0,
                predicted_points=best_squad.get_expected_points(
                    gw,
                    tag,
                    bench_boost=chips.bench_boost == gw,
                    triple_captain=chips.triple_captain == gw,
                ),
            )
            for gw in gameweeks
        ]
    )
    print_squad_table(best_squad.players)
    console.print(
        formation_table(
            best_squad,
            tag,
            gw_start,
            bench_boost=chips.bench_boost == gw_start,
            triple_captain=chips.triple_captain == gw_start,
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
