"""
Running a transfer search: everything around the algorithm itself.

Fetching the starting squad, persisting the suggestions and reporting the result.
The search is behind the `TransferOptimizer` interface, so swapping it does not
mean reimplementing any of this.
"""

import json
from pathlib import Path

from sqlalchemy.orm import Session

from airsenal.core.console import console
from airsenal.core.copy import fastcopy
from airsenal.core.logging import get_logger
from airsenal.db.queries.players import get_player, get_player_name
from airsenal.db.session import get_session
from airsenal.game.enums import Chip
from airsenal.game.season import CURRENT_SEASON
from airsenal.optimization.moves import ChipSchedule, ChipWeeks
from airsenal.optimization.persist import fill_suggestion_table, fill_transaction_table
from airsenal.optimization.plan import Plan
from airsenal.optimization.protocols import (
    SquadOptimizer,
    TransferConstraints,
    TransferOptimizer,
    TransferSearchRequest,
)
from airsenal.optimization.run_squad import build_new_squad
from airsenal.optimization.squad_optimizers import (
    GeneticSquadOptimizer,
)
from airsenal.optimization.squad_score import SquadScoringConfig
from airsenal.optimization.transfer_optimizers import (
    TreeSearchOptimizer,
)
from airsenal.remote.discord import post_webhook
from airsenal.remote.fpl_api import get_fetcher, require_fpl_team_id
from airsenal.reporting.optimization import (
    GameweekRow,
    TransferRow,
    discord_payload,
    lineup_strings,
    print_plan_table,
    print_result_panel,
    print_transfer_table,
)
from airsenal.reporting.squad_view import formation_table
from airsenal.squad.history import get_starting_squad
from airsenal.squad.squad import Squad
from airsenal.squad.state import get_free_transfers

logger = get_logger(__name__)


def save_plan_dump(plans: list[Plan], directory: Path, tag: str) -> None:
    """
    Write every plan considered to one JSON file, for debugging.

    The search itself keeps plans in memory; this exists only because
    inspecting the whole tree is occasionally the fastest way to understand a
    surprising suggestion.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"plans_{tag}.json"
    with path.open("w") as f:
        json.dump([p.to_dict() for p in plans], f, indent=2)
    logger.info("Wrote %s plans to %s", len(plans), path)


def transfer_rows(
    plan: Plan,
    starting_squad: Squad,
    season: str,
    use_api: bool,
    dbsession: Session | None = None,
) -> list[TransferRow]:
    """
    Replay the plan's transfers to find the price each was made at.

    Simulation, not rendering: applying a transfer to a squad is what this
    package knows how to do, and the sale price of a player depends on what the
    squad paid for them, so the walk has to happen in order.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    squad = starting_squad
    rows = []
    for outcome in plan.outcomes:
        gw = outcome.gameweek
        for pid_out, pid_in in zip(
            outcome.players_out, outcome.players_in, strict=True
        ):
            out_player = squad.get_player_from_id(pid_out)
            sale_price = squad.get_sell_price_for_player(
                pid_out, use_api=use_api, gameweek=gw, dbsession=dbsession
            )
            squad.remove_player(pid_out, price=sale_price, gameweek=gw)

            in_player = get_player(pid_in, dbsession=dbsession)
            purchase_price = in_player.price(season, gw) if in_player else None
            squad.add_player(
                pid_in,
                price=purchase_price,
                gameweek=gw,
                check_budget=False,
                check_team=False,
                dbsession=dbsession,
            )
            rows.append(
                TransferRow(
                    gameweek=gw,
                    player_out=str(out_player),
                    position_out=out_player.position,
                    team_out=out_player.team,
                    sale_price=sale_price,
                    player_in=(
                        str(in_player)
                        if in_player
                        else str(get_player_name(pid_in) or pid_in)
                    ),
                    position_in=in_player.position(season) if in_player else None,
                    team_in=in_player.team(season, gw) if in_player else None,
                    purchase_price=purchase_price,
                )
            )
    return rows


def plan_rows(plan: Plan) -> list[GameweekRow]:
    """The plan's per-gameweek moves, in the shape the summary table renders."""
    return [
        GameweekRow(
            gameweek=outcome.gameweek,
            transfers=outcome.move.label(),
            chip=str(outcome.chip) if outcome.chip else None,
            points_hit=outcome.points_hit,
            predicted_points=outcome.undiscounted_points,
        )
        for outcome in plan.outcomes
    ]


def squad_for_next_gw(
    plan: Plan,
    season: str = CURRENT_SEASON,
    fpl_team_id: int | None = None,
    use_api: bool = False,
) -> Squad:
    """The squad the plan's first gameweek leaves us with."""
    outcome = plan.outcomes[0]
    squad = get_starting_squad(
        next_gw=outcome.gameweek,
        season=season,
        fpl_team_id=fpl_team_id,
        use_api=use_api,
    )
    for pid_out in outcome.players_out:
        squad.remove_player(pid_out)
    for pid_in in outcome.players_in:
        squad.add_player(pid_in)
    return squad


def new_squad_from_scratch(
    gameweeks: list[int],
    tag: str,
    season: str,
    fpl_team_id: int,
    chips: ChipWeeks,
    squad_optimizer: SquadOptimizer | None = None,
    scoring: SquadScoringConfig | None = None,
    is_replay: bool = False,
) -> Squad:
    """
    Build a squad from nothing, because there turned out to be nothing to
    transfer from.

    Whether to build rather than transfer is `AIrsenalPipeline._is_new_squad`'s
    decision, and this is not a second copy of it: it is the recovery for a
    database with no transactions for this entry, which is only discoverable by
    trying to load the squad.
    """
    if squad_optimizer is None:
        squad_optimizer = GeneticSquadOptimizer()
    return build_new_squad(
        tag=tag,
        gameweeks=gameweeks,
        season=season,
        fpl_team_id=fpl_team_id,
        optimizer=squad_optimizer,
        scoring=scoring,
        chips=chips,
        is_replay=is_replay,
    )


def run_optimization(
    gameweeks: list[int],
    tag: str,
    season: str = CURRENT_SEASON,
    fpl_team_id: int | None = None,
    chips: ChipWeeks | None = None,
    num_free_transfers: int | None = None,
    constraints: TransferConstraints | None = None,
    optimizer: TransferOptimizer | None = None,
    squad_optimizer: SquadOptimizer | None = None,
    scoring: SquadScoringConfig | None = None,
    save_plans: Path | None = None,
    is_replay: bool = False,  # for replaying seasons
) -> tuple[Squad, Plan | None]:
    """
    This is the actual main function that sets up the multiprocessing
    and calls the optimize function for every move/gameweek
    combination, to find the best plan.
    The chip-related variables e.g. wildcard_week are -1 if that chip
    is not to be played, 0 for 'play it any week', or the gw in which
    it should be played.
    """
    if chips is None:
        chips = ChipWeeks()
    if constraints is None:
        constraints = TransferConstraints()
    if optimizer is None:
        optimizer = TreeSearchOptimizer()
    if scoring is None:
        scoring = SquadScoringConfig()
    fpl_team_id = require_fpl_team_id(fpl_team_id)

    with console.status("Optimising transfers..."):
        logger.info("Running optimization with fpl_team_id %s", fpl_team_id)
        use_api = season == CURRENT_SEASON and not is_replay
        try:
            starting_squad = get_starting_squad(
                next_gw=gameweeks[0],
                season=season,
                fpl_team_id=fpl_team_id,
                use_api=use_api,
                fetcher=get_fetcher(),
            )
        except (ValueError, TypeError):
            # first week for this squad?
            logger.warning(
                "No existing squad or transfers found for team_id %s", fpl_team_id
            )
            logger.info("Will suggest a new starting squad:")
            return new_squad_from_scratch(
                gameweeks,
                tag,
                season,
                fpl_team_id,
                chips,
                squad_optimizer,
                scoring,
                is_replay=is_replay,
            ), None
        # if we got to here, we can assume we are optimizing an existing squad.

        # How many free transfers are we starting with?
        if num_free_transfers is None:
            num_free_transfers = get_free_transfers(
                fpl_team_id,
                gameweeks[0],
                season=season,
                fetcher=get_fetcher(),
                is_replay=is_replay,
            )
        logger.info("Starting with %s free transfers", num_free_transfers)

        # Work out what chips we definitely or possibly will play in each gw
        chip_schedule = ChipSchedule.from_weeks(gameweeks, chips)

        result = optimizer.search(
            TransferSearchRequest(
                starting_squad=starting_squad,
                gameweeks=gameweeks,
                tag=tag,
                season=season,
                chip_schedule=chip_schedule,
                num_free_transfers=num_free_transfers,
                constraints=constraints,
                # so the search weighs the bench the same way the squad builder
                # does
                scoring=scoring,
                # reaches the wildcard and free-hit rebuilds inside the search,
                # not only the from-scratch fallback above
                squad_optimizer=squad_optimizer,
            )
        )

        if save_plans is not None:
            save_plan_dump(list(result.considered), save_plans, tag)
        best_plan = result.best
        if result.baseline is None:
            logger.warning("No baseline plan was evaluated")
        baseline_score = result.baseline_score
        fill_suggestion_table(baseline_score, best_plan, season, fpl_team_id)
        if is_replay:
            # simulating a previous season, so imitate applying transfers by adding
            # the suggestions to the Transaction table
            fill_transaction_table(starting_squad, best_plan, season, fpl_team_id, tag)

    console.print()

    print_result_panel(
        gameweeks=list(best_plan.gameweeks),
        fpl_team_id=fpl_team_id,
        optimised_score=best_plan.total_score,
        baseline_score=baseline_score,
        points_hit=best_plan.total_points_hit,
        chips=tuple(str(c) for c in best_plan.chips_played if c),
    )
    plan = plan_rows(best_plan)
    transfers = transfer_rows(
        best_plan, fastcopy(starting_squad), season, use_api=use_api
    )
    print_plan_table(plan)
    print_transfer_table(transfers)

    best_squad = squad_for_next_gw(
        best_plan, season=season, fpl_team_id=fpl_team_id, use_api=use_api
    )
    console.print(
        formation_table(
            best_squad,
            tag,
            best_plan.outcomes[0].gameweek,
            bench_boost=best_plan.outcomes[0].chip is Chip.BENCH_BOOST,
            triple_captain=best_plan.outcomes[0].chip is Chip.TRIPLE_CAPTAIN,
        )
    )

    post_webhook(
        discord_payload(
            plan,
            transfers,
            lineup_strings(
                best_squad, best_plan.total_score, baseline_score, fpl_team_id
            ),
        )
    )

    return best_squad, best_plan
