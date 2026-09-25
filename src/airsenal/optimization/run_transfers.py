"""
Running a transfer search: everything around the algorithm itself.

Fetching the starting squad, persisting the suggestions and reporting the result.
"""

import json
from dataclasses import replace
from pathlib import Path

from sqlalchemy.orm import Session

from airsenal.core.console import console
from airsenal.core.copy import fastcopy
from airsenal.core.logging import get_logger
from airsenal.db.queries.players import get_player, get_player_name
from airsenal.db.session import get_session
from airsenal.game.chips import chips_used_up
from airsenal.game.enums import Chip
from airsenal.game.mappings import chips_by_api_name
from airsenal.game.season import CURRENT_SEASON
from airsenal.optimization.chip_timing import chip_gameweeks, decide_chips
from airsenal.optimization.moves import ChipGameweeks, ChipSchedule
from airsenal.optimization.persist import fill_suggestion_table, fill_transaction_table
from airsenal.optimization.plan import Plan
from airsenal.optimization.protocols import (
    SquadOptimizer,
    TransferConstraints,
    TransferOptimizer,
    TransferSearchRequest,
)
from airsenal.optimization.run_squad import build_new_squad
from airsenal.optimization.squad_score import SquadScoringConfig
from airsenal.optimization.transfer_optimizers import (
    TreeSearchOptimizer,
)
from airsenal.remote.discord import post_webhook
from airsenal.remote.fpl_api import FPLDataFetcher, get_fetcher, require_fpl_team_id
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
    """Write every plan considered to one JSON file, for debugging."""
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
    fetcher: FPLDataFetcher | None = None,
    dbsession: Session | None = None,
) -> list[TransferRow]:
    """
    Replay the plan's transfers to find the price each was made at.

    Every price is read as at the plan's root gameweek, because that is what the
    search spent: see `optimization.protocols.TransferRequest.root_gameweek`.
    """
    dbsession = get_session(dbsession)
    squad = starting_squad
    # The gameweek the prices come from, as opposed to the gameweek each row is
    # reported under, which is the one the transfer is made in.
    priced_at = plan.root_gameweek
    rows = []
    for outcome in plan.outcomes:
        gameweek = outcome.gameweek
        # A free hit is reverted after the gameweek it is played in, so the search
        # plans the next gameweek from the squad that went into this one. The walk
        # has to put it back the same way or it prices the following gameweek's
        # transfers against players the entry never owned.
        before = squad if outcome.move.carry_forward else fastcopy(squad)
        for pid_out, pid_in in zip(
            outcome.players_out, outcome.players_in, strict=True
        ):
            out_player = squad.get_player_from_id(pid_out)
            sale_price = squad.get_sell_price_for_player(
                pid_out,
                use_api=use_api,
                gameweek=priced_at,
                dbsession=dbsession,
                fetcher=fetcher,
            )
            squad.remove_player(pid_out, price=sale_price, gameweek=priced_at)

            in_player = get_player(pid_in, dbsession=dbsession)
            purchase_price = in_player.price(priced_at, season) if in_player else None
            squad.add_player(
                pid_in,
                price=purchase_price,
                gameweek=priced_at,
                check_budget=False,
                check_team=False,
                dbsession=dbsession,
            )
            rows.append(
                TransferRow(
                    gameweek=gameweek,
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
                    team_in=in_player.team(priced_at, season) if in_player else None,
                    purchase_price=purchase_price,
                )
            )
        squad = before
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


def squad_for_next_gameweek(
    plan: Plan,
    season: str = CURRENT_SEASON,
    fpl_team_id: int | None = None,
    use_api: bool = False,
) -> Squad:
    """The squad the plan's first gameweek leaves us with."""
    outcome = plan.outcomes[0]
    gameweek = outcome.gameweek
    squad = get_starting_squad(
        gameweek=gameweek,
        season=season,
        fpl_team_id=fpl_team_id,
        use_api=use_api,
    )
    # Every price and club here is read as at the gameweek the move is made in.
    # Left to default these take the *current* season's next gameweek, which is
    # not this plan's gameweek at all when replaying a past season.
    for pid_out in outcome.players_out:
        squad.remove_player(pid_out, gameweek=gameweek, use_api=use_api)
    for pid_in in outcome.players_in:
        # A wildcard or free hit adds fifteen players one at a time, so a wrong
        # sale price surfaces here as a squad that cannot afford the last few.
        if not squad.add_player(pid_in, gameweek=gameweek):
            msg = (
                f"Could not add player {pid_in} to the gameweek {gameweek} squad, "
                f"which has {squad.budget} in the bank: the plan's transfers do "
                "not fit the squad they are applied to."
            )
            raise RuntimeError(msg)
    return squad


def available_chips(
    chips: ChipGameweeks,
    gameweek: int,
    season: str = CURRENT_SEASON,
    fpl_team_id: int | None = None,
    fetcher: FPLDataFetcher | None = None,
) -> frozenset[Chip]:
    """
    The chips the entry has left to play in `gameweek`.

    From the API when given a fetcher, which needs a login; otherwise every chip
    `chips.played` has not used up, which is what a replay knows.
    """
    if fetcher is None:
        return frozenset(Chip) - set(chips_used_up(chips.played, gameweek, season))
    names = fetcher.get_available_chips(fpl_team_id)
    unknown = sorted(set(names) - set(chips_by_api_name))
    if unknown:
        msg = f"The FPL API reported chips AIrsenal does not know: {unknown}"
        raise ValueError(msg)
    return frozenset(chips_by_api_name[name] for name in names)


def _chips_by_heuristic(
    chips: ChipGameweeks,
    squad: Squad,
    gameweeks: list[int],
    tag: str,
    season: str = CURRENT_SEASON,
    fpl_team_id: int | None = None,
    fetcher: FPLDataFetcher | None = None,
) -> ChipGameweeks:
    """The chip gameweeks `chip_timing` decides, keeping what was played before."""
    available = available_chips(
        chips, gameweeks[0], season=season, fpl_team_id=fpl_team_id, fetcher=fetcher
    )
    decisions = decide_chips(squad, available, gameweeks, tag, season=season)
    for gameweek, chip, why in decisions:
        if chip is not None:
            logger.info("Chip heuristic: %s in gameweek %s - %s", chip, gameweek, why)
    return replace(chip_gameweeks(decisions), played=chips.played, heuristic=True)


def run_optimization(
    gameweeks: list[int],
    tag: str,
    season: str = CURRENT_SEASON,
    fpl_team_id: int | None = None,
    chips: ChipGameweeks | None = None,
    num_free_transfers: int | None = None,
    constraints: TransferConstraints | None = None,
    optimizer: TransferOptimizer | None = None,
    squad_optimizer: SquadOptimizer | None = None,
    scoring: SquadScoringConfig | None = None,
    save_plans: Path | None = None,
    is_replay: bool = False,
) -> tuple[Squad, Plan | None]:
    """
    Search every move-and-gameweek combination for the best whole-window plan.

    Each chip gameweek is -1 not to play that chip at all, 0 to let the search
    choose the gameweek, or the gameweek to play it in.
    `chips.played` are the chips already spent before `gameweeks`.
    """
    if chips is None:
        chips = ChipGameweeks()
    if constraints is None:
        constraints = TransferConstraints()
    if optimizer is None:
        optimizer = TreeSearchOptimizer()
    if scoring is None:
        scoring = SquadScoringConfig()
    fpl_team_id = require_fpl_team_id(fpl_team_id)
    fetcher = get_fetcher(fpl_team_id)

    with console.status("Optimising transfers..."):
        logger.info("Running optimization with fpl_team_id %s", fpl_team_id)
        use_api = season == CURRENT_SEASON and not is_replay
        try:
            starting_squad = get_starting_squad(
                gameweek=gameweeks[0],
                season=season,
                fpl_team_id=fpl_team_id,
                use_api=use_api,
                fetcher=fetcher,
            )
        except (ValueError, TypeError):
            # first gameweek for this squad?
            logger.warning(
                "No existing squad or transfers found for team_id %s", fpl_team_id
            )
            logger.info("Will suggest a new starting squad:")
            return build_new_squad(
                tag=tag,
                gameweeks=gameweeks,
                season=season,
                fpl_team_id=fpl_team_id,
                optimizer=squad_optimizer,
                scoring=scoring,
                is_replay=is_replay,
                chips=chips,
            ), None

        if num_free_transfers is None:
            num_free_transfers = get_free_transfers(
                gameweeks[0],
                fpl_team_id=fpl_team_id,
                season=season,
                fetcher=fetcher,
                is_replay=is_replay,
            )
        logger.info("Starting with %s free transfers", num_free_transfers)

        if chips.heuristic:
            chips = _chips_by_heuristic(
                chips,
                starting_squad,
                gameweeks,
                tag,
                season=season,
                fpl_team_id=fpl_team_id,
                fetcher=fetcher if use_api else None,
            )
        chip_schedule = ChipSchedule.from_gameweeks(gameweeks, chips)

        result = optimizer.search(
            TransferSearchRequest(
                starting_squad=starting_squad,
                gameweeks=gameweeks,
                tag=tag,
                season=season,
                chip_schedule=chip_schedule,
                num_free_transfers=num_free_transfers,
                constraints=constraints,
                scoring=scoring,
                squad_optimizer=squad_optimizer,
                chips_played=chips.played,
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
            # a replay imitates applying the suggestions by recording transactions
            fill_transaction_table(
                starting_squad,
                best_plan,
                tag=tag,
                season=season,
                fpl_team_id=fpl_team_id,
            )

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
        best_plan,
        fastcopy(starting_squad),
        season,
        use_api=use_api,
        fetcher=fetcher,
    )
    print_plan_table(plan)
    print_transfer_table(transfers)

    best_squad = squad_for_next_gameweek(
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

    if not is_replay:
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
