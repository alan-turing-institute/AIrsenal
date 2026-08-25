"""Posting AIrsenal's recommended starting eleven, captain and bench order to FPL."""

from typing import Any

from airsenal.core.console import console
from airsenal.core.logging import get_logger
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.db.queries.players import get_player, get_player_from_api_id
from airsenal.db.queries.tags import get_latest_prediction_tag
from airsenal.game.enums import Position
from airsenal.remote.fpl_api import FPLDataFetcher
from airsenal.reporting.squad_view import formation_table
from airsenal.squad.player import SquadPlayer, bench_position
from airsenal.squad.squad import Squad

logger = get_logger(__name__)


def check_proceed(squad: Squad, tag: str, gameweek: int) -> bool:
    console.print(formation_table(squad, tag, gameweek))
    proceed = input("Apply changes to lineup? (yes/no) ")
    if proceed == "yes":
        logger.info("Applying Changes...")
        return True
    return False


def build_lineup_payload(squad: Squad) -> list[dict[str, Any]]:
    def to_dict(player: SquadPlayer, pos_int: int) -> dict[str, Any]:
        p = get_player(player.player_id)
        if p is None:
            msg = f"Player with ID {player.player_id} not found"
            raise ValueError(msg)
        if p.fpl_api_id is None:
            msg = f"Player {p} has no FPL API ID"
            raise ValueError(msg)
        return {
            "element": p.fpl_api_id,
            "position": pos_int,
            "is_captain": player.is_captain,
            "is_vice_captain": player.is_vice_captain,
        }

    payload: list[dict[str, Any]] = []
    # payload for starting lineup
    lineup = [p for p in squad.players if p.is_starting]
    position_integer = 1
    for position_category in list(Position.back_to_front()):
        for p in lineup:
            if p.position == position_category:
                payload.append(to_dict(p, position_integer))
                position_integer += 1

    sub_gk = next(
        p for p in squad.players if not p.is_starting and p.position == Position.GK
    )
    payload.append(to_dict(sub_gk, 12))

    available_sub_positions = list(range(4))
    available_sub_positions.remove(bench_position(sub_gk))
    subs_outfield = [
        p for p in squad.players if not p.is_starting and p.position != Position.GK
    ]
    for s in subs_outfield:
        payload.append(
            to_dict(s, 13 + available_sub_positions.index(bench_position(s)))
        )

    return payload


def get_lineup_from_payload(lineup: dict[str, Any]) -> Squad:
    """
    inverse of build_lineup_payload. Returns a squad object from get_lineup

    lineup is a dictionary, with the entry "picks" being a list of dictionaries like:
    {"element":353,"position":1,"selling_price":55,"multiplier":1,"purchase_price":55,"is_captain":false,"is_vice_captain":false}
    """
    s = Squad()
    for p in lineup["picks"]:
        player = get_player_from_api_id(p["element"])
        if player is None:
            msg = f"Player with API ID {p['element']} not found"
            raise ValueError(msg)
        s.add_player(player, check_budget=False)

    if s.is_complete():
        return s
    msg = "Squad incomplete"
    raise RuntimeError(msg)


def set_lineup(
    fpl_team_id: int | None = None,
    skip_check: bool = False,
) -> None:
    """
    Retrieve the latest lineup and apply the latest prediction to it.

    Note that this assumes that the prediction has been ran recently.
    """
    fetcher = FPLDataFetcher(fpl_team_id)
    logger.info("fpl_team_id is %s", fetcher.FPL_TEAM_ID)
    picks = fetcher.get_lineup()
    logger.debug("Got picks %s", picks)
    squad = get_lineup_from_payload(picks)
    logger.debug("got squad: %s", squad)

    tag = get_latest_prediction_tag()
    squad.optimize_lineup(next_gameweek(), tag)

    if not skip_check and not check_proceed(squad, tag, next_gameweek()):
        logger.info("Not proceeding with lineup update")
        return

    payload = build_lineup_payload(squad)
    fetcher.post_lineup(payload)
