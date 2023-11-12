"""
Script to apply recommended squad changes after transfers are made

"""

import argparse
from typing import List, Optional

from airsenal.framework.data_fetcher import FPLDataFetcher
from airsenal.framework.squad import Squad
from airsenal.framework.utils import (
    NEXT_GAMEWEEK,
    get_latest_prediction_tag,
    get_player,
    get_player_from_api_id,
)


def check_proceed(squad: Squad) -> bool:
    print(squad)
    proceed = input("Apply changes to lineup? (yes/no) ")
    if proceed == "yes":
        print("Applying Changes...")
        return True
    else:
        return False


def build_lineup_payload(squad: Squad) -> list:
    def to_dict(player, pos_int):
        return {
            "element": get_player(player.player_id).fpl_api_id,
            "position": pos_int,
            "is_captain": player.is_captain,
            "is_vice_captain": player.is_vice_captain,
        }

    payload = []
    # payload for starting lineup
    lineup = [p for p in squad.players if p.is_starting]
    position_integer = 1
    for position_category in ["GK", "DEF", "MID", "FWD"]:
        for p in lineup:
            if p.position == position_category:
                payload.append(to_dict(p, position_integer))
                position_integer += 1

    sub_gk = [p for p in squad.players if not p.is_starting and p.position == "GK"][0]
    payload.append(to_dict(sub_gk, 12))

    available_sub_positions = list(range(4))
    available_sub_positions.remove(sub_gk.sub_position)
    subs_outfield = [
        p for p in squad.players if not p.is_starting and p.position != "GK"
    ]
    for s in subs_outfield:
        payload.append(to_dict(s, 13 + available_sub_positions.index(s.sub_position)))

    return payload


def get_lineup_from_payload(lineup: dict) -> Squad:
    """
    inverse of build_lineup_payload. Returns a squad object from get_lineup

    lineup is a dictionary, with the entry "picks" being a list of dictionaries like:
    {"element":353,"position":1,"selling_price":55,"multiplier":1,"purchase_price":55,"is_captain":false,"is_vice_captain":false}
    """
    s = Squad()
    for p in lineup["picks"]:
        player = get_player_from_api_id(p["element"])
        s.add_player(player, check_budget=False)

    if s.is_complete():
        return s
    else:
        raise RuntimeError("Squad incomplete")


def make_squad_transfers(squad: Squad, priced_transfers: List[dict]) -> None:
    for t in priced_transfers:
        squad.remove_player(t[0][0], price=t[0][1])
        squad.add_player(t[1][0], price=t[1][1])


def set_lineup(
    fpl_team_id: Optional[int] = None,
    verbose: Optional[bool] = False,
    skip_check: bool = False,
) -> None:
    """
    Retrieve the latest lineup and apply the latest prediction to it.

    Note that this assumes that the prediction has been ran recently.
    """
    fetcher = FPLDataFetcher(fpl_team_id)
    print(f"fpl_team_id is {fetcher.FPL_TEAM_ID}")
    picks = fetcher.get_lineup()
    if verbose:
        print(f"Got picks {picks}")
    squad = get_lineup_from_payload(picks)
    if verbose:
        print(f"got squad: {squad}")

    squad.optimize_lineup(NEXT_GAMEWEEK, get_latest_prediction_tag())

    if check_proceed(squad) and not skip_check:
        payload = build_lineup_payload(squad)
        fetcher.post_lineup(payload)


def main():
    parser = argparse.ArgumentParser("Set the starting 11 and captain")
    parser.add_argument("--fpl_team_id", help="ID of the squad in FPL API", type=int)
    parser.add_argument("--confirm", help="skip confirmation step", action="store_true")
    args = parser.parse_args()
    try:
        set_lineup(args.fpl_team_id, skip_check=args.confirm)
    except Exception as e:
        raise Exception(
            "Something went wrong when setting lineup. Check your lineup manually on "
            "the web-site. If the problem persists, let us know on GitHub."
        ) from e


if __name__ == "__main__":
    main()
