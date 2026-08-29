"""
Find the alternative names each FPL player goes by in the other data sources.

Writes out {name_in_FPL: [short_name_in_FPL, other_name, ...]}.
"""

import json
from collections.abc import Callable

from thefuzz import fuzz

from airsenal.core.logging import get_logger
from airsenal.remote.fpl_api import FPLDataFetcher

logger = get_logger(__name__)


def find_best_match(
    fpl_players: list[str],
    player: str,
    fuzz_method: Callable[[str, str], int] = fuzz.ratio,
) -> tuple[str | None, int]:
    """
    Fuzzy-match a historical player name against the current FPL names.

    Returns:
        The best-matching current FPL name, and its score out of 100.
    """
    best_ratio = 0
    best_match = None
    for p in fpl_players:
        if fuzz_method(p, player) > best_ratio:
            best_ratio = fuzz_method(p, player)
            best_match = p

    return best_match, best_ratio


if __name__ == "__main__":
    # get the team names as used in FPL
    df = FPLDataFetcher()
    playerdict: dict[str, list[str]] = {}
    playerdata = df.get_player_summary_data()
    fpl_players_to_match = []
    # from the API we construct the player name from first_name and second_name
    for k in playerdata:
        player_name = f"{playerdata[k]['first_name']} {playerdata[k]['second_name']}"
        fpl_players_to_match.append(player_name)

    # get the player names from the fpl archives json
    matched: set[str] = set()
    history_players: set[str] = set()
    for season in ["2122", "2021", "1920"]:
        filename = f"../data/player_summary_{season}.json"
        with open(filename) as f:
            player_data = json.load(f)
        for p in player_data:
            history_players.add(p["name"])
    count = 0
    for player in history_players:
        # see if the names match exactly
        if player in fpl_players_to_match:
            matched.add(player)
            fpl_players_to_match.remove(player)
        else:
            # try two separate fuzzy methods, the first
            # is the simplest, but not best for players whose
            # names swap order
            p, score = find_best_match(
                fpl_players_to_match, player, fuzz_method=fuzz.ratio
            )
            if p is None:
                logger.warning("Could not find match for %s", player)
                continue
            if score > 70:
                add_player = input(
                    f"Add {p} : {player}  (score (from ratio)={score})? (y/n):"
                )
                if add_player.lower() == "y":
                    if p not in playerdict:
                        playerdict[p] = []
                    playerdict[p].append(player)
                    matched.add(player)
                    fpl_players_to_match.remove(p)
                    count += 1
            else:
                # this method should be better for swaps of first and second name
                p, score = find_best_match(
                    fpl_players_to_match, player, fuzz_method=fuzz.token_sort_ratio
                )
                if p is None:
                    logger.warning("Could not find match for %s", player)
                    continue
                if score > 80:
                    add_player = input(
                        f"Add {p} : {player}  (score (from token_sort_ratio)={score})? "
                        "(y/n):"
                    )
                    if add_player.lower() == "y":
                        if p not in playerdict:
                            playerdict[p] = []
                        playerdict[p].append(player)
                        matched.add(player)
                        fpl_players_to_match.remove(p)
    logger.info("Num matched: %s", len(matched))

    # write an output csv file with each line containing all possible
    # alternative names for a given current-season name
    with open("../data/alternative_player_names.csv", "w") as outfile:
        for fpl_name, alternatives in playerdict.items():
            outfile.write(f"{fpl_name},{','.join(alternatives)}\n")
