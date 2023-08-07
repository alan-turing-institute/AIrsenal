#!/usr/bin/env python

"""
Find alternative player names for all the players in the 2018/19 FPL.
Write out a dict of the format
{<name_in_FPL> : [ <short_name_in_FPL>,<other_name_1>,... ],
...
}
"""

import json
from typing import Callable, List, Tuple

from fuzzywuzzy import fuzz

from airsenal.framework.data_fetcher import FPLDataFetcher


def find_best_match(
    fpl_players: List[str], player: str, fuzz_method: Callable = fuzz.ratio
) -> Tuple[str, int]:
    """
    use fuzzy matching to see if we can match names

    Parameters
    ==========
    fpl_players: list of str, current FPL player names
    player: str, player name from previous season
    fuzz_method: function from fuzzywuzzy

    Returns:
    ========
    best_match: str, the current FPL player name that best matches the
                     historical one
    best_ratio: int, the score for the match, range 1-100.
    """
    best_ratio = 0.0
    best_match = None
    for p in fpl_players:
        if fuzz_method(p, player) > best_ratio:
            best_ratio = fuzz_method(p, player)
            best_match = p

    return best_match, best_ratio


if __name__ == "__main__":
    # get the team names as used in FPL
    df = FPLDataFetcher()
    playerdict = {}
    playerdata = df.get_player_summary_data()
    fpl_players_to_match = []
    # from the API we construct the player name from first_name and second_name
    for k in playerdata.keys():
        player_name = f"{playerdata[k]['first_name']} {playerdata[k]['second_name']}"
        fpl_players_to_match.append(player_name)

    # get the player names from the fpl archives json
    missing = set()
    matched = set()
    history_players = set()
    for season in ["2122", "2021", "1920"]:
        filename = f"../data/player_summary_{season}.json"
        player_data = json.load(open(filename))
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
            if score > 70:
                add_player = input(
                    f"Add {p} : {player}  (score (from ratio)={score})? (y/n):"
                )
                if add_player.lower() == "y":
                    if p not in playerdict.keys():
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
                if score > 80:
                    add_player = input(
                        f"Add {p} : {player}  (score (from token_sort_ratio)={score})? "
                        "(y/n):"
                    )
                    if add_player.lower() == "y":
                        if p not in playerdict.keys():
                            playerdict[p] = []
                        playerdict[p].append(player)
                        matched.add(player)
                        fpl_players_to_match.remove(p)
    print(f"Num matched: {len(matched)}")

    # write an output csv file with each line containing all possible
    # alternative names for a given current-season name
    with open("../data/alternative_player_names.csv", "w") as outfile:
        for k, v in playerdict.items():
            outfile.write(f"{k},{','.join(v)}\n")
