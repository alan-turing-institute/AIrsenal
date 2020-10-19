#!/usr/bin/env python

"""
Find alternative player names for all the players in the 2018/19 FPL.
Write out a dict of the format
{<name_in_FPL> : [ <short_name_in_FPL>,<other_name_1>,... ],
...
}
"""
import os
import sys


import json

from fuzzywuzzy import fuzz

from airsenal.framework.data_fetcher import FPLDataFetcher


def find_best_match(fpl_players, player):
    """
    use fuzzy matching to see if we can match
    names
    """
    best_ratio = 0.0
    best_match = None
    for p in fpl_players:
        if fuzz.partial_ratio(p, player) > best_ratio:
            best_ratio = fuzz.partial_ratio(p, player)
            best_match = p
    #   print("Best match {}/{}, score {}".format(best_match,
    #                                             player,
    #                                             best_ratio))
    return best_match, best_ratio


if __name__ == "__main__":

    # get the team names as used in FPL
    df = FPLDataFetcher()
    playerdict = {}
    playerdata = df.get_player_summary_data()
    for k in playerdata.keys():
        player_name = "{} {}".format(
            playerdata[k]["first_name"], playerdata[k]["second_name"]
        )
        playerdict[player_name] = [playerdata[k]["web_name"]]

    fpl_players_to_match = list(playerdict.keys())
    # get the player names from the fpl archives json
    missing = set()
    matched = set()
    history_players = set()
    for season in ["1516", "1617"]:
        filename = "../data/player_summary_{}.json".format(season)
        player_data = json.load(open(filename))
        for p in player_data:
            history_players.add(p["name"])

    for player in history_players:
        if player in fpl_players_to_match:
            matched.add(player)
            fpl_players_to_match.remove(player)
        else:
            p, score = find_best_match(fpl_players_to_match, player)
            if score > 90:
                if "Sessegnon" in player:  # false matches
                    missing.add(player)
                    continue
                if "Eder" in player:  # and another one
                    missing.add(player)
                    continue
                print("good match", p, player, score)
                playerdict[p].append(player)
                matched.add(player)
                fpl_players_to_match.remove(p)
            else:
                missing.add(player)

    print("Num matched: {}".format(len(matched)))
    print("Num missing: {}".format(len(missing)))

    # print missing teams (should be the relegated ones
    print("Players not in this seasons FPL: {}".format(missing))

    outfile = open("../data/alternative_player_names.json", "w")
    outfile.write(json.dumps(playerdict))
    outfile.close()
