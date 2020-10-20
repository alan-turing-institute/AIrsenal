#!/usr/bin/env python

"""
Find alternative team names for all the teams in the 2018/19 FPL.
"""
import os
import sys


import json

from fuzzywuzzy import fuzz

from airsenal.framework.data_fetcher import FPLDataFetcher


def find_best_match(fpl_teams, team):
    """
    use fuzzy matching to see if we can match
    names
    """
    best_ratio = 0.0
    best_match = None
    for t in fpl_teams:
        if fuzz.partial_ratio(t, team) > best_ratio:
            best_ratio = fuzz.partial_ratio(t, team)
            best_match = t
    print("Best match {}/{}, score {}".format(best_match, team, best_ratio))
    return best_match, best_ratio


if __name__ == "__main__":

    # get the team names as used in FPL
    df = FPLDataFetcher()
    teamdict = {}
    teamdata = df.get_current_team_data()
    for k in teamdata.keys():
        teamdict[teamdata[k]["name"]] = [teamdata[k]["short_name"]]
    #    teamdicts = [{teamdata[k]['name']:[teamdata[k]['short_name']]} \
    #                for k in teamdata.keys()]
    fpl_teams = list(teamdict.keys())
    # get the team names from the results csv
    missing = set()
    matched = set()
    history_teams = set()
    for season in ["1415", "1516", "1617", "1718"]:
        filename = "../data/results_{}.csv".format(season)
        for line in open(filename).readlines()[1:]:
            history_teams.add(line.split(",")[1])
            history_teams.add(line.split(",")[2])

    for team in history_teams:
        if team in fpl_teams:
            matched.add(team)
        else:
            t, score = find_best_match(fpl_teams, team)
            if score == 100:
                teamdict[t].append(team)
                matched.add(team)
            # ugh, ok, do the last few by hand
            elif team == "Manchester United":
                teamdict["Man Utd"].append(team)
                matched.add(team)
            elif team == "Manchester City":
                teamdict["Man City"].append(team)
                matched.add(team)
            elif team == "Tottenham Hotspur":
                teamdict["Spurs"].append(team)
                matched.add(team)
            else:
                missing.add(team)
    # matched teams should be all except promoted ones that haven't
    # been in the prem recently
    print("Num matched: {}".format(len(matched)))

    # print missing teams (should be the relegated ones
    print("Teams not in this seasons FPL: {}".format(missing))

    outfile = open("../data/alternative_team_names.json", "w")
    outfile.write(json.dumps(teamdict))
    outfile.close()
