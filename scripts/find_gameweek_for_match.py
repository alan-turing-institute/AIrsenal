#!/usr/bin/env python

import sys
import json


"""
The results_xxyy.csv don't have gameweek info in, but we should be able to
match them up with the detailed player info in player_details_xxyy.json
"""

team_name_json = json.load(open("../data/alternative_team_names.json"))

def find_players_for_team(team,summary_data):
    """
    return a list of player names who played for specified team
    in the summary data
    """
    # map given team name to the name used in the summary data json
    players = []
#    print("Looking for players from {}".format(home_team))
    for player in summary_data:
        for k,v in team_name_json.items():
            if player["team"] in v and \
               (team == k or team in v):
#                print("{} played for {}".format(player["name"], k))
                players.append(player["name"])
    return players

def find_matches_against_team(team, player_list, detail_data):
    """
    loop through player list in the detail_data,
    and find any gameweeks where the
    opponent matches the specified team.
    """
    gameweeks = []
    for player in player_list:
        player_detail = detail_data[player]
        for match in player_detail:

#        for k,v in player_detail.items():
            for tk, tv in team_name_json.items():
                if match["opponent"] in tv and \
                   (team == tk or team in tv):
                    gameweeks.append(int(match['gameweek']))
        if len(gameweeks) == 2:
            break
    gameweeks.sort()
    return gameweeks


if __name__ == "__main__":

    season = sys.argv[-1]
    results_file = open("../data/results_{}.csv".format(season))
    output_file = open("../data/results_{}_with_gw.csv".format(season),"w")
    summary_data = json.load(open("../data/player_summary_{}.json"\
                             .format(season)))
    detail_data = json.load(open("../data/player_details_{}.json"\
                             .format(season)))

    seen_matches = []
    linecount = 0
    for line in results_file.readlines():
        if linecount == 0:
            output_file.write(line.strip()+",gameweek\n")
            linecount += 1
            continue
        date, home_team, away_team = line.split(",")[:3]
        print(date,home_team,away_team)
        home_players =find_players_for_team(home_team,
                                            summary_data)
        gameweeks = find_matches_against_team(away_team,
                                              home_players,
                                              detail_data)
        ## remember the results csv is in reverse time order,
        ## so if we haven't already seen the reverse fixture,
        ## we must be in the second of the two possible gameweeks.
        gameweek = -1
        if len(gameweeks) == 2:
            if (away_team, home_team) in seen_matches:
                print("Already saw {} vs {}".format(away_team, home_team))
                gameweek = gameweeks[0]
            else:
                gameweek = gameweeks[1]
        else:
            print("Some sort of problem, only saw {} gameweeks for {} vs {}"\
                  .format(len(gameweeks), home_team, away_team))
        seen_matches.append((home_team,away_team))
        output_file.write(line.strip()+","+str(gameweek)+"\n")

    output_file.close()
