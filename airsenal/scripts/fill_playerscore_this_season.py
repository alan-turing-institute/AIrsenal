#!/usr/bin/env python

"""
Fill the "player_score" table with the last gameweek's results
"""

import os
import sys



import argparse
import json

from ..framework.mappings import (
    alternative_team_names,
    alternative_player_names,
    positions,
)

from ..framework.data_fetcher import FPLDataFetcher
from ..framework.schema import Player, PlayerScore, Match, Base, engine
from ..framework.utils import get_player_name, get_team_name, get_next_gameweek

from sqlalchemy import create_engine, and_, or_
from sqlalchemy.orm import sessionmaker

DBSession = sessionmaker(bind=engine)
session = DBSession()


def find_match_id(season, gameweek, played_for, opponent):
    """
    query the match table using 3 bits of info...
    not 100% guaranteed, as 'played_for' might be incorrect
    if a player moved partway through the season.  First try
    to match all three bits of info.  If that fails, ignore the played_for.
    That should then work, apart from double-game-weeks where 'opponent'
    will have more than one match per gameweek.
    """

    m = (
        session.query(Match)
        .filter_by(season=season)
        .filter_by(gameweek=gameweek)
        .filter(
            or_(
                and_(Match.home_team == opponent, Match.away_team == played_for),
                and_(Match.away_team == opponent, Match.home_team == played_for),
            )
        )
    )
    if m.first():
        return m.first().match_id
    # now try again without the played_for information (player might have moved)
    m = (
        session.query(Match)
        .filter_by(season=season)
        .filter_by(gameweek=gameweek)
        .filter(or_(Match.home_team == opponent, Match.away_team == opponent))
    )

    if not m.first():
        print(
            "Couldn't find a match between {} and {} in gameweek {}".format(
                played_for, opponent, gameweek
            )
        )
        return None
    return m.first().match_id


def fill_playerscore_table(gw_start, gw_end):
    fetcher = FPLDataFetcher()
    input_data = fetcher.get_player_summary_data()
    season = "1819"
    for player_id in input_data.keys():
        player = get_player_name(player_id)
        # find the player id in the player table.  If they're not
        # there, then we don't care (probably not a current player).
        played_for_id = input_data[player_id]["team"]
        played_for = get_team_name(played_for_id)

        if not played_for:
            print("Cant find team for {}".format(player_id))
            continue

        print("Doing {} for {} season".format(player, season))
        player_data = fetcher.get_gameweek_data_for_player(player_id)
        # now loop through all the matches that player played in
        for gameweek, matches in player_data.items():
            if not gameweek in range(gw_start, gw_end):
                continue
            for match in matches:
                # try to find the match in the match table
                opponent = get_team_name(match["opponent_team"])
                match_id = find_match_id(season, gameweek, played_for, opponent)
                if not match_id:
                    print(
                        "  Couldn't find match for {} in gw {}".format(player, gameweek)
                    )
                    continue
                ps = PlayerScore()
                ps.player_id = player_id
                ps.match_id = match_id
                ps.player_team = played_for
                ps.opponent = opponent
                ps.goals = match["goals_scored"]
                ps.assists = match["assists"]
                ps.bonus = match["bonus"]
                ps.points = match["total_points"]
                ps.conceded = match["goals_conceded"]
                ps.minutes = match["minutes"]
                session.add(ps)
                print(
                    "  got {} points vs {} in gameweek {}".format(
                        match["total_points"], opponent, gameweek
                    )
                )
    session.commit()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="fetch this season's data from FPL API"
    )
    parser.add_argument("--gw_start", help="first gw", type=int, default=1)
    parser.add_argument("--gw_end", help="last gw", type=int)
    args = parser.parse_args()

    if not args.gw_end:
        gw_end = get_next_gameweek()
    else:
        gw_end = args.gw_end
    fill_playerscore_table(args.gw_start, gw_end)
