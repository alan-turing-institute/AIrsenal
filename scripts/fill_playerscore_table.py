#!/usr/bin/env python

"""
Fill the "player_score" table with historic results
(player_details_xxyy.json).
"""

import os
import sys
sys.path.append("..")

import json

from data.mappings import alternative_team_names, \
    alternative_player_names, positions

from sqlalchemy import create_engine, and_, or_
from sqlalchemy.orm import sessionmaker

from framework.schema import Player, PlayerScore, Match, Base, engine

DBSession = sessionmaker(bind=engine)
session = DBSession()

def find_player_id(player_name):
    """
    query the player table by name, return the id (or None)
    """
    p = session.query(Player).filter_by(name=player).first()
    if p:
        return p.player_id
    # try alternative names
    for k,v in alternative_player_names.items():
        if player_name in v:
            p = session.query(Player).filter_by(name=k).first()
            if p:
                return p.player_id
# didn't find it - return None
    return None



def find_match_id(season, gameweek, played_for, opponent):
    """
    query the match table using 3 bits of info...
    not 100% guaranteed, as 'played_for' might be incorrect
    if a player moved partway through the season.  First try
    to match all three bits of info.  If that fails, ignore the played_for.
    That should then work, apart from double-game-weeks where 'opponent'
    will have more than one match per gameweek.
    """

    m = session.query(Match).filter_by(season=season)\
                            .filter_by(gameweek=gameweek)\
                            .filter(or_ ( and_( Match.home_team==opponent,
                                                Match.away_team==played_for),
                                          and_( Match.away_team==opponent,
                                                Match.home_team==played_for)))
    if m.first():
        return m.first().match_id
    # now try again without the played_for information (player might have moved)
    m = session.query(Match).filter_by(season=season)\
                            .filter_by(gameweek=gameweek)\
                            .filter(or_ ( Match.home_team==opponent,
                                          Match.away_team==opponent))

    if not m.first():
        print("Couldn't find a match between {} and {} in gameweek {}"\
              .format(played_for, opponent, gameweek))
        return None
    return m.first().match_id


if __name__ == "__main__":
    for season in ["1718","1617"]:#,"1617","1516"]:
        input_data = json.load(open("../data/player_details_{}.json".\
                                    format(season)))
        summary_data = json.load(open("../data/player_summary_{}.json".\
                                    format(season)))
        for player in input_data.keys():
            # find the player id in the player table.  If they're not
            # there, then we don't care (probably not a current player).
            player_id = find_player_id(player)
            if not player_id:
                continue
            # need to find what team the player played for in that season
            played_for = None
            for summary in summary_data:
                if summary['name'] == player:
                    played_for = summary["team"] # WHAT ABOUT PLAYERS THAT MOVED MID-SEASON?
                    break
            if not played_for:
                print("Can't find summary data for {}".format(player))
                continue
            print("Doing {} for {} season".format(player,season))
            # now loop through all the matches that player played in
            for match in input_data[player]:
                # try to find the match in the match table
                gameweek = match["gameweek"]
                opponent = match['opponent']
                match_id = find_match_id(season,
                                         gameweek,
                                         played_for,
                                         opponent)
                if not match_id:
                    print("  Couldn't find match for {} in gw {}"\
                          .format(player, gameweek))
                    continue
                ps = PlayerScore()
                ps.player_id = player_id
                ps.match_id = match_id
                ps.player_team = played_for
                ps.opponent = opponent
                ps.goals = match['goals']
                ps.assists = match['assists']
                ps.bonus = match['bonus']
                ps.points = match['points']
                ps.conceded = match['conceded']
                ps.minutes = match['minutes']
                session.add(ps)
    session.commit()
