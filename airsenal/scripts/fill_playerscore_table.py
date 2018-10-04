#!/usr/bin/env python

"""
Fill the "player_score" table with historic results
(player_details_xxyy.json).
"""
import json
import os

from ..framework.mappings import alternative_player_names
from ..framework.schema import Player, PlayerScore, Result, Fixture, session_scope
from ..framework.utils import get_latest_fixture_tag

from sqlalchemy import create_engine, and_, or_


def find_player(player_name, season, session):
    """
    query the player table by name, return the id (or None)
    """
    p = session.query(Player)\
               .filter_by(season=season)\
               .filter_by(name=player_name).first()
    if p:
        return p
    # try alternative names
    for k, v in alternative_player_names.items():
        if player_name in v:
            p = session.query(Player)\
                       .filter_by(season=season)\
                       .filter_by(name=k).first()
            if p:
                return p
    # didn't find it - return None
    return None


def find_fixture(season, gameweek, played_for, opponent, session):
    """
    query the fixture table using 3 bits of info...
    not 100% guaranteed, as 'played_for' might be incorrect
    if a player moved partway through the season.  First try
    to match all three bits of info.  If that fails, ignore the played_for.
    That should then work, apart from double-game-weeks where 'opponent'
    will have more than one match per gameweek.
    """
    tag = get_latest_fixture_tag(season)
    f = (
        session.query(Fixture)
        .filter_by(tag=tag)
        .filter_by(season=season)
        .filter_by(gameweek=gameweek)
        .filter(
            or_(
                and_(Fixture.home_team == opponent, Fixture.away_team == played_for),
                and_(Fixture.away_team == opponent, Fixture.home_team == played_for),
            )
        )
    )
    if f.first():
        return f.first()
    # now try again without the played_for information (player might have moved)
    f = (
        session.query(Fixture)
        .filter_by(tag=tag)
        .filter_by(season=season)
        .filter_by(gameweek=gameweek)
        .filter(or_(Fixture.home_team == opponent, Fixture.away_team == opponent))
    )

    if not f.first():
        print(
            "Couldn't find a fixture between {} and {} in gameweek {}".format(
                played_for, opponent, gameweek
            )
        )
        return None
    return f.first()


def fill_playerscores_from_json(summary_data, detail_data, season, session):
    for player_name in detail_data.keys():
        # find the player id in the player table.  If they're not
        # there, then we don't care (probably not a current player).
        player = find_player(player_name, season, session)
        if not player:
            continue
        # need to find what team the player played for in that season
        played_for = None
        for summary in summary_data:
            if summary["name"] == player_name:
                played_for = summary[
                    "team"
                ]  # WHAT ABOUT PLAYERS THAT MOVED MID-SEASON?
                break
        if not played_for:
            print("Can't find summary data for {}".format(player))
            continue
        print("Doing {} for {} season".format(player_name, season))
        # now loop through all the fixtures that player played in
        for fixture_data in detail_data[player_name]:
            # try to find the result in the result table
            gameweek = fixture_data["gameweek"]
            opponent = fixture_data["opponent"]
            fixture = find_fixture(season, gameweek, played_for, opponent, session)
            if not fixture:
                print(
                    "  Couldn't find result for {} in gw {}".format(player_name, gameweek)
                )
                continue
            ps = PlayerScore()
            ps.player = player
            ps.result = fixture.result
            ps.fixture = fixture
            ps.player_team = played_for
            ps.opponent = opponent
            ps.goals = fixture_data["goals"]
            ps.assists = fixture_data["assists"]
            ps.bonus = fixture_data["bonus"]
            ps.points = fixture_data["points"]
            ps.conceded = fixture_data["conceded"]
            ps.minutes = fixture_data["minutes"]
            session.add(ps)


def make_playerscore_table(session):
    for season in ["1718", "1617"]:#, "1516"]:
        input_path = os.path.join(os.path.dirname(__file__), "../data/player_details_{}.json".format(season))
        summary_path = os.path.join(os.path.dirname(__file__), "../data/player_summary_{}.json".format(season))
        input_data = json.load(open(input_path))
        summary_data = json.load(open(summary_path))
        fill_playerscores_from_json(summary_data, input_data, season, session)

    session.commit()


if __name__ == "__main__":
    with session_scope() as session:
        make_playerscore_table(session)
