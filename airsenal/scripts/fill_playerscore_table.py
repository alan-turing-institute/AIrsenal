#!/usr/bin/env python

"""
Fill the "player_score" table with historic results
(player_details_xxyy.json).
"""
import json
import os

from sqlalchemy import create_engine, and_, or_

from ..framework.mappings import alternative_player_names
from ..framework.schema import Player, PlayerScore, Result, Fixture, session_scope
from ..framework.utils import get_latest_fixture_tag, get_next_gameweek, get_player, get_team_name
from ..framework.data_fetcher import FPLDataFetcher


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


def fill_playerscores_from_json(detail_data, season, session):
    for player_name in detail_data.keys():
        # find the player id in the player table.  If they're not
        # there, then we don't care (probably not a current player).
        player = get_player(player_name, dbsession=session)
        if not player:
            continue

        print("Doing {} for {} season".format(player_name, season))
        # now loop through all the fixtures that player played in
        for fixture_data in detail_data[player_name]:
            # try to find the result in the result table
            gameweek = int(fixture_data["gameweek"])
            played_for = player.team(season, gameweek)
            opponent = fixture_data["opponent"]
            fixture = find_fixture(season, gameweek, played_for, opponent, session)
            if not fixture:
                print(
                    "  Couldn't find result for {} in gw {}".format(player_name, gameweek)
                )
                continue
            ps = PlayerScore()
            ps.player_team = played_for
            ps.opponent = opponent
            ps.goals = fixture_data["goals"]
            ps.assists = fixture_data["assists"]
            ps.bonus = fixture_data["bonus"]
            ps.points = fixture_data["points"]
            ps.conceded = fixture_data["conceded"]
            ps.minutes = fixture_data["minutes"]
            ps.player = player
            ps.result = fixture.result
            ps.fixture = fixture
            player.scores.append(ps)
         #   session.add(ps)

def fill_playerscores_from_api(season, session, gw_start=1):
    gw_end = get_next_gameweek()
    fetcher = FPLDataFetcher()
    input_data = fetcher.get_player_summary_data()
    for player_id in input_data.keys():
        player = get_player(player_id, dbsession=session)
        # find the player in the player table.  If they're not
        # there, then we don't care (probably not a current player).
        played_for_id = input_data[player_id]["team"]
        played_for = get_team_name(played_for_id)

        if not played_for:
            print("Cant find team for {}".format(player_id))
            continue

        print("Doing {} for {} season".format(player.name, season))
        player_data = fetcher.get_gameweek_data_for_player(player_id)
        # now loop through all the matches that player played in
        for gameweek, results in player_data.items():
            if not gameweek in range(gw_start, gw_end):
                continue
            for result in results:
                # try to find the match in the match table
                opponent = get_team_name(result["opponent_team"])
                fixture = find_fixture(season, gameweek, played_for, opponent, session)
                if not fixture:
                    print(
                        "  Couldn't find match for {} in gw {}".format(player.name, gameweek)
                    )
                    continue
                ps = PlayerScore()
                ps.player_team = played_for
                ps.opponent = opponent
                ps.goals = result["goals_scored"]
                ps.assists = result["assists"]
                ps.bonus = result["bonus"]
                ps.points = result["total_points"]
                ps.conceded = result["goals_conceded"]
                ps.minutes = result["minutes"]
                ps.player = player
                ps.fixture = fixture
                ps.result = fixture.result
                player.scores.append(ps)
                session.add(ps)
                print(
                    "  got {} points vs {} in gameweek {}".format(
                        result["total_points"], opponent, gameweek
                    )
                )

def make_playerscore_table(session):
    # previous seasons data from json files
    for season in ["1718", "1617"]:#, "1516"]:
        input_path = os.path.join(os.path.dirname(__file__), "../data/player_details_{}.json".format(season))
        input_data = json.load(open(input_path))
        fill_playerscores_from_json(input_data, season, session)
    # this season's data from the API
    fill_playerscores_from_api("1819",session)

    session.commit()


if __name__ == "__main__":
    with session_scope() as session:
        make_playerscore_table(session)
