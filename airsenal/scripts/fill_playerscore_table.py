#!/usr/bin/env python

"""
Fill the "player_score" table with historic results
(player_details_xxyy.json).
"""
import json
import os

from airsenal.framework.data_fetcher import FPLDataFetcher
from airsenal.framework.schema import PlayerScore, session_scope
from airsenal.framework.utils import (
    NEXT_GAMEWEEK,
    get_player,
    get_team_name,
    get_past_seasons,
    CURRENT_SEASON,
    find_fixture,
    get_player_team_from_fixture,
)


def fill_playerscores_from_json(detail_data, season, session):
    for player_name in detail_data.keys():
        # find the player id in the player table.  If they're not
        # there, then we don't care (probably not a current player).
        player = get_player(player_name, dbsession=session)
        if not player:
            print("Couldn't find player {}".format(player_name))
            continue

        print("SCORES {} {}".format(season, player.name))
        # now loop through all the fixtures that player played in
        for fixture_data in detail_data[player_name]:
            # try to find the result in the result table
            gameweek = int(fixture_data["gameweek"])
            if "played_for" in fixture_data.keys():
                played_for = fixture_data["played_for"]
            else:
                played_for = player.team(season, gameweek)
            if not played_for:
                continue

            if fixture_data["was_home"] == "True":
                was_home = True
            elif fixture_data["was_home"] == "False":
                was_home = False
            else:
                was_home = None

            fixture = find_fixture(
                gameweek,
                played_for,
                other_team=fixture_data["opponent"],
                was_home=was_home,
                kickoff_time=fixture_data["kickoff_time"],
                season=season,
                dbsession=session,
            )

            if not fixture:
                print(
                    "  Couldn't find result for {} in gw {}".format(
                        player.name, gameweek
                    )
                )
                continue
            ps = PlayerScore()
            ps.player_team = played_for
            ps.opponent = fixture_data["opponent"]
            ps.goals = fixture_data["goals"]
            ps.assists = fixture_data["assists"]
            ps.bonus = fixture_data["bonus"]
            ps.points = fixture_data["points"]
            ps.conceded = fixture_data["conceded"]
            ps.minutes = fixture_data["minutes"]
            ps.player = player
            ps.result = fixture.result
            ps.fixture = fixture

            # extended features
            # get features excluding the core ones already populated above
            extended_feats = [
                col
                for col in ps.__table__.columns.keys()
                if col
                not in [
                    "id",
                    "player_team",
                    "opponent",
                    "goals",
                    "assists",
                    "bonus",
                    "points",
                    "conceded",
                    "minutes",
                    "player_id",
                    "result_id",
                    "fixture_id",
                ]
            ]
            for feat in extended_feats:
                try:
                    ps.__setattr__(feat, fixture_data[feat])
                except KeyError:
                    pass

            session.add(ps)


def fill_playerscores_from_api(season, session, gw_start=1, gw_end=NEXT_GAMEWEEK):

    fetcher = FPLDataFetcher()
    input_data = fetcher.get_player_summary_data()
    for player_id in input_data.keys():
        # find the player in the player table.  If they're not
        # there, then we don't care (probably not a current player).
        player = get_player(player_id, dbsession=session)
        if not player:
            print("No player with id {}".format(player_id))

        print("SCORES {} {}".format(season, player.name))
        player_data = fetcher.get_gameweek_data_for_player(player.fpl_api_id)
        # now loop through all the matches that player played in
        for gameweek, results in player_data.items():
            if gameweek not in range(gw_start, gw_end):
                continue
            for result in results:
                # try to find the match in the match table
                opponent = get_team_name(result["opponent_team"])

                played_for, fixture = get_player_team_from_fixture(
                    gameweek,
                    opponent,
                    player_at_home=result["was_home"],
                    kickoff_time=result["kickoff_time"],
                    season=season,
                    dbsession=session,
                    return_fixture=True,
                )

                if not fixture or not played_for:
                    print(
                        "  Couldn't find match for {} in gw {}".format(
                            player.name, gameweek
                        )
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

                # extended features
                # get features excluding the core ones already populated above
                extended_feats = [
                    col
                    for col in ps.__table__.columns.keys()
                    if col
                    not in [
                        "id",
                        "player_team",
                        "opponent",
                        "goals",
                        "assists",
                        "bonus",
                        "points",
                        "conceded",
                        "minutes",
                        "player_id",
                        "result_id",
                        "fixture_id",
                    ]
                ]
                for feat in extended_feats:
                    try:
                        ps.__setattr__(feat, result[feat])
                    except KeyError:
                        pass

                session.add(ps)
                print(
                    "  got {} points vs {} in gameweek {}".format(
                        result["total_points"], opponent, gameweek
                    )
                )


def make_playerscore_table(session, seasons=[]):
    # previous seasons data from json files
    if not seasons:
        seasons = get_past_seasons(3)
        seasons.append(CURRENT_SEASON)
    for season in seasons:
        if season == CURRENT_SEASON:
            continue
        input_path = os.path.join(
            os.path.dirname(__file__), "../data/player_details_{}.json".format(season)
        )
        input_data = json.load(open(input_path))
        fill_playerscores_from_json(input_data, season, session)
    # this season's data from the API
    if CURRENT_SEASON in seasons:
        fill_playerscores_from_api(CURRENT_SEASON, session)

    session.commit()


if __name__ == "__main__":
    with session_scope() as session:
        make_playerscore_table(session)
