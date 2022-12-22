#!/usr/bin/env python

"""
Fill the "player_score" table with historic results
(player_details_xxyy.json).
"""
import json
import os
from typing import List, Optional

from sqlalchemy.orm.session import Session

from airsenal.framework.data_fetcher import FPLDataFetcher
from airsenal.framework.schema import PlayerScore, session, session_scope
from airsenal.framework.season import CURRENT_SEASON, sort_seasons
from airsenal.framework.utils import (
    NEXT_GAMEWEEK,
    find_fixture,
    get_past_seasons,
    get_player,
    get_player_from_api_id,
    get_player_scores,
    get_player_team_from_fixture,
    get_team_name,
)


def fill_playerscores_from_json(
    detail_data: list, season: str, dbsession: Session = session
) -> None:
    for player_name in detail_data.keys():
        # find the player id in the player table.  If they're not
        # there, then we don't care (probably not a current player).
        player = get_player(player_name, dbsession=dbsession)
        if not player:
            print(f"Couldn't find player {player_name}")
            continue

        print(f"SCORES {season} {player}")
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
                played_for,
                was_home=was_home,
                other_team=fixture_data["opponent"],
                gameweek=gameweek,
                season=season,
                kickoff_time=fixture_data["kickoff_time"],
                dbsession=dbsession,
            )

            if not fixture or not fixture.result:
                print(f"  Couldn't find result for {player} in gw {gameweek}")
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

            dbsession.add(ps)
    dbsession.commit()


def fill_playerscores_from_api(
    season: str,
    gw_start: int = 1,
    gw_end: int = NEXT_GAMEWEEK,
    dbsession: Session = session,
) -> None:
    fetcher = FPLDataFetcher()
    input_data = fetcher.get_player_summary_data()
    for player_api_id in input_data.keys():
        player = get_player_from_api_id(player_api_id, dbsession=dbsession)
        if not player:
            # If no player found with this API ID something has gone wrong with the
            # Player table, e.g. clashes between players with the same name
            print(f"ERROR! No player with API id {player_api_id}. Skipped.")
            continue

        print(f"SCORES {season} {player}")
        player_data = fetcher.get_gameweek_data_for_player(player_api_id)
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
                    dbsession=dbsession,
                    return_fixture=True,
                )

                if not fixture or not played_for or not fixture.result:
                    print(f"  Couldn't find match result for {player} in gw {gameweek}")
                    continue

                ps = get_player_scores(
                    fixture=fixture, player=player, dbsession=dbsession
                )
                if ps is None:
                    ps = PlayerScore()
                    add = True
                else:
                    add = False
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

                if add:
                    dbsession.add(ps)
                print(
                    f"  got {result['total_points']} points vs {opponent} in gameweek "
                    f"{gameweek}"
                )
    dbsession.commit()


def make_playerscore_table(
    seasons: Optional[List[str]] = [], dbsession: Session = session
) -> None:
    # previous seasons data from json files
    if not seasons:
        seasons = [CURRENT_SEASON]
        seasons += get_past_seasons(3)
    for season in sort_seasons(seasons):
        if season == CURRENT_SEASON:
            # current season - use API
            fill_playerscores_from_api(CURRENT_SEASON, dbsession=dbsession)
        else:
            input_path = os.path.join(
                os.path.dirname(__file__), f"../data/player_details_{season}.json"
            )
            input_data = json.load(open(input_path))
            fill_playerscores_from_json(input_data, season, dbsession=dbsession)


if __name__ == "__main__":
    with session_scope() as session:
        make_playerscore_table(dbsession=session)
