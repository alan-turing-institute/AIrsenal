#!/usr/bin/env python

"""
Fill the "Player" table with info from this and past seasonss FPL
"""
import os
import sys
import json

from ..framework.mappings import alternative_player_names

from ..framework.schema import (Player,
                                PlayerAttributes,
                                Result,
                                Fixture,
                                session_scope)

from ..framework.utils import (get_latest_fixture_tag,
                               get_next_gameweek,
                               get_player,
                               get_team_name,
                               get_past_seasons,
                               CURRENT_SEASON,
                               get_player_attributes)

from ..framework.data_fetcher import FPLDataFetcher


def fill_attributes_table_from_file(detail_data, season, session):
    """Fill player attributes table for previous season using data from
    player detail JSON files.
    """
       
    for player_name in detail_data.keys():
        # find the player id in the player table.  If they're not
        # there, then we don't care (probably not a current player).
        player = get_player(player_name, dbsession=session)
        if not player:
            print("Couldn't find player {}".format(player_name))
            continue

        print("ATTRIBUTES {} {}".format(season, player.name))
        # now loop through all the fixtures that player played in
        # Only one attributes row per gameweek - create list of gameweeks
        # encountered so can ignore duplicates (e.g. from double gameweeks).
        previous_gameweeks = []
        for fixture_data in detail_data[player_name]:
            gameweek = int(fixture_data["gameweek"])
            if gameweek in previous_gameweeks:
                # already done this gameweek
                continue
            else:
                previous_gameweeks.append(gameweek)
                
                pa = PlayerAttributes()
                pa.player = player
                pa.player_id = player.player_id
                pa.season = season
                pa.gameweek = gameweek
                pa.price = int(fixture_data["value"])
                pa.team = fixture_data["played_for"]
                pa.position = fixture_data["position"]

                player.attributes.append(pa)
                session.add(pa)


def fill_attributes_table_from_api(season, session):
    """
    use the FPL API
    """
    raise NotImplementedError()


def fill_missing_attributes(start_season, start_gameweek,
                            end_season, end_gameweek):
    """Player details files only contain info when a player had a fixture, so
    has gaps due to blank gameweeks. Fill them with the most recent available
    information before the blank.
    """
    raise NotImplementedError()


def make_attributes_table(session):
    """Create the player attributes table using the previous 3 seasons (from 
    player details JSON files) and the current season (from API)
    """ 
    seasons = get_past_seasons(3)
    
    for season in seasons:
        input_path = os.path.join(os.path.dirname(__file__),
                                  "../data/player_details_{}.json".format(season))        
        with open(input_path, "r") as f:
            input_data = json.load(f)
        
        fill_attributes_table_from_file(input_data, season, session)
    
    # this season's data from the API
    fill_attributes_table_from_api(CURRENT_SEASON, session)

    session.commit()
    
    fill_missing_attributes(start_season=seasons[-1],
                            start_gameweek=1,
                            end_season=CURRENT_SEASON,
                            end_gameweek=get_next_gameweek())


if __name__ == "__main__":
    with session_scope() as session:
        make_attributes_table(session)
