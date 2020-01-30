#!/usr/bin/env python

"""
Fill the "Player" table with info from this and past seasonss FPL
"""
import os
import sys
import json

from ..framework.mappings import alternative_player_names, positions

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
                               get_player_attributes,
                               get_player_team_from_fixture)

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
                pa.transfers_balance = int(fixture_data["transfers_balance"])
                pa.selected = int(fixture_data["selected"])
                pa.transfers_in = int(fixture_data["transfers_in"])
                pa.transfers_out = int(fixture_data["transfers_out"])
                session.add(pa)


def fill_attributes_table_from_api(season, session, gw_start=1, gw_end=None):
    """
    use the FPL API to get player attributes info for the current season
    """
    next_gw = get_next_gameweek(season, session)
    if not gw_end:
        gw_end = next_gw
    
    fetcher = FPLDataFetcher()
    
    # needed for selected by calculation from percentage below
    n_players = fetcher.get_current_summary_data()["total_players"]
    
    input_data = fetcher.get_player_summary_data()

    for player_id in input_data.keys():
        # find the player in the player table
        player = get_player(player_id, dbsession=session)
        print("ATTRIBUTES {} {}".format(season, player.name))
                
        # First update the current gameweek using the summary data
        p_summary = input_data[player_id]
        position = positions[p_summary["element_type"]]
        
        pa = get_player_attributes(player.player_id,
                                   season=season,
                                   gameweek=next_gw,
                                   dbsession=session)
        if pa:
            # found pre-existing attributes for this gameweek
            update = True
        else:
            # no attributes for this gameweek for this player yet
            pa = PlayerAttributes()
            update = False
        
        pa.player = player
        pa.player_id = player.player_id
        pa.season = season
        pa.gameweek = next_gw
        pa.price = int(p_summary["now_cost"])
        pa.team = get_team_name(p_summary["team"],
                                season=season,
                                dbsession=session)
        pa.position = positions[p_summary["element_type"]]
        pa.selected = int(float(p_summary["selected_by_percent"])
                          * n_players / 100)
        pa.transfers_in = int(p_summary["transfers_in_event"])
        pa.transfers_out = int(p_summary["transfers_out_event"])
        pa.transfers_balance = pa.transfers_in - pa.transfers_out
        
        if not update:
            # only need to add to the session for new entries, if we're doing
            # an update the final session.commit() is enough
            session.add(pa)
               
        # now get data for previous gameweeks
        player_data = fetcher.get_gameweek_data_for_player(player_id)
        for gameweek, data in player_data.items():
            if gameweek not in range(gw_start, gw_end):
                continue
            
            for result in data:
                # check whether there are pre-existing attributes to update
                pa = get_player_attributes(player.player_id,
                                           season=season,
                                           gameweek=next_gw,
                                           dbsession=session)
                if pa:
                    update = True
                else:
                    pa = PlayerAttributes()
                    update = False
                
                # determine the team the player played for in this fixture
                opponent_id = result["opponent_team"]
                was_home = result["was_home"]
                kickoff_time = result["kickoff_time"]
                team = get_player_team_from_fixture(gameweek,
                                                    opponent_id,
                                                    was_home,
                                                    kickoff_time,
                                                    season=season,
                                                    dbsession=session)
                
                pa.player = player
                pa.player_id = player.player_id
                pa.season = season
                pa.gameweek = gameweek
                pa.price = int(result["value"])
                pa.team = team
                pa.position = position  # does not change during season
                pa.transfers_balance = int(result["transfers_balance"])
                pa.selected = int(result["selected"])
                pa.transfers_in = int(result["transfers_in"])
                pa.transfers_out = int(result["transfers_out"])
                
                if update:
                    # don't need to add to session if updating pre-existing row
                    session.add(pa)
                               
                break  # done this gameweek now


def fill_missing_attributes(start_season, start_gameweek,
                            end_season, end_gameweek):
    """Player details files only contain info when a player had a fixture, so
    has gaps due to blank gameweeks. Fill them with the most recent available
    information before the blank.
    """
    #raise NotImplementedError()


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
