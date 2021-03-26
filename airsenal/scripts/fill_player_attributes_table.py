#!/usr/bin/env python

"""
Fill the "Player" table with info from this and past seasonss FPL
"""
import os

import json

from airsenal.framework.mappings import positions
from airsenal.framework.schema import PlayerAttributes, session_scope, session

from airsenal.framework.utils import (
    get_next_gameweek,
    get_player,
    get_player_from_api_id,
    get_team_name,
    get_past_seasons,
    CURRENT_SEASON,
    get_player_attributes,
    get_player_team_from_fixture,
    get_return_gameweek_from_news,
)

from airsenal.framework.data_fetcher import FPLDataFetcher


def fill_attributes_table_from_file(detail_data, season, dbsession=session):
    """Fill player attributes table for previous season using data from
    player detail JSON files.
    """

    for player_name in detail_data.keys():
        # find the player id in the player table.  If they're not
        # there, then we don't care (probably not a current player).
        player = get_player(player_name, dbsession=dbsession)
        if not player:
            print("Couldn't find player {}".format(player_name))
            continue

        print("ATTRIBUTES {} {}".format(season, player))
        # now loop through all the fixtures that player played in
        #  Only one attributes row per gameweek - create list of gameweeks
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
                dbsession.add(pa)


def fill_attributes_table_from_api(season, gw_start=1, dbsession=session):
    """
    use the FPL API to get player attributes info for the current season
    """
    fetcher = FPLDataFetcher()
    next_gw = get_next_gameweek(season=season, dbsession=dbsession)

    # needed for selected by calculation from percentage below
    n_players = fetcher.get_current_summary_data()["total_players"]

    input_data = fetcher.get_player_summary_data()

    for player_api_id in input_data.keys():
        # find the player in the player table
        player = get_player_from_api_id(player_api_id, dbsession=dbsession)
        if not player:
            print(
                "ATTRIBUTES {} No player found with id {}".format(season, player_api_id)
            )
            continue

        print("ATTRIBUTES {} {}".format(season, player.name))

        # First update the current gameweek using the summary data
        p_summary = input_data[player_api_id]
        position = positions[p_summary["element_type"]]

        pa = get_player_attributes(
            player.player_id, season=season, gameweek=next_gw, dbsession=dbsession
        )
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
        pa.team = get_team_name(p_summary["team"], season=season, dbsession=dbsession)
        pa.position = positions[p_summary["element_type"]]
        pa.selected = int(float(p_summary["selected_by_percent"]) * n_players / 100)
        pa.transfers_in = int(p_summary["transfers_in_event"])
        pa.transfers_out = int(p_summary["transfers_out_event"])
        pa.transfers_balance = pa.transfers_in - pa.transfers_out
        pa.chance_of_playing_next_round = p_summary["chance_of_playing_next_round"]
        pa.news = p_summary["news"]
        if (
            pa.chance_of_playing_next_round is not None
            and pa.chance_of_playing_next_round <= 50
        ):
            pa.return_gameweek = get_return_gameweek_from_news(
                p_summary["news"],
                season=season,
                dbsession=dbsession,
            )

        if not update:
            # only need to add to the dbsession for new entries, if we're doing
            #  an update the final dbsession.commit() is enough
            dbsession.add(pa)

        # now get data for previous gameweeks
        player_data = fetcher.get_gameweek_data_for_player(player_api_id)
        if not player_data:
            print("Failed to get data for", player.name)
            continue
        for gameweek, data in player_data.items():
            if gameweek < gw_start:
                continue

            for result in data:
                # check whether there are pre-existing attributes to update
                pa = get_player_attributes(
                    player.player_id,
                    season=season,
                    gameweek=gameweek,
                    dbsession=dbsession,
                )
                if pa:
                    update = True
                else:
                    pa = PlayerAttributes()
                    update = False

                # determine the team the player played for in this fixture
                opponent_id = result["opponent_team"]
                was_home = result["was_home"]
                kickoff_time = result["kickoff_time"]
                team = get_player_team_from_fixture(
                    gameweek,
                    opponent_id,
                    was_home,
                    kickoff_time,
                    season=season,
                    dbsession=dbsession,
                )

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

                if not update:
                    # don't need to add to dbsession if updating pre-existing row
                    dbsession.add(pa)

                break  # done this gameweek now


def make_attributes_table(seasons=[], dbsession=session):
    """Create the player attributes table using the previous 3 seasons (from
    player details JSON files) and the current season (from API)
    """
    if not seasons:
        seasons = get_past_seasons(3)
        seasons.append(CURRENT_SEASON)

    for season in seasons:
        if season == CURRENT_SEASON:
            continue
        input_path = os.path.join(
            os.path.dirname(__file__), "../data/player_details_{}.json".format(season)
        )
        with open(input_path, "r") as f:
            input_data = json.load(f)

        fill_attributes_table_from_file(
            detail_data=input_data, season=season, dbsession=dbsession
        )

    # this season's data from the API
    if CURRENT_SEASON in seasons:
        fill_attributes_table_from_api(season=CURRENT_SEASON, dbsession=dbsession)

    dbsession.commit()


if __name__ == "__main__":
    with session_scope() as dbsession:
        make_attributes_table(dbsession=dbsession)
