"""
Fill the "Player" table with info from this and past seasonss FPL
"""

import json
import os

from sqlalchemy.orm.session import Session

from airsenal.framework.data_fetcher import FPLDataFetcher
from airsenal.framework.mappings import positions
from airsenal.framework.schema import PlayerAttributes, session, session_scope
from airsenal.framework.season import CURRENT_SEASON, sort_seasons
from airsenal.framework.utils import (
    find_fixture,
    get_next_gameweek,
    get_past_seasons,
    get_player,
    get_player_attributes,
    get_player_from_api_id,
    get_player_team_from_fixture,
    get_return_gameweek_from_news,
    get_team_name,
)


def fill_attributes_table_from_file(
    detail_data: dict, season: str, dbsession: Session = session
) -> None:
    """Fill player attributes table for previous season using data from
    player detail JSON files.
    """

    for player_name, player_data in detail_data.items():
        # find the player id in the player table.  If they're not
        # there, then we don't care (probably not a current player).
        player = get_player(player_name, dbsession=dbsession)
        if not player:
            print(f"Couldn't find player {player_name}")
            continue

        print(f"ATTRIBUTES {season} {player}")
        # now loop through all the fixtures that player played in
        # Only one attributes row per gameweek - create list of gameweeks
        # encountered so can ignore duplicates (e.g. from double gameweeks).
        previous_gameweeks = []
        for fixture_data in player_data:
            gameweek = int(fixture_data["gameweek"])
            if gameweek in previous_gameweeks:
                # already done this gameweek
                continue
            previous_gameweeks.append(gameweek)

            pa = PlayerAttributes()
            pa.player = player
            pa.player_id = player.player_id
            pa.season = season
            pa.gameweek = gameweek
            # obtain attributes if available, otherwise set to None or default value
            pa.price = int(fixture_data.get("value", 0))
            pa.team = fixture_data.get("played_for", "")
            pa.position = fixture_data.get("position", "")
            pa.transfers_balance = (
                int(fixture_data.get("transfers_balance"))
                if fixture_data.get("transfers_balance") is not None
                else None
            )
            pa.selected = (
                int(fixture_data.get("selected"))
                if fixture_data.get("selected") is not None
                else None
            )
            pa.transfers_in = (
                int(fixture_data.get("transfers_in"))
                if fixture_data.get("transfers_in") is not None
                else None
            )
            pa.transfers_out = (
                int(fixture_data.get("transfers_out"))
                if fixture_data.get("transfers_out") is not None
                else None
            )
            dbsession.add(pa)
    dbsession.commit()


def fill_attributes_table_from_api(
    season: str, gw_start: int = 1, dbsession: Session = session
) -> None:
    """
    use the FPL API to get player attributes info for the current season
    """
    fetcher = FPLDataFetcher()
    next_gw = get_next_gameweek(season=season, dbsession=dbsession)

    # needed for selected by calculation from percentage below
    n_players = fetcher.get_current_summary_data()["total_players"]

    input_data = fetcher.get_player_summary_data()

    for player_api_id in input_data:
        # find the player in the player table
        player = get_player_from_api_id(player_api_id, dbsession=dbsession)
        if not player:
            print(f"ATTRIBUTES {season} No player found with id {player_api_id}")
            continue

        print(f"ATTRIBUTES {season} {player}")

        # First update the current gameweek using the summary data
        p_summary = input_data[player_api_id]

        if player.opta_code is None and "opta_code" in p_summary:
            player.opta_code = p_summary["opta_code"]

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
        team = get_team_name(p_summary["team"], season=season, dbsession=dbsession)
        if team is None:
            print(f"Couldn't find team {p_summary['team']} for player {player}")
            continue
        pa.team = team
        pa.position = positions[p_summary["element_type"]]
        pa.selected = int(float(p_summary["selected_by_percent"]) * n_players / 100)
        transfers_in = int(p_summary["transfers_in"])
        transfers_out = int(p_summary["transfers_out"])
        pa.transfers_in = transfers_in
        pa.transfers_out = transfers_out
        pa.transfers_balance = transfers_in - transfers_out
        pa.news = p_summary["news"]
        chance_of_playing_next_round = p_summary["chance_of_playing_next_round"]
        pa.chance_of_playing_next_round = chance_of_playing_next_round
        if (
            chance_of_playing_next_round is not None
            and chance_of_playing_next_round <= 50
        ):
            pa.return_gameweek = get_return_gameweek_from_news(
                p_summary["news"],
                team=team,
                season=season,
                dbsession=dbsession,
            )

        if not update:
            # only need to add to the dbsession for new entries, if we're doing
            # an update the final dbsession.commit() is enough
            dbsession.add(pa)

        # now get data for previous gameweeks
        if next_gw > 1:
            player_data = fetcher.get_gameweek_data_for_player(player_api_id)
            if not player_data:
                print(f"Failed to get data for {player}")
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
                    fixture = find_fixture(
                        opponent_id,
                        was_home=not was_home,
                        gameweek=gameweek,
                        season=season,
                        kickoff_time=kickoff_time,
                        dbsession=dbsession,
                    )
                    if fixture is None:
                        print(
                            f"Couldn't find fixture for {player} vs {opponent_id} in "
                            f"gameweek {gameweek}"
                        )
                        continue
                    team = get_player_team_from_fixture(
                        fixture,
                        opponent_id,
                        player_at_home=was_home,
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
    dbsession.commit()


def make_attributes_table(
    seasons: list[str] | None = None, dbsession: Session = session
) -> None:
    """Create the player attributes table using the previous 3 seasons (from
    player details JSON files) and the current season (from API)
    """
    if seasons is None:
        seasons = []
    if not seasons:
        seasons = [CURRENT_SEASON]
        seasons += get_past_seasons(3)
    for season in sort_seasons(seasons):
        if season == CURRENT_SEASON:
            # current season - use API
            fill_attributes_table_from_api(season=CURRENT_SEASON, dbsession=dbsession)
        else:
            input_path = os.path.join(
                os.path.dirname(__file__), f"../data/player_details_{season}.json"
            )
            with open(input_path) as f:
                input_data = json.load(f)

            fill_attributes_table_from_file(
                detail_data=input_data, season=season, dbsession=dbsession
            )
    dbsession.commit()


if __name__ == "__main__":
    with session_scope() as dbsession:
        make_attributes_table(dbsession=dbsession)
