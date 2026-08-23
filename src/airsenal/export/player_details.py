"""
Save all player gameweek scores for the current season from the API
"""

import json
from typing import Any

from airsenal.core.console import track
from airsenal.core.data_files import data_file
from airsenal.core.season import CURRENT_SEASON
from airsenal.fetch.fpl_api import FPLDataFetcher

RENAME_KEYS = {
    "round": "gameweek",
    "total_points": "points",
    "goals_scored": "goals",
    "goals_conceded": "conceded",
    "opponent_team": "opponent",
}
REMOVE_KEYS = [
    "element",
    "fixture",
    "modified",
    "team_h_score",
    "team_a_score",
    "was_home",
]
SAVE_FILE = str(data_file("player_details_{}.json"))


def make_player_details(season: str = CURRENT_SEASON) -> None:
    if season != CURRENT_SEASON:
        msg = "This script is only designed to work for the current season"
        raise ValueError(msg)

    fetcher = FPLDataFetcher()
    player_summary_data = fetcher.get_player_summary_data()
    gameweeks = fetcher.get_event_data().keys()
    team_id_to_name = get_team_mapping(fetcher)
    fixture_teams = get_fixture_teams(fetcher)

    player_details: dict[str, list[dict[str, Any]]] = {}
    for player_id, player_meta in track(player_summary_data.items()):
        player_details[player_meta["opta_code"]] = []
        for gw in gameweeks:
            gw_details = fetcher.get_gameweek_data_for_player(player_id, gw)
            for result in gw_details:
                played_for = get_played_for(result, fixture_teams)
                result["played_for"] = team_id_to_name[played_for]
                result["opponent_team"] = team_id_to_name[result["opponent_team"]]
                for old_key, new_key in RENAME_KEYS.items():
                    result[new_key] = result.pop(old_key)
                for key in REMOVE_KEYS:
                    result.pop(key)
                player_details[player_meta["opta_code"]].append(result)

    with open(SAVE_FILE.format(season), "w") as f:
        json.dump(player_details, f)


def get_team_mapping(fetcher: FPLDataFetcher) -> dict[int, str]:
    team_mapping = {}
    for team_data in fetcher.get_current_team_data().values():
        team_mapping[team_data["id"]] = team_data["short_name"]
    return team_mapping


def get_fixture_teams(fetcher: FPLDataFetcher) -> dict[int, dict[str, int]]:
    fixture_teams = {}
    for fixture in fetcher.get_fixture_data():
        fixture_teams[fixture["id"]] = {
            "team_h": fixture["team_h"],
            "team_a": fixture["team_a"],
        }
    return fixture_teams


def get_played_for(
    player_result: dict[str, Any], fixture_teams: dict[int, dict[str, int]]
) -> int:
    fixture = fixture_teams[player_result["fixture"]]
    if fixture["team_a"] == player_result["opponent_team"]:
        return fixture["team_h"]
    if fixture["team_h"] == player_result["opponent_team"]:
        return fixture["team_a"]
    msg = f"Opponent {player_result['opponent_team']} not found in fixture {fixture}"
    raise KeyError(msg)


if __name__ == "__main__":
    make_player_details()
