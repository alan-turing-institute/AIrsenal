"""Build the player summary files from the saved FPL season data JSON."""

import json
from typing import Any

from airsenal.core.data_files import data_file
from airsenal.core.logging import get_logger

logger = get_logger(__name__)

INPUT_FILE = str(data_file("FPL_{}.json"))
SAVE_FILE = str(data_file("player_summary_{}.json"))

# dict of {key in input file: key in output file}
keys_to_extract = {
    # name - construct from first name and second name
    "bonus": "bonus",
    "goals_scored": "goals",
    "assists": "assists",
    "minutes": "minutes",
    "penalties_missed": "penalties_missed",
    "penalties_saved": "penalties_saved",
    "clean_sheets": "clean_sheets",
    "total_points": "points",
    "red_cards": "reds",
    "yellow_cards": "yellows",
    "team": "team",  # need to convert index to string
    "element_type": "position",  # need to convert index to string
    "now_cost": "cost",
}


def opta_code(player: dict[str, Any]) -> str:
    """
    The player's opta code, the key that identifies them across seasons.

    The FPL API has sent `opta_code` only since 24/25. For a player it is always
    "p" followed by their `code`, which every season has, so an older season's
    code is built from that.
    """
    return player.get("opta_code") or f"p{player['code']}"


def make_player_summary(season: str) -> None:
    with open(INPUT_FILE.format(season)) as f:
        data = json.load(f)

    teams = {team["id"]: team["short_name"] for team in data["teams"]}
    positions = {et["id"]: et["singular_name_short"] for et in data["element_types"]}

    player_summaries = []

    for player in data["elements"]:
        name = player["first_name"] + " " + player["second_name"]
        logger.debug("%s %s", player["first_name"], player["second_name"])
        player_dict = {"name": name}
        for input_key, output_key in keys_to_extract.items():
            player_dict[output_key] = player[input_key]
        player_dict["opta_code"] = opta_code(player)

        player_dict["team"] = teams[player_dict["team"]]
        player_dict["position"] = positions[player_dict["position"]]

        player_summaries.append(player_dict)

    with open(SAVE_FILE.format(season), "w") as f:
        json.dump(player_summaries, f)
