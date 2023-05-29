"""
Make player summary files from FPL season data json
"""
import json
import os

from airsenal.framework.utils import get_past_seasons

SCRIPT_DIR = os.path.dirname(__file__)
INPUT_FILE = os.path.join(SCRIPT_DIR, "../data/FPL_{}.json")
SAVE_FILE = os.path.join(SCRIPT_DIR, "../data/player_summary_{}.json")

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


def make_player_summary(season: str) -> None:
    with open(INPUT_FILE.format(season), "r") as f:
        data = json.load(f)

    teams = {team["id"]: team["short_name"] for team in data["teams"]}
    positions = {et["id"]: et["singular_name_short"] for et in data["element_types"]}

    player_summaries = []

    for player in data["elements"]:
        name = player["first_name"] + " " + player["second_name"]
        print(player["first_name"] + " " + player["second_name"])
        player_dict = {"name": name}
        for input_key, output_key in keys_to_extract.items():
            player_dict[output_key] = player[input_key]

        player_dict["team"] = teams[player_dict["team"]]
        player_dict["position"] = positions[player_dict["position"]]

        player_summaries.append(player_dict)

    with open(SAVE_FILE.format(season), "w") as f:
        json.dump(player_summaries, f)


if __name__ == "__main__":
    for season in get_past_seasons(3):
        print(f"---- MAKING PLAYER SUMMARIES FOR {season} SEASON ----")
        make_player_summary(season)
    print("---- DONE ----")
