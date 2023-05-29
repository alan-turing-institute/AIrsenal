"""
Generate player details json files from a locally cloned copy of the
https://github.com/vaastav/Fantasy-Premier-League repository on GitHub.

"""
import json
import os
import shutil
import subprocess
from functools import cache
from glob import glob
from typing import List, Optional, Tuple, Union

import pandas as pd

from airsenal.framework.mappings import alternative_team_names, positions
from airsenal.framework.utils import get_past_seasons

# directory of this script
SCRIPT_DIR = os.path.dirname(__file__)

# ------------------------------------
# vaastav/Fantasy-Premier-League Files
# ------------------------------------
# players directory for season of interest from this git repo:
# https://github.com/vaastav/Fantasy-Premier-League
# repo is assumed to be cloned locally in same directory as parent AIrsenal
# directory.
# {} will be formatted with season of interest
GIT_REPO = "https://github.com/vaastav/Fantasy-Premier-League.git"
REPO_DIR = os.path.join(SCRIPT_DIR, "Fantasy-Premier-League")
DATA_DIR = os.path.join(REPO_DIR, "data", "{}")
# Path to directory of player data
PLAYERS_DIR = os.path.join(DATA_DIR, "players")
# file containing GW data in every sub_directory in PLAYERS_DIR
PLAYERS_FILE = "gw.csv"
# Path to fixtures files
FIXTURES_PATH = os.path.join(DATA_DIR, "fixtures.csv")
# Path to raw player summary data
RAW_PATH = os.path.join(DATA_DIR, "players_raw.csv")

# ------------------------------------
# AIrsenal Files
# ------------------------------------
# teams path - to get mapping of ids to team names
# {} will be formatted with season of interest
TEAM_PATH = os.path.join(SCRIPT_DIR, "../data/teams_{}.csv")
# results path - used if FIXTURES_PATH not available
# {} will be formatted with season of interest
RESULTS_PATH = os.path.join(SCRIPT_DIR, "../data/results_{}_with_gw.csv")
# where to save output
# {} will be formatted with season of interest
SAVE_NAME = os.path.join(SCRIPT_DIR, "../data/player_details_{}.json")
# duplicate names file - used to rename certain players to disambiguate multiple
# players with the same name
DUPLICATE_PATH = os.path.join(SCRIPT_DIR, "../data/duplicate_player_names.csv")

#  Dictionary of features to extract {name in files: name in database}
key_dict = {
    "round": "gameweek",
    "total_points": "points",
    "goals_scored": "goals",
    "assists": "assists",
    "goals_conceded": "conceded",
    "bonus": "bonus",
    "minutes": "minutes",
    "opponent_team": "opponent",  # id in input, 3 letters in output!!!
    # extended features
    "clean_sheets": "clean_sheets",
    "own_goals": "own_goals",
    "penalties_saved": "penalties_saved",
    "penalties_missed": "penalties_missed",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "saves": "saves",
    "bps": "bps",
    "influence": "influence",
    "creativity": "creativity",
    "threat": "threat",
    "ict_index": "ict_index",
    "transfers_balance": "transfers_balance",
    "selected": "selected",
    "transfers_in": "transfers_in",
    "transfers_out": "transfers_out",
    "expected_goals": "expected_goals",
    "expected_assists": "expected_assists",
    "expected_goal_involvements": "expected_goal_involvements",
    "expected_goals_conceded": "expected_goals_conceded",
    # attributes
    "value": "value",
    "played_for": "played_for",
    # needed to determine fixture if two teams play each other twice same gw
    "kickoff_time": "kickoff_time",
    "was_home": "was_home",
}


def path_to_name(path: str) -> str:
    """function to take a sub directory path into a key for output json
    i.e. player name from directory path
    """
    # get directory name from full path
    dir_name = os.path.basename(os.path.dirname(path))

    return " ".join(x for x in dir_name.split("_") if not x.isdigit())


def path_to_index(path: str) -> Optional[int]:
    """function to take a sub directory path into a key for output json
    i.e. player name from directory path
    """
    # get directory name from full path
    dir_name = os.path.basename(os.path.dirname(path))
    try:
        return int(" ".join(x for x in dir_name.split("_") if x.isdigit()))
    except ValueError:
        return None


def get_long_season_name(short_name: str) -> str:
    """Convert short season name of format 1718 to long name like 2017-18."""
    return "20" + short_name[:2] + "-" + short_name[2:]


def get_teams_dict(season: str) -> dict:
    teams_df = pd.read_csv(TEAM_PATH.format(season))
    return {row["team_id"]: row["name"] for _, row in teams_df.iterrows()}


def get_positions_df(season: str) -> pd.DataFrame:
    """
    Get dataframe of player names and their positions for the given season,
    using the players_raw file from the FPL results repo.
    """
    season_longname = get_long_season_name(season)
    raw_df = pd.read_csv(RAW_PATH.format(season_longname))

    raw_df["name"] = raw_df["first_name"] + " " + raw_df["second_name"]

    # replace element type codes with position strings
    raw_df["position"] = raw_df["element_type"].replace(positions)

    raw_df.set_index("id", inplace=True)
    return raw_df


def get_fixtures_df(season: str) -> Tuple[pd.DataFrame, bool]:
    """Load fixture info (which teams played in which matches), either
    from vaastav/Fantasy-Premier-League repo or AIrsenal data depending
    on what's available.
    """
    season_longname = get_long_season_name(season)

    if os.path.exists(FIXTURES_PATH.format(season_longname)):
        #  fixtures file in vaastav/Fantasy-Premier-League repo
        # contains fixture ids
        fixtures_df = pd.read_csv(FIXTURES_PATH.format(season_longname), index_col="id")
        got_fixtures = True
    elif os.path.exists(RESULTS_PATH.format(season)):
        #  match results files in airsenal data
        # need to match teams by gameweek etc.
        fixtures_df = pd.read_csv(RESULTS_PATH.format(season))

        #  replace full team names with 3 letter codes
        for short_name, long_names in alternative_team_names.items():
            replace_dict = {name: short_name for name in long_names}
            fixtures_df["home_team"].replace(replace_dict, inplace=True)
            fixtures_df["away_team"].replace(replace_dict, inplace=True)

        got_fixtures = False
    else:
        raise FileNotFoundError(f"Couldn't find fixtures file for {season} season")

    return fixtures_df, got_fixtures


def get_played_for_from_fixtures(
    fixture_id: int, opponent_id: int, was_home: bool, fixtures_df: pd.DataFrame
) -> str:
    """Get the team a player played for given the id of a fixture
    and the id of the opposing team.
    """
    fixture = fixtures_df.loc[fixture_id]

    if (not was_home) and (fixture["team_h"] == opponent_id):
        return fixture["team_a"]
    elif was_home and (fixture["team_a"] == opponent_id):
        return fixture["team_h"]
    else:
        raise ValueError(
            f"""Error finding team played for with fixture id {fixture_id},
                         opponent_id {opponent_id} and was_home {was_home}"""
        )


def get_played_for_from_results(player_row, results_df, teams_dict):
    """
    Find what team played for given the gameweek, match date, opposing team,
    and whether the player was at home or not.
    """
    opponent = teams_dict[player_row["opponent_team"]]
    gw = player_row["round"]
    was_home = player_row["was_home"]

    if was_home:
        matches = results_df[
            (results_df["away_team"] == opponent) & (results_df["gameweek"] == gw)
        ]
    else:
        matches = results_df[
            (results_df["home_team"] == opponent) & (results_df["gameweek"] == gw)
        ]

    if len(matches) > 1:
        # Opponent appeared in more than one match in this gameweek, so filter
        # matches further based on date.
        matches["date"] = pd.to_datetime(matches["date"]).dt.date
        player_date = pd.to_datetime(player_row["kickoff_time"]).date()
        matches = matches[matches["date"] == player_date]

    if len(matches) != 1:
        # Couldn't find a unique fixture
        raise ValueError(
            f"""Found no matches with gw {gw}, was_home {was_home}
                                and opponent {opponent}"""
        )

    # Found a unique fixture corresponding to the input data.
    if was_home:
        return matches["home_team"].iloc[0]
    else:
        return matches["away_team"].iloc[0]


def process_file(path, teams_dict, fixtures_df, got_fixtures):
    """function to load and process one of the player score files"""
    # load input file
    df = pd.read_csv(path)

    if got_fixtures:
        df["played_for"] = [
            get_played_for_from_fixtures(
                row["fixture"], row["opponent_team"], row["was_home"], fixtures_df
            )
            for _, row in df.iterrows()
        ]
    else:
        df["played_for"] = [
            get_played_for_from_results(row, fixtures_df, teams_dict)
            for _, row in df.iterrows()
        ]

    # extract columns of interest
    df = df[key_dict.keys()]

    # rename columns to desired output names
    df.rename(columns=key_dict, inplace=True)

    # rename team ids with short names
    df["opponent"].replace(teams_dict, inplace=True)
    df["played_for"].replace(teams_dict, inplace=True)

    # want everything in output to be strings
    df = df.applymap(str)

    # return json like dictionary
    return df.to_dict(orient="records")


@cache
def get_duplicates_df() -> pd.DataFrame:
    return pd.read_csv(DUPLICATE_PATH)


def check_duplicates(idx: int, season: str, name: str) -> Union[pd.DataFrame, str]:
    if name == "Danny Ward":
        print("Danny Ward")
    df = get_duplicates_df()
    matches = df[(df["id"] == idx) & (df["season"] == int(season))]
    return matches.iloc[0]["name"] if len(matches) > 0 else name


def get_player_details(season: str) -> dict:
    """generate player details json files"""
    season_longname = get_long_season_name(season)
    print("-------- SEASON", season_longname, "--------")

    teams_dict = get_teams_dict(season)
    positions_df = get_positions_df(season)

    fixtures_df, got_fixtures = get_fixtures_df(season)

    # names of all player directories for this season
    sub_dirs = glob(PLAYERS_DIR.format(season_longname) + "/*/")

    output = {}
    for directory in sub_dirs:
        name = path_to_name(directory)
        idx = path_to_index(directory)
        print("Doing", name)

        player_dict = process_file(
            os.path.join(directory, PLAYERS_FILE),
            teams_dict,
            fixtures_df,
            got_fixtures,
        )

        # get player position
        if idx in positions_df.index:
            position = str(positions_df.loc[idx, "position"])
        else:
            matches = positions_df[positions_df["name"] == name]
            if len(matches) == 1:
                position = str(matches["position"].iloc[0])
            else:
                position = None
                print(f"!!!FAILED!!! {len(matches)} possibilities for {name}")

        for fixture in player_dict:
            fixture["position"] = position

        name = check_duplicates(idx, season, name)
        output[name] = player_dict

    return output


def make_player_details(seasons: Optional[List[str]] = []):
    print(f"Cloning {GIT_REPO}...")
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            GIT_REPO,
            REPO_DIR,
        ]
    )

    if not seasons:
        seasons = get_past_seasons(3)
    for season in seasons:
        output = get_player_details(season)
        print("Saving JSON")
        with open(SAVE_NAME.format(season), "w") as f:
            json.dump(output, f)

    print(f"Deleting {REPO_DIR}...")
    shutil.rmtree(REPO_DIR)

    print("DONE!")


if __name__ == "__main__":
    make_player_details(["2223"])
