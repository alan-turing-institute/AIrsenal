"""
Generate player details json files from a locally cloned copy of the
https://github.com/vaastav/Fantasy-Premier-League repository on GitHub.

"""
import json
import os
from glob import glob

import pandas as pd

from airsenal.framework.mappings import (
    alternative_player_names,
    alternative_team_names,
    positions,
)
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
REPO_DIR = os.path.join(SCRIPT_DIR, "../../../Fantasy-Premier-League/data/{}")
# Path to directory of player data
PLAYERS_DIR = os.path.join(REPO_DIR, "players")
# file containing GW data in every sub_directory in PLAYERS_DIR
PLAYERS_FILE = "gw.csv"
# Path to fixtures files
FIXTURES_PATH = os.path.join(REPO_DIR, "fixtures.csv")
# Path to raw player summary data
RAW_PATH = os.path.join(REPO_DIR, "players_raw.csv")

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
# player summary file - to get player's position for a season
SUMMARY_PATH = os.path.join(SCRIPT_DIR, "../data/player_summary_{}.json")

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
    # attributes
    "value": "value",
    "played_for": "played_for",
    # needed to determine fixture if two teams play each other twice same gw
    "kickoff_time": "kickoff_time",
    "was_home": "was_home",
}


def path_to_name(path):
    """function to take a sub directory path into a key for output json
    i.e. player name from directory path
    """
    # get directory name from full path
    dir_name = os.path.basename(os.path.dirname(path))

    return " ".join(x for x in dir_name.split("_") if not x.isdigit())


def get_long_season_name(short_name):
    """Convert short season name of format 1718 to long name like 2017-18."""
    return "20" + short_name[:2] + "-" + short_name[2:]


def get_teams_dict(season):
    teams_df = pd.read_csv(TEAM_PATH.format(season))
    return {row["team_id"]: row["name"] for _, row in teams_df.iterrows()}


def get_positions_df(season):
    """
    Get dataframe of player names and their positions for the given season,
    using the players_raw file from the FPL results repo.
    """
    season_longname = get_long_season_name(season)
    raw_df = pd.read_csv(RAW_PATH.format(season_longname))

    raw_df["name"] = raw_df["first_name"] + " " + raw_df["second_name"]

    # replace element type codes with position strings
    raw_df["position"] = raw_df["element_type"].replace(positions)

    raw_df.set_index("name", inplace=True)
    return raw_df["position"]


def get_fixtures_df(season):
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
        raise FileNotFoundError(
            "Couldn't find fixtures file for {} season".format(season)
        )

    return fixtures_df, got_fixtures


def get_played_for_from_fixtures(fixture_id, opponent_id, was_home, fixtures_df):
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
            """Error finding team played for with fixture id {},
                         opponent_id {} and was_home {}""".format(
                fixture_id, opponent_id, was_home
            )
        )


def get_played_for_from_results(player_row, results_df, teams_dict):
    """
    Find what team a played for given the gameweek, match date, opposing team,
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
            """Found no matches with gw {}, was_home {}
                                and opponent {}""".format(
                gw, was_home, opponent
            )
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


def make_player_details(seasons=get_past_seasons(3)):
    """generate player details json files"""
    if isinstance(seasons, str):
        seasons = [seasons]

    for season in seasons:
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
            print("Doing", name)

            player_dict = process_file(
                os.path.join(directory, PLAYERS_FILE),
                teams_dict,
                fixtures_df,
                got_fixtures,
            )

            # get player position
            if name in positions_df.index:
                position = str(positions_df.loc[name])
            else:
                position = "NA"
                for k, v in alternative_player_names.items():
                    if name == k or name in v:
                        if k in positions_df.index:
                            position = str(positions_df.loc[k])
                        else:
                            for alt_name in v:
                                if alt_name in positions_df.index:
                                    position = str(positions_df.loc[alt_name])
                                    break
                        print("found", position, "via alternative name")
                        break
            if position == "NA":
                print("!!!FAILED!!! Could not find position for", name)
            for fixture in player_dict:
                fixture["position"] = position

            output[name] = player_dict

        print("Saving JSON")
        with open(SAVE_NAME.format(season), "w") as f:
            json.dump(output, f)

    print("DONE!")


if __name__ == "__main__":
    make_player_details()
