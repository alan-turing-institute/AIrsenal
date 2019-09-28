from glob import glob
import os
import re
import json
import pandas as pd
from airsenal.framework.schema import PlayerScore
from airsenal.framework.utils import get_past_seasons

# players directory for season of interest from this git repo:
# https://github.com/vaastav/Fantasy-Premier-League
DATA_DIR = '/Users/jroberts/GitHub/Fantasy-Premier-League/data/{}/players'
           
# file of interest present in every sub_directory in data_dir
FILE_NAME = 'gw.csv'

# teams path - to get mapping of ids to team names
TEAM_PATH = '../data/teams_{}.csv'

# where to save output
SAVE_NAME = '../data/player_details_{}.json'

# dictionary of key in input file to key in output file

# Features to extract {name in files: name in database}
key_dict = {
    'round': 'gameweek',
    'total_points': 'points',
    'goals_scored': 'goals',
    'assists': 'assists',
    'goals_conceded': 'conceded',
    'bonus': 'bonus',
    'minutes': 'minutes',
    'opponent_team': 'opponent',  # id in input, 3 letters in output!!!
    # extended features
    'clean_sheets': 'clean_sheets',
    'own_goals': 'own_goals',
    'penalties_saved': 'penalties_saved',
    'penalties_missed': 'penalties_missed',
    'yellow_cards': 'yellow_cards',
    'red_cards': 'red_cards',
    'saves': 'saves',
    'bps': 'bps',
    'influence': 'influence',
    'creativity': 'creativity',
    'threat': 'threat',
    'ict_index': 'ict_index',
    'value': 'value',
    'transfers_balance': 'transfers_balance',
    'selected': 'selected',
    'transfers_in': 'transfers_in',
    'transfers_out': 'transfers_out'
}


def path_to_name(path):
    """function to take a sub directory path into a key for output json
    i.e. player name from directory path
    """
    # get directory name from full path
    dir_name = os.path.basename(os.path.dirname(path))
    
    # replace _ with spaces, plus exclude anything that's numeric
    # (some seasons have player id in directory name, we just want name)
    name = " ".join([x for x in dir_name.split("_") if not x.isdigit()])

    return name


def get_long_season_name(short_name):
    """convert short season name of format 1718 to long name like 2017-18.
    Past generations: sorry this doesn't work for 1999 and earlier!
    Future generations: sorry this doesn't work for the 2100s onwards!
    """
    return '20' + short_name[:2] + '-' + short_name[2:]


def get_teams_dict(season):
    teams_df = pd.read_csv(TEAM_PATH.format(season))
    return {row['team_id']: row['name'] for _, row in teams_df.iterrows()}


def process_file(path, teams_dict):
    """function to load and process one of the player score files
    """
    # load input file
    df = pd.read_csv(path)

    # extract columns of interest
    df = df[key_dict.keys()]
    
    # rename columns to desired output names
    df.rename(columns=key_dict, inplace=True)
    
    # rename opponent ids with short names
    df['opponent'].replace(teams_dict, inplace=True)
    
    # want everything in output to be strings
    df = df.applymap(str)
    
    # return json like dictionary
    return df.to_dict(orient='records')


def make_player_details(seasons=get_past_seasons(3)):
    """generate player details json files"""
    if isinstance(seasons, str):
        seasons = [seasons]
        
    for season in seasons:
        season_longname = get_long_season_name(season)
        print('SEASON', season_longname)
        
        sub_dirs = glob(DATA_DIR.format(season_longname) + '/*/')
        
        teams_dict = get_teams_dict(season)
        
        output = {}
        for directory in sub_dirs:
            name = path_to_name(directory)
            print('Doing', name)
            player_dict = process_file(os.path.join(directory, FILE_NAME),
                                       teams_dict)
            output[name] = player_dict
        
        print('Saving JSON')
        with open(SAVE_NAME.format(season), 'w') as f:
            json.dump(output, f)

    
if __name__ == '__main__':
    make_player_details()
