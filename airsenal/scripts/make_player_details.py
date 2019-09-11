from glob import glob
import os
import re
import json
import pandas as pd
from airsenal.framework.schema import PLAYERSCORE_EXTENDED_FEATS

season_longname = '2016-17'
season_shortname = '1617'

# players directory for season of interest from this git repo:
# https://github.com/vaastav/Fantasy-Premier-League
data_dir = '/Users/jroberts/GitHub/Fantasy-Premier-League/data/{}/players'\
           .format(season_longname)

# file of interest present in every sub_directory in data_dir
file_name = 'gw.csv'

# teams path - to get mapping of ids to team names
team_path = '../data/teams_{}.csv'.format(season_shortname)

# where to save output
save_name = '../data/player_details_{}.json'.format(season_shortname)

# dictionary of key in input file to key in output file

# Core features (used in model)
key_dict = {
    'round': 'gameweek',
    'total_points': 'points',
    'goals_scored': 'goals',
    'assists': 'assists',
    'goals_conceded': 'conceded',
    'bonus': 'bonus',
    'own_goals': 'own_goals',
    'minutes': 'minutes',
    'opponent_team': 'opponent'  # id in input, 3 letters in output!!!
}
# Additional features (may be used in future)
additional_features = PLAYERSCORE_EXTENDED_FEATS.keys()
for feat in additional_features:
    key_dict[feat] = feat


def path_to_key(path):
    """function to take a sub directory path into a key for output json
    i.e. player name from directory path
    """
    # get directory name from full path
    dir_name = os.path.basename(os.path.dirname(path))

    # strip everything after final underscore (should be player number)
    key = re.sub('_[^_]+$', '', dir_name)

    # replace remaining underscores with spaces
    key = key.replace('_', ' ')

    return key


def get_teams_dict():    
    teams_df = pd.read_csv(team_path)

    return {row['team_id']: row['name'] for _, row in teams_df.iterrows()}


def process_file(path, teams_dict):
    """function to load and process one of the files
    """
    # load columns of interest from input file
    df = pd.read_csv(path, usecols=key_dict.keys())

    # rename columns to desired output names
    df.rename(columns=key_dict, inplace=True)
    
    # renane opponent ids with short names
    df['opponent'].replace(teams_dict, inplace=True)
    
    # want everything in output to be strings
    df = df.applymap(str)

    # return json like dictionary
    return df.to_dict(orient='records')
    

if __name__ == '__main__':
    sub_dirs = glob(data_dir + '/*/')
    
    teams_dict = get_teams_dict()
    
    output = {path_to_key(directory):
              process_file(os.path.join(directory, file_name), teams_dict)
              for directory in sub_dirs}

    with open(save_name, 'w') as f:
        json.dump(output, f)
