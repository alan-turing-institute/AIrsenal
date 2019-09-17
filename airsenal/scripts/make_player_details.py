from glob import glob
import os
import re
import json
import pandas as pd
from airsenal.framework.schema import PlayerScore

season_longname = '2018-19'
season_shortname = '1819'

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


def get_teams_dict():
    teams_df = pd.read_csv(team_path)

    return {row['team_id']: row['name'] for _, row in teams_df.iterrows()}


def process_file(path, teams_dict):
    """function to load and process one of the files
    """
    # load columns of interest from input file
    df = pd.read_csv(path)

    # player id
    key = str(df['element'][0])
    
    # extract columns of interest
    df = df[key_dict.keys()]
    
    # rename columns to desired output names
    df.rename(columns=key_dict, inplace=True)
    
    # renane opponent ids with short names
    df['opponent'].replace(teams_dict, inplace=True)
    
    # want everything in output to be strings
    df = df.applymap(str)
    
    # return json like dictionary
    return key, df.to_dict(orient='records')
    

if __name__ == '__main__':
    sub_dirs = glob(data_dir + '/*/')
    
    teams_dict = get_teams_dict()
    
    output = {}
    for directory in sub_dirs:
        print(directory)
        key, player_dict = process_file(os.path.join(directory, file_name),
                                        teams_dict)
        output[key] = player_dict
    
    with open(save_name, 'w') as f:
        json.dump(output, f)
