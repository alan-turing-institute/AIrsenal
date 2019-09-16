from glob import glob
import os
import re
import json
import pandas as pd
from airsenal.framework.schema import PlayerScore

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
    'minutes': 'minutes',
    'opponent_team': 'opponent'  # id in input, 3 letters in output!!!
}
# Additional features (may be used in future)
# get features excluding the core ones already defined above
ps = PlayerScore()
extended_feats = [col for col in ps.__table__.columns.keys()
                  if col not in ["id",
                                 "player_team",
                                 "opponent",
                                 "goals",
                                 "assists",
                                 "bonus",
                                 "points",
                                 "conceded",
                                 "minutes",
                                 "player_id",
                                 "result_id",
                                 "fixture_id"]]
for feat in extended_feats:
    key_dict[feat] = feat


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
