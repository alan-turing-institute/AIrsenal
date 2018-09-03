"""
Classes to query the FPL API to retrieve current FPL data,
and to query football-data.org to retrieve match and fixture data.
"""
import os
import requests
import json
from .mappings import alternative_team_names

FOOTBALL_DATA_URL = "http://api.football-data.org/v2/competitions/2021"
FOOTBALL_DATA_API_KEY=""
if os.path.exists("../data/FD_API_KEY"):
    FOOTBALL_DATA_API_KEY = open("../data/FD_API_KEY").read().strip()
elif os.path.exists("data/FD_API_KEY"):
    FOOTBALL_DATA_API_KEY = open("data/FD_API_KEY").read().strip()
else:
    print("Couldn't find data/FD_API_KEY - can't use football-data.org API")

FPL_API_URL = "https://fantasy.premierleague.com/drf/bootstrap-static"
FPL_DETAIL_URL = "https://fantasy.premierleague.com/drf/element-summary"
DATA_DIR = "./data"

class FPLDataFetcher(object):
    """
    hold current and historic FPL data in memory,
    or retrieve it if not already cached.
    """
    def __init__(self):
        self.current_data = None
        self.historic_data = {}
        self.current_player_data = None
        self.current_team_data = None
        self.player_gameweek_data = {}

    def get_current_data(self):
        """
        return cached data if present, otherwise retrieve it
        from the API.
        """
        if self.current_data:
            return self.current_data
        else:
            r = requests.get(FPL_API_URL)
            if not r.status_code == 200:
                print("Unable to access FPL API")
                return None
            self.current_data = json.loads(r.content.decode("utf-8"))
        return self.current_data


    def get_player_summary_data(self):
        """
        Use the current_data to build a dictionary, keyed by player_id
        in order to retrieve a player without having to loop through
        a whole list.
        """
        if self.current_player_data:
            return self.current_player_data
        self.current_player_data = {}
        all_data = self.get_current_data()
        for player in all_data['elements']:
            self.current_player_data[player['id']] = player
        return self.current_player_data


    def get_current_team_data(self):
        """
        Use the current_data to build a dictionary keyed by team code,
        in order to retrieve a player's team without looping through the
        whole list.
        """
        if self.current_team_data:
            return self.current_team_data
        self.current_team_data = {}
        all_data = self.get_current_data()
        for team in all_data['teams']:
            self.current_team_data[team['code']] = team
        return self.current_team_data

    def get_gameweek_data_for_player(self,player_id, gameweek=None):
        """
        return cached data if available, otherwise
        fetch it from API.
        Return a list, as in double-gameweeks, a player can play more than
        one match in a gameweek.
        """
        if not player_id in self.player_gameweek_data.keys():
            self.player_gameweek_data[player_id] = {}
            if (not gameweek) or (not gameweek in self.player_gameweek_data[player_id].keys()):

                r = requests.get("{}/{}".format(FPL_DETAIL_URL,player_id))
                if not r.status_code == 200:
                    print("Error retrieving data for player {}".format(player_id))
                    return []
                player_detail = json.loads(r.content)

                for game in player_detail['history']:
                    gw = game['round']
                    if not gw in self.player_gameweek_data[player_id].keys():
                        self.player_gameweek_data[player_id][gw] = []
                    self.player_gameweek_data[player_id][gw].append(game)
        if gameweek:
            if not gameweek in self.player_gameweek_data[player_id].keys():
                print("Data not available for player {} week {}".format(
                    player_id, gameweek))
                return []
            return self.player_gameweek_data[player_id][gameweek]
        else:
            return self.player_gameweek_data[player_id]


class MatchDataFetcher(object):
    """
    Access the football-data.org API to get information on match results and fixtures.
    """
    def __init__(self):
        self.data = {}
        pass


    def _get_gameweek_data(self, gameweek):
        """
        query the matches endpoint
        """
        uri = "{}/matches?matchday={}".format(FOOTBALL_DATA_URL, gameweek)
        headers = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}
        r = requests.get(uri, headers=headers)
        if r.status_code != 200:
            return r
        self.data[gameweek] = json.loads(r.content)['matches']


    def get_results(self, gameweek):
        """
        get results for matches that have been played.
        Return list of tuples:
        [(date, home_team, away_team, home_score, away_score)]
        """
        output_list = []
        if not gameweek in self.data.keys():
            self._get_gameweek_data(gameweek)
        if not self.data[gameweek][0]['status'] == "FINISHED":
            print("Fixtures not finished - have they been played yet?")
            return output_list

        for match in self.data[gameweek]:
            home_team = None
            away_team = None
            for k,v in alternative_team_names.items():
                if match['homeTeam']['name'] in v:
                    home_team = k
                elif match['awayTeam']['name'] in v:
                    away_team = k
                if home_team and away_team:
                    break
            output_list.append((match['utcDate'],
                                home_team,
                                away_team,
                                match['score']['fullTime']['homeTeam'],
                                match['score']['fullTime']['awayTeam'])
            )
        return output_list



    def get_fixtures(self, gameweek):
        """
        get upcoming fixtures.
        Return list of tuples:
        [(date, home_team, away_team)]
        """
        output_list = []
        if not gameweek in self.data.keys():
            self._get_gameweek_data(gameweek)
        if not self.data[gameweek][0]['status'] == "SCHEDULED":
            print("Fixtures not scheduled - have they already been played?")
            return output_list

        for fixture in self.data[gameweek]:
            home_team = None
            away_team = None
            for k,v in alternative_team_names:
                if fixture['homeTeam'] in v:
                    home_team = k
                elif fixture['awayTeam'] in v:
                    away_team = k
                if home_team and away_team:
                    break
            output_list.append((fixture['utcDate'],
                                home_team,
                                away_team))
        return output_list
