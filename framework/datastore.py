"""
functions to retrieve current and historical FPL data.
"""

import requests
import json

FPL_API_URL = "https://fantasy.premierleague.com/drf/bootstrap-static"

DATA_DIR = "./data"

class DataStore(object):
    """
    hold current and historic FPL data in memory,
    or retrieve it if not already cached.
    """
    def __init__(self):
        self.current_data = None
        self.historic_data = {}
        self.current_player_data = None
        self.current_team_data = None


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


    def get_current_player_data(self):
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

    def get_historic_data(self,year):
        """
        return cached data if available, otherwise
        scrape it.
        """
        if year in self.historic_data.keys():
            return self.historic_data[year]
