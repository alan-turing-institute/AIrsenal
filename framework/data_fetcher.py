"""
functions to retrieve current FPL data.
"""

import requests
import json

##from .utils import get_gameweek_by_date

FPL_API_URL = "https://fantasy.premierleague.com/drf/bootstrap-static"
FPL_DETAIL_URL = "https://fantasy.premierleague.com/drf/element-summary"
DATA_DIR = "./data"


class DataFetcher(object):
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
        for player in all_data["elements"]:
            self.current_player_data[player["id"]] = player
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
        for team in all_data["teams"]:
            self.current_team_data[team["code"]] = team
        return self.current_team_data

    def get_gameweek_data_for_player(self, player_id, gameweek=None):
        """
        return cached data if available, otherwise
        fetch it from API.
        Return a list, as in double-gameweeks, a player can play more than
        one match in a gameweek.
        """
        if not player_id in self.player_gameweek_data.keys():
            self.player_gameweek_data[player_id] = {}
            if (not gameweek) or (
                not gameweek in self.player_gameweek_data[player_id].keys()
            ):

                r = requests.get("{}/{}".format(FPL_DETAIL_URL, player_id))
                if not r.status_code == 200:
                    print("Error retrieving data for player {}".format(player_id))
                    return []
                player_detail = json.loads(r.content)

                for game in player_detail["history"]:
                    gw = game["round"]
                    if not gw in self.player_gameweek_data[player_id].keys():
                        self.player_gameweek_data[player_id][gw] = []
                    self.player_gameweek_data[player_id][gw].append(game)
        if gameweek:
            if not gameweek in self.player_gameweek_data[player_id].keys():
                print(
                    "Data not available for player {} week {}".format(
                        player_id, gameweek
                    )
                )
                return []
            return self.player_gameweek_data[player_id][gameweek]
        else:
            return self.player_gameweek_data[player_id]
