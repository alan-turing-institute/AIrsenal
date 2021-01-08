"""
Classes to query the FPL API to retrieve current FPL data,
and to query football-data.org to retrieve match and fixture data.
"""
import os
import requests
import json
import time
import getpass

API_HOME = "https://fantasy.premierleague.com/api"


class FPLDataFetcher(object):
    """
    hold current and historic FPL data in memory,
    or retrieve it if not already cached.
    """

    def __init__(self, fpl_team_id=None):
        self.current_summary_data = None
        self.current_event_data = None
        self.current_player_data = None
        self.current_team_data = None
        self.player_gameweek_data = {}
        self.fpl_team_history_data = None
        self.fpl_transfer_history_data = None
        self.fpl_league_data = None
        self.fpl_team_data = {}  # players in squad, by gameweek
        self.fixture_data = None
        for ID in ["FPL_LEAGUE_ID", "FPL_TEAM_ID", "FPL_LOGIN", "FPL_PASSWORD"]:
            if ID in os.environ.keys():
                self.__setattr__(ID, os.environ[ID])
            elif os.path.exists(
                os.path.join(os.path.dirname(__file__), "..", "data", "{}".format(ID))
            ):
                self.__setattr__(
                    ID,
                    open(
                        os.path.join(
                            os.path.dirname(__file__), "..", "data", "{}".format(ID)
                        )
                    )
                    .read()
                    .strip(),
                )
            else:
                self.__setattr__(ID, "MISSING_ID")
        if fpl_team_id is not None:
            self.FPL_TEAM_ID = fpl_team_id  # update entry with command line arg
        self.FPL_SUMMARY_API_URL = API_HOME + "/bootstrap-static/"
        self.FPL_DETAIL_URL = API_HOME + "/element-summary/{}/"
        self.FPL_HISTORY_URL = API_HOME + "/entry/{}/history/"
        self.FPL_TEAM_URL = API_HOME + "/entry/{}/event/{}/picks/"
        self.FPL_TEAM_TRANSFER_URL = API_HOME + "/entry/{}/transfers/"
        self.FPL_LEAGUE_URL = API_HOME + (
            "/leagues-classic/{}/standings/?page_new_entries=1&page_standings=1"
        ).format(self.FPL_LEAGUE_ID)
        self.FPL_FIXTURE_URL = API_HOME + "/fixtures/"

    def get_fpl_credentials(self):
        """
        If we didn't have FPL_LOGIN and FPL_PASSWORD available as files in
        airsenal/data or as environment variables, prompt the user for them.
        """
        print(
            """
            Accessing FPL mini-league data requires the login (email address) and
            password for your FPL account.
            """
        )
        self.FPL_LOGIN = input("Please enter FPL login: ")
        self.FPL_PASSWORD = getpass.getpass("Please enter FPL password: ")
        data_loc = os.path.join(os.path.dirname(__file__), "..", "data")
        store_credentials = ""
        while not (
            store_credentials.lower() == "y" or store_credentials.lower() == "n"
        ):
            store_credentials = input(
                (
                    "\nWould you like to store these credentials in {}"
                    " so that you won't be prompted for them again? (y/n): "
                ).format(data_loc)
            )
        if store_credentials.lower() == "y":
            with open(os.path.join(data_loc, "FPL_LOGIN"), "w") as login_file:
                login_file.write(self.FPL_LOGIN)
            with open(os.path.join(data_loc, "FPL_PASSWORD"), "w") as passwd_file:
                passwd_file.write(self.FPL_PASSWORD)

    def get_current_summary_data(self):
        """
        return cached data if present, otherwise retrieve it
        from the API.
        """
        if self.current_summary_data:
            return self.current_summary_data
        else:
            r = requests.get(self.FPL_SUMMARY_API_URL)
            if not r.status_code == 200:
                print("Unable to access FPL API")
                return None
            self.current_summary_data = json.loads(r.content.decode("utf-8"))
        return self.current_summary_data

    def get_fpl_team_data(self, gameweek, team_id=None):
        """
        Use team id to get team data from the FPL API.
        If no team_id is specified, we assume it is 'our' team
        $TEAM_ID, and cache the results in a dictionary.
        """
        if not team_id and gameweek in self.fpl_team_data.keys():
            return self.fpl_team_data[gameweek]
        else:
            if not team_id:
                team_id = self.FPL_TEAM_ID
            url = self.FPL_TEAM_URL.format(team_id, gameweek)
            r = requests.get(url)
            if not r.status_code == 200:
                print("Unable to access FPL team API {}".format(url))
                return None
            team_data = json.loads(r.content.decode("utf-8"))
            if not team_id:
                self.fpl_team_data[gameweek] = team_data
        return team_data

    def get_fpl_team_history_data(self, team_id=None):
        """
        Use our team id to get history data from the FPL API.
        """
        if self.fpl_team_history_data and not team_id:
            return self.fpl_team_history_data
        else:
            if not team_id:
                team_id = self.FPL_TEAM_ID
            url = self.FPL_HISTORY_URL.format(team_id)
            r = requests.get(url)
            if not r.status_code == 200:
                print("Unable to access FPL team history API")
                return None
            self.fpl_team_history_data = json.loads(r.content.decode("utf-8"))
        return self.fpl_team_history_data

    def get_fpl_transfer_data(self):
        """
        Get our transfer history from the FPL API.
        """
        # return cached value if we already retrieved it.
        if self.fpl_transfer_history_data:
            return self.fpl_transfer_history_data
        # or get it from the API.
        url = self.FPL_TEAM_TRANSFER_URL.format(self.FPL_TEAM_ID)
        r = requests.get(url)
        if not r.status_code == 200:
            print("Unable to access FPL transfer history API")
            return None
        # get transfer history from api and reverse order so that
        # oldest transfers at start of list and newest at end.
        self.fpl_transfer_history_data = list(
            reversed(json.loads(r.content.decode("utf-8")))
        )
        return self.fpl_transfer_history_data

    def get_fpl_league_data(self):
        """
        Use our league id to get history data from the FPL API.
        """
        if self.fpl_league_data:
            return self.fpl_league_data
        else:
            session = requests.session()
            url = "https://users.premierleague.com/accounts/login/"
            print("FPL credentials {} {}".format(self.FPL_LOGIN, self.FPL_PASSWORD))
            if (
                (not self.FPL_LOGIN)
                or (not self.FPL_PASSWORD)
                or self.FPL_LOGIN == "MISSING_ID"
                or self.FPL_PASSWORD == "MISSING_ID"
            ):
                # prompt the user for credentials
                self.get_fpl_credentials()
            headers = {
                "login": self.FPL_LOGIN,
                "password": self.FPL_PASSWORD,
                "app": "plfpl-web",
                "redirect_uri": "https://fantasy.premierleague.com/a/login",
            }
            session.post(url, data=headers)

            r = session.get(self.FPL_LEAGUE_URL, headers=headers)
            if not r.status_code == 200:
                print("Unable to access FPL league API")
                return None
            self.fpl_league_data = json.loads(r.content.decode("utf-8"))
        return self.fpl_league_data

    def get_event_data(self):
        """
        return a dict of gameweeks - whether they are finished or not, and
        the transfer deadline.
        """
        if self.current_event_data:
            return self.current_event_data
        self.current_event_data = {}
        all_data = self.get_current_summary_data()
        for event in all_data["events"]:
            self.current_event_data[event["id"]] = {
                "deadline": event["deadline_time"],
                "is_finished": event["finished"],
            }
        return self.current_event_data

    def get_player_summary_data(self):
        """
        Use the current_data to build a dictionary, keyed by player_api_id
        in order to retrieve a player without having to loop through
        a whole list.
        """
        if self.current_player_data:
            return self.current_player_data
        self.current_player_data = {}
        all_data = self.get_current_summary_data()
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
        all_data = self.get_current_summary_data()
        for team in all_data["teams"]:
            self.current_team_data[team["code"]] = team
        return self.current_team_data

    def get_gameweek_data_for_player(self, player_api_id, gameweek=None):
        """
        return cached data if available, otherwise
        fetch it from API.
        Return a list, as in double-gameweeks, a player can play more than
        one match in a gameweek.
        """
        if player_api_id not in self.player_gameweek_data.keys():
            self.player_gameweek_data[player_api_id] = {}
            if (not gameweek) or (
                gameweek not in self.player_gameweek_data[player_api_id].keys()
            ):
                got_data = False
                n_tries = 0
                player_detail = {}
                while (not got_data) and n_tries < 3:
                    try:
                        r = requests.get(self.FPL_DETAIL_URL.format(player_api_id))
                        if not r.status_code == 200:
                            print(
                                "Error retrieving data for player {}".format(
                                    player_api_id
                                )
                            )
                            return []
                        player_detail = json.loads(r.content)
                        got_data = True
                    except requests.exceptions.ConnectionError:
                        print("connection error, retrying {}".format(n_tries))
                        time.sleep(1)
                        n_tries += 1
                if not player_detail:
                    print(
                        "Unable to get player_detail data for {}".format(player_api_id)
                    )
                    return []
                for game in player_detail["history"]:
                    gw = game["round"]
                    if gw not in self.player_gameweek_data[player_api_id].keys():
                        self.player_gameweek_data[player_api_id][gw] = []
                    self.player_gameweek_data[player_api_id][gw].append(game)
        if gameweek:
            if gameweek not in self.player_gameweek_data[player_api_id].keys():
                print(
                    "Data not available for player {} week {}".format(
                        player_api_id, gameweek
                    )
                )
                return []
            return self.player_gameweek_data[player_api_id][gameweek]
        else:
            return self.player_gameweek_data[player_api_id]

    def get_fixture_data(self):
        """
        Get the fixture list from the FPL API.
        """
        if not self.fixture_data:
            self.fixture_data = requests.get(self.FPL_FIXTURE_URL).json()
        else:
            pass
        return self.fixture_data
