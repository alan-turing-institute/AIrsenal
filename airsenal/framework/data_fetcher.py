"""
Classes to query the FPL API to retrieve current FPL data,
and to query football-data.org to retrieve match and fixture data.
"""
import getpass
import json
import os
import time

import requests

API_HOME = "https://fantasy.premierleague.com/api"


class FPLDataFetcher(object):
    """
    hold current and historic FPL data in memory,
    or retrieve it if not already cached.
    """

    def __init__(self, fpl_team_id=None, rsession=None):
        self.rsession = rsession if rsession else requests.session()
        self.logged_in = False
        self.current_summary_data = None
        self.current_event_data = None
        self.current_player_data = None
        self.current_team_data = None
        self.current_squad_data = {}
        self.player_gameweek_data = {}
        self.fpl_team_history_data = None
        # transfer history data is a dict, keyed by fpl_team_id
        self.fpl_transfer_history_data = {}
        self.fpl_league_data = None
        self.fpl_team_data = {}  # players in squad, by gameweek
        self.fixture_data = None
        for ID in [
            "FPL_LEAGUE_ID",
            "FPL_TEAM_ID",
            "FPL_LOGIN",
            "FPL_PASSWORD",
            "DISCORD_WEBHOOK",
        ]:
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
        if self.FPL_TEAM_ID is not None and self.FPL_TEAM_ID != "MISSING_ID":
            try:
                self.FPL_TEAM_ID = int(self.FPL_TEAM_ID)
            except ValueError:
                raise ValueError(
                    f"FPL_TEAM_ID in environment variable and/or data/FPL_TEAM_ID "
                    f"file should be a valid integer. Please correct it or remove "
                    f" it if you're using the command line argument. "
                    f"Found: {self.FPL_TEAM_ID}"
                )
        if fpl_team_id is not None:
            if isinstance(fpl_team_id, int):
                self.FPL_TEAM_ID = fpl_team_id  # update entry with command line arg
            else:
                raise ValueError(
                    f"FPL_TEAM_ID should be an integer. Found: {fpl_team_id}"
                )
        self.FPL_SUMMARY_API_URL = API_HOME + "/bootstrap-static/"
        self.FPL_DETAIL_URL = API_HOME + "/element-summary/{}/"
        self.FPL_HISTORY_URL = API_HOME + "/entry/{}/history/"
        self.FPL_TEAM_URL = API_HOME + "/entry/{}/event/{}/picks/"
        self.FPL_TEAM_TRANSFER_URL = API_HOME + "/entry/{}/transfers/"
        self.FPL_LEAGUE_URL = API_HOME + (
            "/leagues-classic/{}/standings/?page_new_entries=1&page_standings=1"
        ).format(self.FPL_LEAGUE_ID)
        self.FPL_FIXTURE_URL = API_HOME + "/fixtures/"
        self.FPL_LOGIN_URL = "https://users.premierleague.com/accounts/login/"
        self.FPL_LOGIN_REDIRECT_URL = "https://fantasy.premierleague.com/a/login"
        self.FPL_MYTEAM_URL = API_HOME + "/my-team/{}/"

        # login, if desired

    #        self.login()

    def get_fpl_credentials(self):
        """
        If we didn't have FPL_LOGIN and FPL_PASSWORD available as files in
        airsenal/data or as environment variables, prompt the user for them.
        """
        print(
            """
            Accessing the most up-to-date data on your squad, or automatic transfers,
            requires the login (email address) and password for your FPL account.
            """
        )
        self.FPL_LOGIN = input("Please enter FPL login: ")
        self.FPL_PASSWORD = getpass.getpass("Please enter FPL password: ")
        data_loc = os.path.join(os.path.dirname(__file__), "..", "data")
        store_credentials = ""
        while store_credentials.lower() not in ["y", "n"]:
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
            print(
                """
                Wrote files {} and {}.
                You may need to do 'pip install .' for these to be picked up.
                """.format(
                    login_file, passwd_file
                )
            )

    def login(self):
        """
        only needed for accessing mini-league data, or team info for current gw.
        """
        if self.logged_in:
            return
        if (
            (not self.FPL_LOGIN)
            or (not self.FPL_PASSWORD)
            or (self.FPL_LOGIN == "MISSING_ID")
            or (self.FPL_PASSWORD == "MISSING_ID")
        ):
            do_login = ""
            while do_login.lower() not in ["y", "n"]:
                do_login = input(
                    (
                        "\nWould you like to login to the FPL API?"
                        "\nThis is not necessary for most AIrsenal actions, "
                        "\nbut may improve accuracy of player sell values,"
                        "\nand free transfers for your team, and will also "
                        "\nenable AIrsenal to make transfers for you through "
                        "\nthe API. (y/n): "
                    )
                )
            if do_login.lower() == "y":
                self.get_fpl_credentials()
            else:
                return

        headers = {
            "User-Agent": "Dalvik/2.1.0 (Linux; U; Android 5.1; PRO 5 Build/LMY47D)"
        }
        data = {
            "login": self.FPL_LOGIN,
            "password": self.FPL_PASSWORD,
            "app": "plfpl-web",
            "redirect_uri": self.FPL_LOGIN_REDIRECT_URL,
        }
        response = self.rsession.post(self.FPL_LOGIN_URL, data=data, headers=headers)
        if response.status_code != 200:
            print(f"Error loging in: {response.content}")
        else:
            print("Logged in successfully")
            self.logged_in = True

    def get_current_squad_data(self, fpl_team_id=None):
        """
        Requires login.  Return the current squad data, including
        "picks", bank, and free transfers.
        """
        if not fpl_team_id:
            if self.FPL_TEAM_ID and self.FPL_TEAM_ID != "MISSING_ID":
                fpl_team_id = self.FPL_TEAM_ID
            else:
                raise RuntimeError("Please specify FPL team ID")
        if fpl_team_id in self.current_squad_data:
            return self.current_squad_data[fpl_team_id]
        self.login()
        url = self.FPL_MYTEAM_URL.format(fpl_team_id)
        self.current_squad_data[fpl_team_id] = self._get_request(url)
        return self.current_squad_data[fpl_team_id]

    def get_current_picks(self, fpl_team_id=None):
        """
        Returns the players picked for the upcoming gameweek, including
        purchase and selling prices, and whether they are subs or not.
        Requires login
        """
        squad_data = self.get_current_squad_data(fpl_team_id)
        return squad_data["picks"]

    def get_num_free_transfers(self, fpl_team_id=None):
        """
        Returns the number of free transfers for the upcoming gameweek.
        Requires login
        """
        squad_data = self.get_current_squad_data(fpl_team_id)
        return squad_data["transfers"]["limit"]

    def get_current_bank(self, fpl_team_id=None):
        """
        Returns the remaining bank (in 0.1M) for the upcoming gameweek.
        Requires login
        """
        squad_data = self.get_current_squad_data(fpl_team_id)
        return squad_data["transfers"]["bank"]

    def get_available_chips(self, fpl_team_id=None):
        """
        Returns a list of chips that are available to be played in upcoming gameweek.
        """
        squad_data = self.get_current_squad_data(fpl_team_id)
        chip_list = [
            chip["name"]
            for chip in squad_data["chips"]
            if chip["status_for_entry"] == "available"
        ]
        return chip_list

    def get_current_summary_data(self):
        """
        return cached data if present, otherwise retrieve it
        from the API.
        """
        if self.current_summary_data:
            return self.current_summary_data
        self.current_summary_data = self._get_request(self.FPL_SUMMARY_API_URL)
        return self.current_summary_data

    def get_fpl_team_data(self, gameweek, fpl_team_id=None):
        """
        Use FPL team id to get team data from the FPL API.
        If no fpl_team_id is specified, we assume it is 'our' team
        $FPL_TEAM_ID, and cache the results in a dictionary.
        """
        if (not fpl_team_id) and (gameweek in self.fpl_team_data.keys()):
            return self.fpl_team_data[gameweek]
        if not fpl_team_id:
            fpl_team_id = self.FPL_TEAM_ID
        url = self.FPL_TEAM_URL.format(fpl_team_id, gameweek)
        fpl_team_data = self._get_request(
            url, err_msg="Unable to access FPL team API {}".format(url)
        )
        if not fpl_team_id:
            self.fpl_team_data[gameweek] = fpl_team_data
        return fpl_team_data

    def get_fpl_team_history_data(self, team_id=None):
        """
        Use our team id to get history data from the FPL API.
        """
        if self.fpl_team_history_data and not team_id:
            return self.fpl_team_history_data
        if not team_id:
            team_id = self.FPL_TEAM_ID
        url = self.FPL_HISTORY_URL.format(team_id)
        self.fpl_team_history_data = self._get_request(
            url, err_msg="Unable to access FPL team history API"
        )
        return self.fpl_team_history_data

    def get_fpl_transfer_data(self, fpl_team_id=None):
        """
        Get our transfer history from the FPL API.
        """
        if not fpl_team_id:
            fpl_team_id = self.FPL_TEAM_ID
        # return cached value if we already retrieved it.
        if (
            self.fpl_transfer_history_data
            and fpl_team_id in self.fpl_transfer_history_data.keys()
            and self.fpl_transfer_history_data[fpl_team_id] is not None
        ):
            return self.fpl_transfer_history_data[fpl_team_id]
        # or get it from the API.
        url = self.FPL_TEAM_TRANSFER_URL.format(fpl_team_id)
        # get transfer history from api and reverse order so that
        # oldest transfers at start of list and newest at end.
        self.fpl_transfer_history_data[fpl_team_id] = list(
            reversed(
                self._get_request(
                    url,
                    "Unable to access FPL transfer history API for team_id {}".format(
                        fpl_team_id
                    ),
                )
            )
        )
        return self.fpl_transfer_history_data[fpl_team_id]

    def get_fpl_league_data(self):
        """
        Use our league id to get history data from the FPL API.
        """
        if self.fpl_league_data:
            return self.fpl_league_data

        url = "https://users.premierleague.com/accounts/login/"
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
        self.rsession.post(url, data=headers)

        r = self.rsession.get(self.FPL_LEAGUE_URL, headers=headers)
        if r.status_code != 200:
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
                        player_detail = self._get_request(
                            self.FPL_DETAIL_URL.format(player_api_id),
                            "Error retrieving data for player {}".format(player_api_id),
                        )
                        if player_detail is None:
                            return []
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
        if not gameweek:
            return self.player_gameweek_data[player_api_id]

        if gameweek not in self.player_gameweek_data[player_api_id].keys():
            print(
                "Data not available for player {} week {}".format(
                    player_api_id, gameweek
                )
            )
            return []
        return self.player_gameweek_data[player_api_id][gameweek]

    def get_fixture_data(self):
        """
        Get the fixture list from the FPL API.
        """
        if not self.fixture_data:
            self.fixture_data = self._get_request(self.FPL_FIXTURE_URL)
        return self.fixture_data

    def get_transfer_deadlines(self):
        """
        Get a list of transfer deadlines.
        """
        summary_data = self._get_request(self.FPL_SUMMARY_API_URL)
        deadlines = [
            ev["deadline_time"]
            for ev in summary_data["events"]
            if "deadline_time" in ev.keys()
        ]
        return deadlines

    def get_lineup(self):
        """
        Retrieve up to date lineup from api
        """

        self.login()

        team_url = self.FPL_MYTEAM_URL.format(self.FPL_TEAM_ID)

        return self._get_request(team_url)

    def post_lineup(self, payload):
        """
        Set the lineup for a specific team
        """

        self.login()

        payload = json.dumps({"chip": None, "picks": payload})
        headers = {
            "Content-Type": "application/json; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://fantasy.premierleague.com/a/team/my",
        }

        team_url = self.FPL_MYTEAM_URL.format(self.FPL_TEAM_ID)

        resp = self.rsession.post(team_url, data=payload, headers=headers)
        if resp.status_code == 200:
            print("SUCCESS....lineup made!")
        else:
            print("Lineup changes not made due to unknown error")
            print(f"Response status code: {resp.status_code}")
            print(f"Response text: {resp.text}")

    def post_transfers(self, transfer_payload):

        self.login()

        # adapted from https://github.com/amosbastian/fpl/blob/master/fpl/utils.py
        headers = {
            "Content-Type": "application/json; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://fantasy.premierleague.com/a/squad/transfers",
        }

        transfer_url = "https://fantasy.premierleague.com/api/transfers/"

        resp = self.rsession.post(
            transfer_url, data=json.dumps(transfer_payload), headers=headers
        )
        if "non_form_errors" in resp:
            raise Exception(resp["non_form_errors"])
        elif resp.status_code == 200:
            print("SUCCESS....transfers made!")
        else:
            print("Transfers unsuccessful due to unknown error")
            print(f"Response status code: {resp.status_code}")
            print(f"Response text: {resp.text}")

    def _get_request(self, url, err_msg="Unable to access FPL API"):
        r = self.rsession.get(url)
        if r.status_code != 200:
            print(err_msg)
            return None
        return json.loads(r.content.decode("utf-8"))
