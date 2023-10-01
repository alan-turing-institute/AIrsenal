"""
Classes to query the FPL API to retrieve current FPL data,
and to query football-data.org to retrieve match and fixture data.
"""
import getpass
import json
import time
import warnings

import requests

from airsenal.framework.env import get_env, save_env

API_HOME = "https://fantasy.premierleague.com/api"


class FPLDataFetcher(object):
    """
    hold current and historic FPL data in memory,
    or retrieve it if not already cached.
    """

    def __init__(self, fpl_team_id=None, rsession=None):
        self.rsession = rsession or requests.session()
        self.logged_in = False
        self.login_failed = False
        self.continue_without_login = False
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
            self.__setattr__(
                ID,
                get_env(ID, default="MISSING_ID"),
            )

        if self.FPL_TEAM_ID is not None and self.FPL_TEAM_ID != "MISSING_ID":
            try:
                self.FPL_TEAM_ID = int(self.FPL_TEAM_ID)
            except ValueError as e:
                raise ValueError(
                    f"FPL_TEAM_ID in environment variable and/or data/FPL_TEAM_ID "
                    f"file should be a valid integer. Please correct it or remove "
                    f" it if you're using the command line argument. "
                    f"Found: {self.FPL_TEAM_ID}"
                ) from e

        if fpl_team_id is not None:
            if isinstance(fpl_team_id, int):
                self.FPL_TEAM_ID = fpl_team_id  # update entry with command line arg
            else:
                raise ValueError(
                    f"FPL_TEAM_ID should be an integer. Found: {fpl_team_id}"
                )
        self.FPL_SUMMARY_API_URL = f"{API_HOME}/bootstrap-static/"
        self.FPL_DETAIL_URL = API_HOME + "/element-summary/{}/"
        self.FPL_HISTORY_URL = API_HOME + "/entry/{}/history/"
        self.FPL_TEAM_URL = API_HOME + "/entry/{}/event/{}/picks/"
        self.FPL_TEAM_TRANSFER_URL = API_HOME + "/entry/{}/transfers/"
        self.FPL_LEAGUE_URL = (
            f"{API_HOME}/leagues-classic/{self.FPL_LEAGUE_ID}"
            "/standings/?page_new_entries=1&page_standings=1"
        )
        self.FPL_FIXTURE_URL = f"{API_HOME}/fixtures/"
        self.FPL_LOGIN_URL = "https://users.premierleague.com/accounts/login/"
        self.FPL_LOGIN_REDIRECT_URL = "https://fantasy.premierleague.com/a/login"
        self.FPL_MYTEAM_URL = API_HOME + "/my-team/{}/"

    def get_fpl_credentials(self):
        """
        If we didn't have FPL_LOGIN and FPL_PASSWORD available as files in
        AIRSENAL_HOME or as environment variables, prompt the user for them.
        """
        print(
            """
            Accessing the most up-to-date data on your squad, or automatic transfers,
            requires the login (email address) and password for your FPL account.
            """
        )

        self.FPL_LOGIN = input("Please enter FPL login: ")
        self.FPL_PASSWORD = getpass.getpass("Please enter FPL password: ")
        store_credentials = ""
        while store_credentials.lower() not in ["y", "n"]:
            store_credentials = input(
                "\nWould you like to store these credentials so that"
                " you won't be prompted for them again? (y/n): "
            )

        if store_credentials.lower() == "y":
            save_env("FPL_LOGIN", self.FPL_LOGIN)
            save_env("FPL_PASSWORD", self.FPL_PASSWORD)

    def login(self, attempts=3):
        """
        only needed for accessing mini-league data, or team info for current gw.
        """
        if self.logged_in or self.continue_without_login:
            return
        if self.login_failed:
            raise RuntimeError(
                "Attempted to use a function requiring login, but login previously "
                "failed."
            )
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
                self.login_failed = True
                self.continue_without_login = True
                warnings.warn(
                    "Skipping login which means AIrsenal may have out of date "
                    "information for your team."
                )
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
        tried = 0
        while tried < attempts:
            print(f"Login attempt {tried+1}/{attempts}...", end=" ")
            response = self.rsession.post(
                self.FPL_LOGIN_URL, data=data, headers=headers
            )
            if response.status_code == 200:
                print("Logged in successfully")
                self.logged_in = True
                return
            print("Failed")
            tried += 1
            time.sleep(1)
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            self.login_failed = True
            raise requests.HTTPError(f"Error logging in to FPL API: {e}")

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
        Requires login
        """
        squad_data = self.get_current_squad_data(fpl_team_id)
        return [
            chip["name"]
            for chip in squad_data["chips"]
            if chip["status_for_entry"] == "available"
        ]

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
            url, err_msg=f"Unable to access FPL team API {url}"
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
                    (
                        "Unable to access FPL transfer history API for "
                        f"team_id {fpl_team_id}"
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
                player_detail = self._get_request(
                    self.FPL_DETAIL_URL.format(player_api_id),
                    f"Error retrieving data for player {player_api_id}",
                )
                for game in player_detail["history"]:
                    gw = game["round"]
                    if gw not in self.player_gameweek_data[player_api_id].keys():
                        self.player_gameweek_data[player_api_id][gw] = []
                    self.player_gameweek_data[player_api_id][gw].append(game)
        if not gameweek:
            return self.player_gameweek_data[player_api_id]

        if gameweek not in self.player_gameweek_data[player_api_id].keys():
            print(f"Data not available for player {player_api_id} week {gameweek}")
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
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            raise requests.HTTPError(
                f"{e}\nLineup changes not made due to the error above! Make the "
                "changes manually on the web-site if needed."
            )
        if resp.status_code == 200:
            print("SUCCESS....lineup made!")
            return
        raise Exception(
            f"Unexpected error in post_lineup: "
            f"code={resp.status_code}, content={resp.content}"
        )

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
            raise requests.RequestException(
                f"{resp['non_form_errors']}\nMaking transfers failed due to the "
                "error above! Make the changes manually on the web-site if needed."
            )
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            raise requests.HTTPError(
                f"{e}\nMaking transfers failed due to the error above! Make the "
                "changes manually on the web-site if needed."
            )
        if resp.status_code == 200:
            print("SUCCESS....transfers made!")
            return
        raise Exception(
            f"Unexpected error in post_transfers: "
            f"code={resp.status_code}, content={resp.content}"
        )

    def _get_request(self, url, err_msg="Unable to access FPL API", attempts=3):
        tries = 0
        while tries < attempts:
            try:
                r = self.rsession.get(url)
                break
            except requests.exceptions.ConnectionError as e:
                tries += 1
                if tries == attempts:
                    raise requests.exceptions.ConnectionError(
                        f"{err_msg}: Failed to connect to FPL API when requesting {url}"
                    ) from e
                time.sleep(1)

        if r.status_code == 200:
            return json.loads(r.content.decode("utf-8"))

        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            raise requests.HTTPError(f"{err_msg}: {e}") from e
        raise Exception(
            f"Unexpected error in _get_request to {url}: "
            f"code={r.status_code}, content={r.content}"
        )
