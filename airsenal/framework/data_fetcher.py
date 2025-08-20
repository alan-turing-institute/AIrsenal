"""
Classes to query the FPL API.

Thanks to:
- @Moose on the FPLDev Discord for the authentication implementation.
- https://github.com/amosbastian/fpl/blob/master/fpl/utils.py for posting transfers and
  lineups.
"""

import base64
import getpass
import hashlib
import json
import re
import secrets
import time
import uuid
import warnings

from curl_cffi import requests

from airsenal.framework.env import (
    DISCORD_WEBHOOK,
    FPL_LEAGUE_ID,
    FPL_LOGIN,
    FPL_PASSWORD,
    FPL_TEAM_ID,
    save_env,
)

API_HOME = "https://fantasy.premierleague.com/api"

LOGIN_BASE = "https://account.premierleague.com"
LOGIN_URLS = {
    "auth": f"{LOGIN_BASE}/as/authorize",
    "start": f"{LOGIN_BASE}/davinci/policy/262ce4b01d19dd9d385d26bddb4297b6/start",
    "login": f"{LOGIN_BASE}/davinci/connections/{{}}/capabilities/customHTMLTemplate",
    "resume": f"{LOGIN_BASE}/as/resume",
    "token": f"{LOGIN_BASE}/as/token",
    "me": f"{API_HOME}/me/",
}

CLIENT_ID = "bfcbaf69-aade-4c1b-8f00-c1cb8a193030"
STANDARD_CONNECTION_ID = "0d8c928e4970386733ce110b9dda8412"


def generate_code_verifier():
    return secrets.token_urlsafe(64)[:128]


def generate_code_challenge(verifier):
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


class FPLDataFetcher:
    """
    hold current and historic FPL data in memory,
    or retrieve it if not already cached.
    """

    def __init__(self, fpl_team_id: int | None = None, rsession=None):
        self.rsession = rsession or requests.Session(impersonate="chrome")
        self.headers: dict[str, str] = {}
        self.logged_in = False
        self.login_failed = False
        self.continue_without_login = False
        self.current_summary_data: dict = {}
        self.current_event_data: dict = {}
        self.current_player_data: dict = {}
        self.current_team_data: dict = {}
        self.current_squad_data: dict = {}
        self.player_gameweek_data: dict = {}
        self.fpl_team_history_data: dict = {}
        # transfer history data is a dict, keyed by fpl_team_id
        self.fpl_transfer_history_data: dict = {}
        self.fpl_league_data: dict = {}
        self.fpl_team_data: dict = {}  # players in squad, by gameweek
        self.fixture_data: dict = {}

        self.FPL_TEAM_ID = FPL_TEAM_ID if fpl_team_id is None else fpl_team_id
        self.FPL_LOGIN = FPL_LOGIN
        self.FPL_PASSWORD = FPL_PASSWORD
        self.FPL_LEAGUE_ID = FPL_LEAGUE_ID
        self.DISCORD_WEBHOOK = DISCORD_WEBHOOK

        self.FPL_SUMMARY_API_URL = f"{API_HOME}/bootstrap-static/"
        self.FPL_DETAIL_URL = API_HOME + "/element-summary/{}/"
        self.FPL_HISTORY_URL = API_HOME + "/entry/{}/history/"
        self.FPL_TEAM_URL = API_HOME + "/entry/{}/event/{}/picks/"
        self.FPL_GET_TRANSFERS_URL = API_HOME + "/entry/{}/transfers/"
        self.FPL_SET_TRANSFERS_URL = API_HOME + "/transfers/"
        self.FPL_LEAGUE_URL = (
            f"{API_HOME}/leagues-classic/{self.FPL_LEAGUE_ID}"
            "/standings/?page_new_entries=1&page_standings=1"
        )
        self.FPL_FIXTURE_URL = f"{API_HOME}/fixtures/"
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

    def login(self):
        """
        only needed for accessing mini-league data, or team info for current gw.
        """
        if self.logged_in or self.continue_without_login:
            return
        if self.login_failed:
            msg = (
                "Attempted to use a function requiring login, but login previously "
                "failed."
            )
            raise RuntimeError(msg)
        if (not self.FPL_LOGIN) or (not self.FPL_PASSWORD):
            do_login = ""
            while do_login.lower() not in ["y", "n"]:
                do_login = input(
                    "\nWould you like to login to the FPL API?"
                    "\nThis is not necessary for most AIrsenal actions, "
                    "\nbut may improve accuracy of player sell values,"
                    "\nand free transfers for your team, and will also "
                    "\nenable AIrsenal to make transfers for you through "
                    "\nthe API. (y/n): "
                )
            if do_login.lower() == "y":
                self.get_fpl_credentials()
            else:
                self.login_failed = True
                self.continue_without_login = True
                warnings.warn(
                    "Skipping login which means AIrsenal may have out of date "
                    "information for your team.",
                    stacklevel=2,
                )
                return

        code_verifier = generate_code_verifier()  # code_verifier for PKCE
        code_challenge = generate_code_challenge(
            code_verifier
        )  # code_challenge from the code_verifier
        initial_state = uuid.uuid4().hex  # random initial state for the OAuth flow

        # Step 1: Request authorization page
        params = {
            "client_id": "bfcbaf69-aade-4c1b-8f00-c1cb8a193030",
            "redirect_uri": "https://fantasy.premierleague.com/",
            "response_type": "code",
            "scope": "openid profile email offline_access",
            "state": initial_state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        auth_response = self.rsession.get(LOGIN_URLS["auth"], params=params)
        login_html = auth_response.text

        if match := re.search(r'"accessToken":"([^"]+)"', login_html):
            access_token = match.group(1)
        else:
            msg = "Login failed. Failed to extract access token."
            raise RuntimeError(msg)
        # need to read state here for when we resume the OAuth flow later on
        if match := re.search(
            r'<input[^>]+name="state"[^>]+value="([^"]+)"', login_html
        ):
            new_state = match.group(1)
        else:
            msg = "Login failed. Failed to extract state."
            raise RuntimeError(msg)

        # Step 2: Use accessToken to get interaction id and token
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        response = self.rsession.post(LOGIN_URLS["start"], headers=headers).json()
        interaction_id = response["interactionId"]
        interaction_token = response["interactionToken"]

        # Step 3: log in with interaction tokens (requires 3 post requests)
        response = self.rsession.post(
            LOGIN_URLS["login"].format(STANDARD_CONNECTION_ID),
            headers={
                "interactionId": interaction_id,
                "interactionToken": interaction_token,
            },
            json={
                "id": response["id"],
                "eventName": "continue",
                "parameters": {"eventType": "polling"},
                "pollProps": {
                    "status": "continue",
                    "delayInMs": 10,
                    "retriesAllowed": 1,
                    "pollChallengeStatus": False,
                },
            },
        )

        response = self.rsession.post(
            LOGIN_URLS["login"].format(STANDARD_CONNECTION_ID),
            headers={
                "interactionId": interaction_id,
                "interactionToken": interaction_token,
            },
            json={
                "id": response.json()["id"],
                "nextEvent": {
                    "constructType": "skEvent",
                    "eventName": "continue",
                    "params": [],
                    "eventType": "post",
                    "postProcess": {},
                },
                "parameters": {
                    "buttonType": "form-submit",
                    "buttonValue": "SIGNON",
                    "username": self.FPL_LOGIN,
                    "password": self.FPL_PASSWORD,
                },
                "eventName": "continue",
            },
        ).json()

        response = self.rsession.post(
            LOGIN_URLS["login"].format(
                response["connectionId"]
            ),  # need to use new connectionId from prev response
            headers=headers,
            json={
                "id": response["id"],
                "nextEvent": {
                    "constructType": "skEvent",
                    "eventName": "continue",
                    "params": [],
                    "eventType": "post",
                    "postProcess": {},
                },
                "parameters": {
                    "buttonType": "form-submit",
                    "buttonValue": "SIGNON",
                },
                "eventName": "continue",
            },
        )

        # Step 4: Resume the login using the dv_response and handle redirect
        response = self.rsession.post(
            LOGIN_URLS["resume"],
            data={
                "dvResponse": response.json()["dvResponse"],
                "state": new_state,
            },
            allow_redirects=False,
        )

        if (location := response.headers.get("Location")) and (
            match := re.search(r"[?&]code=([^&]+)", location)
        ):
            auth_code = match.group(1)
        else:
            msg = "Login failed. Failed to extract auth code."
            raise RuntimeError(msg)

        # Step 5: Exchange auth code for access token
        response = self.rsession.post(
            LOGIN_URLS["token"],
            data={
                "grant_type": "authorization_code",
                "redirect_uri": "https://fantasy.premierleague.com/",
                "code": auth_code,  # from the parsed redirect URL
                "code_verifier": code_verifier,  # code_verifier generated at the start
                "client_id": "bfcbaf69-aade-4c1b-8f00-c1cb8a193030",
            },
        )

        access_token = response.json()["access_token"]
        self.headers = {"X-API-Authorization": f"Bearer {access_token}"}
        response = self._get_request(LOGIN_URLS["me"])
        if "player" in response:
            self.logged_in = True
        else:
            self.login_failed = True

    def get_current_squad_data(self, fpl_team_id=None):
        """
        Requires login.  Return the current squad data, including
        "picks", bank, and free transfers.
        """
        if fpl_team_id is None:
            if self.FPL_TEAM_ID is None:
                msg = "Please specify FPL team ID"
                raise RuntimeError(msg)
            fpl_team_id = self.FPL_TEAM_ID

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
        if (not fpl_team_id) and (gameweek in self.fpl_team_data):
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
            and fpl_team_id in self.fpl_transfer_history_data
            and self.fpl_transfer_history_data[fpl_team_id] is not None
        ):
            return self.fpl_transfer_history_data[fpl_team_id]
        # or get it from the API.
        url = self.FPL_GET_TRANSFERS_URL.format(fpl_team_id)
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

        self.login()
        r = self._get_request(self.FPL_LEAGUE_URL)
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
        if player_api_id not in self.player_gameweek_data:
            self.player_gameweek_data[player_api_id] = {}
            if (not gameweek) or (
                gameweek not in self.player_gameweek_data[player_api_id]
            ):
                player_detail = self._get_request(
                    self.FPL_DETAIL_URL.format(player_api_id),
                    f"Error retrieving data for player {player_api_id}",
                )
                for game in player_detail["history"]:
                    gw = game["round"]
                    if gw not in self.player_gameweek_data[player_api_id]:
                        self.player_gameweek_data[player_api_id][gw] = []
                    self.player_gameweek_data[player_api_id][gw].append(game)
        if not gameweek:
            return self.player_gameweek_data[player_api_id]

        if gameweek not in self.player_gameweek_data[player_api_id]:
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
        return [
            ev["deadline_time"]
            for ev in summary_data["events"]
            if "deadline_time" in ev
        ]

    def get_lineup(self):
        """
        Retrieve up to date lineup from api
        """
        self.login()
        team_url = self.FPL_MYTEAM_URL.format(self.FPL_TEAM_ID)
        return self._get_request(team_url)

    def post_lineup(self, payload):
        """Set the lineup for a specific team"""
        self.login()
        payload = json.dumps({"chip": None, "picks": payload})
        team_url = self.FPL_MYTEAM_URL.format(self.FPL_TEAM_ID)
        self._post_data(
            team_url,
            payload,
            err_msg=(
                "Failed to set lineup. Make the changes manually on the web-site if "
                "needed"
            ),
        )
        print("Lineup set!")

    def post_transfers(self, transfer_payload):
        """Make transfers via the API.

        WARNING: This can't be undone and may incur points hits. It also doesn't support
        activating chips currently, so this must be done manually especially if you are
        using a wildcard or free hit chip (in which case the transfers will be applied
        as normal transfers with points hits).
        """
        self.login()
        err_msg = (
            "Failed to set transfers. Make the changes manually on the web-site if "
            "needed."
        )
        resp = self._post_data(
            self.FPL_SET_TRANSFERS_URL,
            data=transfer_payload,
            err_msg=err_msg,
        )
        if "non_form_errors" in resp.json():
            msg = f"{resp.json()['non_form_errors']}\n{err_msg}"
            raise requests.exceptions.RequestException(msg)
        print("Transfers made!")

    def _get_request(
        self, url, err_msg="Unable to access FPL API", attempts=3, **params
    ):
        tries = 0
        r = None
        while tries < attempts:
            try:
                r = self.rsession.get(url, headers=self.headers, params=params)
                break
            except requests.exceptions.ConnectionError as e:
                tries += 1
                if tries == attempts:
                    msg = (
                        f"{err_msg}: Failed to connect to FPL API when requesting {url}"
                    )
                    raise requests.exceptions.ConnectionError(msg) from e
                time.sleep(1)

        if r is None:
            msg = f"{err_msg}: Failed to connect to FPL API when requesting {url}"
            raise RuntimeError(msg)

        if r.status_code == 200:
            return json.loads(r.content.decode("utf-8"))

        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            msg = f"{err_msg}: {e}"
            raise requests.exceptions.HTTPError(msg) from e
        msg = (
            f"Unexpected error in _get_request to {url}: "
            f"code={r.status_code}, content={r.content.decode('utf-8')}"
        )
        raise RuntimeError(msg)

    def _post_data(self, url, data, err_msg="Failed to post data to FPL API"):
        headers = {
            "Content-Type": "application/json; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
            **self.headers,
        }
        resp = self.rsession.post(url, json=data, headers=headers)
        if resp.status_code == 200:
            return json.loads(resp.content.decode("utf-8"))
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            msg = f"{err_msg}: {e}"
            raise requests.exceptions.HTTPError(msg) from e
        msg = (
            f"{err_msg} Unexpected error in _post_request to {url}: "
            f"code={resp.status_code}, content={resp.content.decode('utf-8')}"
        )
        raise RuntimeError(msg)
