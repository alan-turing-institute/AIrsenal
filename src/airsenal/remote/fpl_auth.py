"""
Logging in to the FPL account service.

OAuth/PKCE against the FPL account service. This module owns the session every
request goes out on, the credentials, the bearer header a successful login
produces, and whether the attempt has already been made and failed.

Thanks to @Moose on the FPLDev Discord for the authentication implementation.
"""

import base64
import getpass
import hashlib
import json
import re
import secrets
import uuid

from curl_cffi import requests

from airsenal.core.env import (
    FPL_LOGIN,
    FPL_PASSWORD,
    save_env,
)
from airsenal.core.logging import get_logger
from airsenal.remote.errors import RemoteConnectionError
from airsenal.remote.fpl_http import API_HOME, Session, get_json

logger = get_logger(__name__)

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
STANDARD_CONNECTION_ID = "867ed4363b2bc21c860085ad2baa817d"


def generate_code_verifier() -> str:
    return secrets.token_urlsafe(64)[:128]


def generate_code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


class FPLAuth:
    """The session an FPL request goes out on, and how it comes to be authorised."""

    def __init__(self, session: Session | None = None) -> None:
        self.session: Session = session or requests.Session(impersonate="chrome")
        self.headers: dict[str, str] = {}
        self.logged_in = False
        self.login_failed = False
        self.FPL_LOGIN = FPL_LOGIN
        self.FPL_PASSWORD = FPL_PASSWORD

    def get_fpl_credentials(self) -> None:
        """
        If we didn't have FPL_LOGIN and FPL_PASSWORD available as files in
        AIRSENAL_HOME or as environment variables, prompt the user for them.
        """
        logger.info(
            "Accessing the most up-to-date data on your squad, or automatic "
            "transfers, requires the login (email address) and password for your "
            "FPL account."
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

    def login(self) -> None:
        """
        only needed for accessing mini-league data, or team info for current gw.

        The flow itself makes seven requests directly rather than through
        `_get_request`, so the translation to `RemoteError` happens here. Callers
        such as `squad.state.get_bank` fall back to unauthenticated data on any
        remote failure, and a failure while logging in has to be one of them.
        """
        try:
            self._login_flow()
        except requests.exceptions.RequestException as e:
            msg = "Failed to log in to the FPL API"
            raise RemoteConnectionError(msg) from e

    def _login_flow(self) -> None:
        """
        Run the OAuth/PKCE exchange, or return without doing anything.

        Returns early - leaving `logged_in` False and raising nothing - when a
        session is already authenticated, when a previous attempt failed, or when
        there are no stored credentials and the user declines the interactive
        prompt. Callers must therefore check `logged_in` rather than assume a
        clean return means success.
        """
        if self.logged_in:
            return
        if self.login_failed:
            logger.warning(
                "Attempted to use a function requiring login, but login previously "
                "failed."
            )
            return
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
                self._set_login_failed(msg="Credentials not provided.")
                return

        code_verifier = generate_code_verifier()  # code_verifier for PKCE
        code_challenge = generate_code_challenge(
            code_verifier
        )  # code_challenge from the code_verifier
        initial_state = uuid.uuid4().hex  # random initial state for the OAuth flow

        # Step 1: Request authorization page
        params = {
            "client_id": CLIENT_ID,
            "redirect_uri": "https://fantasy.premierleague.com/",
            "response_type": "code",
            "scope": "openid profile email offline_access",
            "state": initial_state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        auth_response = self.session.get(LOGIN_URLS["auth"], params=params)
        login_html = auth_response.text

        if match := re.search(r'"accessToken":"([^"]+)"', login_html):
            access_token = match.group(1)
        else:
            self._set_login_failed(msg="Failed to extract access token.")
            return
        # need to read state here for when we resume the OAuth flow later on
        if match := re.search(
            r'<input[^>]+name="state"[^>]+value="([^"]+)"', login_html
        ):
            new_state = match.group(1)
        else:
            self._set_login_failed(msg="Failed to extract state.")
            return

        # Step 2: Use accessToken to get interaction id
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        response = self.session.post(LOGIN_URLS["start"], headers=headers)
        try:
            r_json = response.json()
            interaction_id = r_json["interactionId"]
            response_id = r_json["id"]
        except (json.JSONDecodeError, KeyError) as e:
            self._set_login_failed(exception=e, msg="Failed to extract interaction ID.")
            return

        # Step 3: log in with interaction ID (requires 3 post requests)
        response = self.session.post(
            LOGIN_URLS["login"].format(STANDARD_CONNECTION_ID),
            headers={
                "interactionId": interaction_id,
            },
            json={
                "id": response_id,
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
        try:
            response_id = response.json()["id"]
        except (json.JSONDecodeError, KeyError) as e:
            self._set_login_failed(
                exception=e, msg="Interaction Post 1 Failed (id generation)"
            )
            return

        response = self.session.post(
            LOGIN_URLS["login"].format(STANDARD_CONNECTION_ID),
            headers={
                "interactionId": interaction_id,
            },
            json={
                "id": response_id,
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
        )
        try:
            r_json = response.json()
            response_id = r_json["id"]
            connection_id = r_json["connectionId"]
        except (json.JSONDecodeError, KeyError) as e:
            self._set_login_failed(
                exception=e,
                msg="Interaction Post 2 Failed (connectionID generation)",
            )
            return

        response = self.session.post(
            LOGIN_URLS["login"].format(connection_id),
            headers=headers,
            json={
                "id": response_id,
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
        try:
            dv_response = response.json()["dvResponse"]
        except (json.JSONDecodeError, KeyError) as e:
            self._set_login_failed(
                exception=e,
                msg="Interaction Post 3 Failed (dvResponse generation)",
            )
            return

        # Step 4: Resume the login using the dv_response and handle redirect
        response = self.session.post(
            LOGIN_URLS["resume"],
            data={"dvResponse": dv_response, "state": new_state},
            allow_redirects=False,
        )
        if (location := response.headers.get("Location")) and (
            match := re.search(r"[?&]code=([^&]+)", location)
        ):
            auth_code = match.group(1)
        else:
            self._set_login_failed(msg="Failed to extract auth code.")
            return

        # Step 5: Exchange auth code for access token
        response = self.session.post(
            LOGIN_URLS["token"],
            data={
                "grant_type": "authorization_code",
                "redirect_uri": "https://fantasy.premierleague.com/",
                "code": auth_code,  # from the parsed redirect URL
                "code_verifier": code_verifier,  # code_verifier generated at the start
                "client_id": CLIENT_ID,
            },
        )
        try:
            access_token = response.json()["access_token"]
        except (json.JSONDecodeError, KeyError) as e:
            self._set_login_failed(exception=e, msg="Failed to retrieve access token.")
            return

        self.headers = {"X-API-Authorization": f"Bearer {access_token}"}
        response = get_json(self.session, LOGIN_URLS["me"], headers=self.headers)
        if "player" in response:
            self.logged_in = True
        else:
            self._set_login_failed(
                msg="All login steps succeeded but team data retrieval failed."
            )
            return

    def _set_login_failed(
        self, exception: Exception | None = None, msg: str = ""
    ) -> None:
        self.login_failed = True
        help = (
            "Login failed due to the error above. Continuing without login but this "
            "may cause issues later due to not having your latest team details. Login "
            "failures could be caused by issues with your username and password, "
            "connection problems, or changes to the API."
        )
        logger.warning("%s\n%s", msg, help, exc_info=exception)
