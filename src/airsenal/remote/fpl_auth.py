"""
Logging in to the FPL account service, by OAuth/PKCE.

This module owns the session every request goes out on, the credentials, the
bearer header a successful login produces, and whether an attempt has already
been made and failed.

Thanks to @Moose on the FPLDev Discord for the authentication implementation.
"""

import base64
import getpass
import hashlib
import json
import re
import secrets
import uuid
from typing import Any

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
REDIRECT_URI = "https://fantasy.premierleague.com/"
_NEXT_EVENT = {
    "constructType": "skEvent",
    "eventName": "continue",
    "params": [],
    "eventType": "post",
    "postProcess": {},
}


def generate_code_verifier() -> str:
    return secrets.token_urlsafe(64)[:128]


def generate_code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


class _LoginStepError(Exception):
    """A login step that could not go on, and the message saying which."""


def _ask_yes_no(prompt: str) -> bool:
    """Ask until the answer is y or n, and return whether it was y."""
    answer = ""
    while answer not in ("y", "n"):
        answer = input(prompt).lower()
    return answer == "y"


def _json_fields(response: Any, *keys: str, msg: str) -> list[Any]:
    """The named fields of a JSON response, or a failed step saying `msg`."""
    try:
        body = response.json()
        return [body[key] for key in keys]
    except (json.JSONDecodeError, KeyError) as e:
        raise _LoginStepError(msg) from e


def _search(pattern: str, text: str, msg: str) -> str:
    """The first group of `pattern` in `text`, or a failed step saying `msg`."""
    if match := re.search(pattern, text):
        return match.group(1)
    raise _LoginStepError(msg)


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
        """Prompt for FPL_LOGIN and FPL_PASSWORD, and offer to save them."""
        logger.info(
            "Accessing the most up-to-date data on your squad, or automatic "
            "transfers, requires the login (email address) and password for your "
            "FPL account."
        )

        self.FPL_LOGIN = input("Please enter FPL login: ")
        self.FPL_PASSWORD = getpass.getpass("Please enter FPL password: ")
        if _ask_yes_no(
            "\nWould you like to store these credentials so that"
            " you won't be prompted for them again? (y/n): "
        ):
            save_env("FPL_LOGIN", self.FPL_LOGIN)
            save_env("FPL_PASSWORD", self.FPL_PASSWORD)

    def login(self) -> None:
        """Log in to the FPL API, or raise a RemoteConnectionError if it fails."""
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
        try:
            self._ensure_credentials()
            code_verifier = generate_code_verifier()
            page_token, state = self._authorise(code_verifier)
            dv_response = self._sign_on(page_token)
            auth_code = self._resume(dv_response, state)
            access_token = self._token(auth_code, code_verifier)
            self.headers = {"X-API-Authorization": f"Bearer {access_token}"}
            self._check_team_access()
        except _LoginStepError as e:
            self._set_login_failed(exception=e.__cause__, msg=str(e))
            return
        self.logged_in = True

    def _ensure_credentials(self) -> None:
        """Prompt for credentials if none are stored and the user wants to log in."""
        if self.FPL_LOGIN and self.FPL_PASSWORD:
            return
        if not _ask_yes_no(
            "\nWould you like to login to the FPL API?"
            "\nThis is not necessary for most AIrsenal actions, "
            "\nbut may improve accuracy of player sell values,"
            "\nand free transfers for your team, and will also "
            "\nenable AIrsenal to make transfers for you through "
            "\nthe API. (y/n): "
        ):
            msg = "Credentials not provided."
            raise _LoginStepError(msg)
        self.get_fpl_credentials()

    def _authorise(self, code_verifier: str) -> tuple[str, str]:
        """Request the authorisation page, and read its access token and state."""
        params = {
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "response_type": "code",
            "scope": "openid profile email offline_access",
            "state": uuid.uuid4().hex,
            "code_challenge": generate_code_challenge(code_verifier),
            "code_challenge_method": "S256",
        }
        login_html = self.session.get(LOGIN_URLS["auth"], params=params).text
        access_token = _search(
            r'"accessToken":"([^"]+)"', login_html, "Failed to extract access token."
        )
        # The state is needed to resume the OAuth flow after signing on.
        state = _search(
            r'<input[^>]+name="state"[^>]+value="([^"]+)"',
            login_html,
            "Failed to extract state.",
        )
        return access_token, state

    def _sign_on(self, access_token: str) -> str:
        """Sign on in an interaction the access token starts; return its dvResponse."""
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        interaction_id, response_id = _json_fields(
            self.session.post(LOGIN_URLS["start"], headers=headers),
            "interactionId",
            "id",
            msg="Failed to extract interaction ID.",
        )
        login_url = LOGIN_URLS["login"].format(STANDARD_CONNECTION_ID)
        (response_id,) = _json_fields(
            self.session.post(
                login_url,
                headers={"interactionId": interaction_id},
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
            ),
            "id",
            msg="Interaction Post 1 Failed (id generation)",
        )
        response_id, connection_id = _json_fields(
            self.session.post(
                login_url,
                headers={"interactionId": interaction_id},
                json={
                    "id": response_id,
                    "nextEvent": _NEXT_EVENT,
                    "parameters": {
                        "buttonType": "form-submit",
                        "buttonValue": "SIGNON",
                        "username": self.FPL_LOGIN,
                        "password": self.FPL_PASSWORD,
                    },
                    "eventName": "continue",
                },
            ),
            "id",
            "connectionId",
            msg="Interaction Post 2 Failed (connectionID generation)",
        )
        dv_response: str = _json_fields(
            self.session.post(
                LOGIN_URLS["login"].format(connection_id),
                headers=headers,
                json={
                    "id": response_id,
                    "nextEvent": _NEXT_EVENT,
                    "parameters": {
                        "buttonType": "form-submit",
                        "buttonValue": "SIGNON",
                    },
                    "eventName": "continue",
                },
            ),
            "dvResponse",
            msg="Interaction Post 3 Failed (dvResponse generation)",
        )[0]
        return dv_response

    def _resume(self, dv_response: str, state: str) -> str:
        """Resume the OAuth flow, and read the auth code from its redirect."""
        response = self.session.post(
            LOGIN_URLS["resume"],
            data={"dvResponse": dv_response, "state": state},
            allow_redirects=False,
        )
        return _search(
            r"[?&]code=([^&]+)",
            response.headers.get("Location") or "",
            "Failed to extract auth code.",
        )

    def _token(self, auth_code: str, code_verifier: str) -> str:
        """Exchange the auth code for an access token."""
        response = self.session.post(
            LOGIN_URLS["token"],
            data={
                "grant_type": "authorization_code",
                "redirect_uri": REDIRECT_URI,
                "code": auth_code,
                "code_verifier": code_verifier,
                "client_id": CLIENT_ID,
            },
        )
        access_token: str = _json_fields(
            response, "access_token", msg="Failed to retrieve access token."
        )[0]
        return access_token

    def _check_team_access(self) -> None:
        """Check the new header can read the entry's own data."""
        response = get_json(self.session, LOGIN_URLS["me"], headers=self.headers)
        if "player" not in response:
            msg = "All login steps succeeded but team data retrieval failed."
            raise _LoginStepError(msg)

    def _set_login_failed(
        self, exception: BaseException | None = None, msg: str = ""
    ) -> None:
        self.login_failed = True
        help = (
            "Login failed due to the error above. Continuing without login but this "
            "may cause issues later due to not having your latest team details. Login "
            "failures could be caused by issues with your username and password, "
            "connection problems, or changes to the API."
        )
        logger.warning("%s\n%s", msg, help, exc_info=exception)
