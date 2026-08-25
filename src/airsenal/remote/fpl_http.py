"""
Talking to the FPL API: where it lives, and how a request to it is made.

The retry, and the translation of a transport failure into a `RemoteError`, are
not about being a data fetcher - the login flow needs them too, and it is the one
caller whose failures every `except RemoteError` fallback in squad/ and pipeline/
depends on. So they are functions over a session rather than methods on the
client, and both `FPLDataFetcher` and `FPLAuth` reach the API through them.
"""

import json
import time
from typing import Any

from curl_cffi import requests

from airsenal.remote.errors import (
    RemoteConnectionError,
    RemoteError,
    RemoteHTTPError,
)

API_HOME = "https://fantasy.premierleague.com/api"

# The type of a curl_cffi Session. Any, because curl_cffi ships py.typed but
# leaves the methods this module calls unannotated.
type Session = Any


def get_json(
    session: Session,
    url: str,
    headers: dict[str, str] | None = None,
    err_msg: str = "Unable to access FPL API",
    attempts: int = 3,
    **params: Any,
) -> Any:
    # Any, not a payload type: this is the decoded body of an untyped external
    # API, and every caller narrows it for its own endpoint.
    tries = 0
    r = None
    while tries < attempts:
        try:
            r = session.get(url, headers=headers or {}, params=params)
            break
        except requests.exceptions.ConnectionError as e:
            tries += 1
            if tries == attempts:
                msg = f"{err_msg}: Failed to connect to FPL API when requesting {url}"
                raise RemoteConnectionError(msg) from e
            time.sleep(1)

    if r is None:
        msg = f"{err_msg}: Failed to connect to FPL API when requesting {url}"
        raise RemoteConnectionError(msg)

    if r.status_code == 200:
        return json.loads(r.content.decode("utf-8"))

    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        msg = f"{err_msg}: {e}"
        raise RemoteHTTPError(msg, r.status_code) from e
    msg = (
        f"Unexpected error requesting {url}: "
        f"code={r.status_code}, content={r.content.decode('utf-8')}"
    )
    raise RemoteError(msg)


def post_json(
    session: Session,
    url: str,
    data: Any,
    headers: dict[str, str] | None = None,
    err_msg: str = "Failed to post data to FPL API",
) -> None:
    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
        **(headers or {}),
    }
    resp = session.post(url, json=data, headers=headers)
    if "non_form_errors" in resp.text or "non_field_errors" in resp.text:
        msg = f"{resp.text}\n{err_msg}"
        raise RemoteError(msg)
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        msg = f"{err_msg}: {e} {resp.text}"
        raise RemoteHTTPError(msg, resp.status_code) from e
