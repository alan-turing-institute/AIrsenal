"""
The distinction the remote error hierarchy exists to preserve.

`get_entry_start_gameweek` walks forward until the API has picks for a gameweek.
It has to tell "this gameweek is a 404, ask about the next one" from "there is no
network, stop asking" - if those merge, a total outage looks like "this team was
entered in GW1", which is the answer that makes the pipeline rebuild a squad that
already exists. Nothing covered that before, and a single flat error class would
have passed every other test in the suite.
"""

import pytest
from curl_cffi import requests

from airsenal.remote.errors import (
    RemoteConnectionError,
    RemoteError,
    RemoteHTTPError,
)
from airsenal.remote.fpl_http import get_json


def test_every_remote_error_is_catchable_as_one():
    # The five "any failure -> fall back to the DB" sites catch only the base.
    assert issubclass(RemoteConnectionError, RemoteError)
    assert issubclass(RemoteHTTPError, RemoteError)


def test_remote_error_is_a_runtime_error():
    # Matches ConfigError/NoFixtureDataError, and means a bare `except RuntimeError`
    # upstream keeps working.
    assert issubclass(RemoteError, RuntimeError)


def test_status_error_carries_the_status():
    err = RemoteHTTPError("nope", 404)
    assert err.status_code == 404


def test_a_connection_failure_is_not_mistaken_for_a_status():
    assert not isinstance(RemoteConnectionError("down"), RemoteHTTPError)


class _Boom:
    """A session whose every request fails the way `exc` says."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def get(self, *args: object, **kwargs: object) -> object:
        raise self.exc


def test_get_request_translates_a_connection_failure():
    session = _Boom(requests.exceptions.ConnectionError("x"))
    # attempts=1 so the retry loop does not sleep.
    with pytest.raises(RemoteConnectionError):
        get_json(session, "https://example.com/x", attempts=1)


class _Status:
    def __init__(self, code: int) -> None:
        self.status_code = code
        self.content = b"{}"
        self.text = ""

    def get(self, *args: object, **kwargs: object) -> "_Status":
        return self

    def raise_for_status(self) -> None:
        msg = f"{self.status_code}"
        raise requests.exceptions.HTTPError(msg)


def test_get_request_translates_a_bad_status_and_keeps_the_code():
    with pytest.raises(RemoteHTTPError) as excinfo:
        get_json(_Status(404), "https://example.com/x", attempts=1)
    assert excinfo.value.status_code == 404


def test_a_200_is_not_an_error():
    assert get_json(_Status(200), "https://example.com/x", attempts=1) == {}
