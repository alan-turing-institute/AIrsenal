"""
Fetching a Transfermarkt page.

Only `_get` is covered: everything above it parses HTML this repo has no recorded
copy of, but every one of those functions goes through this, so the timeout and
the error translation are worth pinning.
"""

import pytest
import requests

from airsenal.remote.errors import RemoteConnectionError, RemoteHTTPError
from airsenal.remote.transfermarkt import TIMEOUT_SECONDS, _get


class FakeResponse:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code
        self.content = b"<html></html>"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            msg = str(self.status_code)
            raise requests.exceptions.HTTPError(msg)


def test_a_request_is_given_a_timeout(monkeypatch):
    """
    `requests` waits for ever by default, so the timeout has to be passed.

    A scrape is three pages per player across a whole division; one stalled
    connection with no timeout stops the run with nothing to show for it.
    """
    calls = {}

    def fake_get(url, **kwargs):
        calls.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)
    _get("https://www.transfermarkt.co.uk/anything")

    assert calls["timeout"] == TIMEOUT_SECONDS


def test_an_unreachable_site_is_a_remote_connection_error(monkeypatch):
    def fake_get(url, **kwargs):
        msg = "timed out"
        raise requests.exceptions.ConnectTimeout(msg)

    monkeypatch.setattr(requests, "get", fake_get)
    with pytest.raises(RemoteConnectionError):
        _get("https://www.transfermarkt.co.uk/anything")


def test_an_error_page_is_a_remote_http_error_carrying_the_status(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(503))
    with pytest.raises(RemoteHTTPError) as excinfo:
        _get("https://www.transfermarkt.co.uk/anything")

    assert excinfo.value.status_code == 503
