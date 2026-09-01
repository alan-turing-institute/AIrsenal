"""
Fetching a Transfermarkt page.

Only `_get` is covered here - the timeout, the error translation, and which
failures are worth retrying. What the parsers make of what comes back is in
`test_transfermarkt_parsing.py`, against recorded copies of the pages.
"""

import pytest
import requests

from airsenal.remote import transfermarkt
from airsenal.remote.errors import RemoteConnectionError, RemoteHTTPError
from airsenal.remote.transfermarkt import TIMEOUT_SECONDS, _get


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    """Run the requests and retries at once, rather than at scraping pace."""
    monkeypatch.setattr(transfermarkt, "RETRY_BACKOFF_SECONDS", 0)
    monkeypatch.setattr(transfermarkt, "REQUEST_DELAY_SECONDS", 0)


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


def test_a_timeout_is_retried(monkeypatch):
    """
    Transfermarkt times requests out part way through a scrape of a division.

    Giving up on the first one loses that player's absences from the season, and
    `get_season_absences` carries on without them.
    """
    attempts = []

    def flaky_get(url, **kwargs):
        attempts.append(url)
        if len(attempts) < 3:
            msg = "timed out"
            raise requests.exceptions.ConnectTimeout(msg)
        return FakeResponse()

    monkeypatch.setattr(requests, "get", flaky_get)
    _get("https://www.transfermarkt.co.uk/anything")

    assert len(attempts) == 3


def test_being_asked_to_slow_down_is_retried(monkeypatch):
    attempts = []

    def rate_limited(url, **kwargs):
        attempts.append(url)
        return FakeResponse(429) if len(attempts) < 2 else FakeResponse()

    monkeypatch.setattr(requests, "get", rate_limited)
    _get("https://www.transfermarkt.co.uk/anything")

    assert len(attempts) == 2


def test_a_missing_page_is_not_retried(monkeypatch):
    """A 404 is the site's answer, not a request to try again."""
    attempts = []

    def missing(url, **kwargs):
        attempts.append(url)
        return FakeResponse(404)

    monkeypatch.setattr(requests, "get", missing)
    with pytest.raises(RemoteHTTPError):
        _get("https://www.transfermarkt.co.uk/anything")

    assert len(attempts) == 1


def test_requests_are_paced(monkeypatch):
    """
    Unpaced, Transfermarkt starts timing us out inside the first thirty requests.

    A season is roughly 1800 of them, so the delay is what makes the difference
    between a complete scrape and one missing a tenth of its players.
    """
    waits = []
    monkeypatch.setattr(transfermarkt.time, "sleep", waits.append)
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse())

    _get("https://www.transfermarkt.co.uk/anything")

    assert waits == [transfermarkt.REQUEST_DELAY_SECONDS]
