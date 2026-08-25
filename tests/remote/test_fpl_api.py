"""
The gameweek method folded in from `fetch/gameweeks.py`, and the login boundary.
"""

import pytest
from curl_cffi import requests

from airsenal.remote.errors import RemoteConnectionError
from airsenal.remote.fpl_api import FPLDataFetcher
from airsenal.remote.fpl_auth import FPLAuth


def _fetcher_with_events(finished: dict[int, bool]) -> FPLDataFetcher:
    """A fetcher whose event cache is already warm, so nothing is requested."""
    fetcher = FPLDataFetcher()
    fetcher.current_event_data = {
        gw: {"is_finished": done, "deadline": ""} for gw, done in finished.items()
    }
    return fetcher


def test_last_finished_gameweek_is_the_last_one_marked_finished():
    fetcher = _fetcher_with_events({1: True, 2: True, 3: False})
    assert fetcher.get_last_finished_gameweek() == 2


def test_last_finished_gameweek_is_zero_before_the_season_starts():
    fetcher = _fetcher_with_events({1: False, 2: False})
    assert fetcher.get_last_finished_gameweek() == 0


def test_a_stray_finished_flag_after_a_gap_does_not_pull_the_answer_forward():
    # The reason this stops at the first unfinished gameweek instead of taking the
    # maximum: a postponed fixture leaves a later gameweek marked finished.
    fetcher = _fetcher_with_events({1: True, 2: False, 3: True, 4: True})
    assert fetcher.get_last_finished_gameweek() == 1


def test_all_finished_returns_the_last_gameweek():
    fetcher = _fetcher_with_events({1: True, 2: True, 3: True})
    assert fetcher.get_last_finished_gameweek() == 3


class _LoginBoom:
    """A session that cannot reach the login host."""

    def get(self, *args: object, **kwargs: object) -> object:
        msg = "no route to host"
        raise requests.exceptions.ConnectionError(msg)

    def post(self, *args: object, **kwargs: object) -> object:
        msg = "no route to host"
        raise requests.exceptions.ConnectionError(msg)


def test_login_transport_failure_is_a_remote_error():
    # `login` makes its seven requests directly rather than through _get_request, so
    # without translation here a raw curl_cffi error escapes past every
    # `except RemoteError` fallback in squad/ and pipeline/ - which is how an offline
    # run would end in a traceback instead of falling back to the database.
    auth = FPLAuth(_LoginBoom())
    auth.FPL_LOGIN = "someone@example.com"
    auth.FPL_PASSWORD = "secret"
    with pytest.raises(RemoteConnectionError):
        auth.login()


def test_the_fetcher_logs_in_through_its_auth():
    """The endpoints and the login flow are separate objects but one client."""
    auth = FPLAuth(_LoginBoom())
    auth.FPL_LOGIN = "someone@example.com"
    auth.FPL_PASSWORD = "secret"
    fetcher = FPLDataFetcher(auth=auth)
    assert fetcher.logged_in is False
    with pytest.raises(RemoteConnectionError):
        fetcher.login()
