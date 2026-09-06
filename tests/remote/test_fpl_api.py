"""`FPLDataFetcher`'s gameweek lookups, and the login boundary."""

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


def _fetcher_with_history(rounds: list[int]) -> FPLDataFetcher:
    """A fetcher whose one player has played a match in each of `rounds`."""
    fetcher = FPLDataFetcher()
    history = [{"round": r, "total_points": r} for r in rounds]
    fetcher._get = lambda *args, **kwargs: {"history": history}
    return fetcher


def test_a_players_whole_history_comes_back_keyed_by_gameweek():
    """
    Without a gameweek, the answer is every gameweek, not the last one.

    `fill_attributes_table_from_api` calls `.items()` on this, so returning one
    gameweek's list instead of the dict is an AttributeError several frames away.
    """
    fetcher = _fetcher_with_history([1, 2, 3])
    assert fetcher.get_gameweek_data_for_player(123) == {
        1: [{"round": 1, "total_points": 1}],
        2: [{"round": 2, "total_points": 2}],
        3: [{"round": 3, "total_points": 3}],
    }


def test_asking_for_one_gameweek_returns_just_that_gameweeks_matches():
    fetcher = _fetcher_with_history([1, 2, 3])
    assert fetcher.get_gameweek_data_for_player(123, 2) == [
        {"round": 2, "total_points": 2}
    ]


def test_a_double_gameweek_keeps_both_matches():
    """The value is a list because a player can play twice in one gameweek."""
    fetcher = _fetcher_with_history([1, 2, 2])
    assert fetcher.get_gameweek_data_for_player(123, 2) == [
        {"round": 2, "total_points": 2},
        {"round": 2, "total_points": 2},
    ]


def test_a_gameweek_the_player_did_not_play_in_is_empty():
    fetcher = _fetcher_with_history([1, 2])
    assert fetcher.get_gameweek_data_for_player(123, 5) == []


class _LoginBoom:
    """A session that cannot reach the login host."""

    def get(self, *args: object, **kwargs: object) -> object:
        msg = "no route to host"
        raise requests.exceptions.ConnectionError(msg)

    def post(self, *args: object, **kwargs: object) -> object:
        msg = "no route to host"
        raise requests.exceptions.ConnectionError(msg)


def test_login_transport_failure_is_a_remote_error():
    # `login` makes its requests directly rather than through _get_request, so
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
