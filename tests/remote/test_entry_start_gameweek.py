"""
That `get_entry_start_gameweek` still branches on *which* remote failure happened.

This is the behaviour the three-class error hierarchy exists for. A single flat
`RemoteError` would make both cases take the same path, and every other test in
the suite would still pass.
"""

from airsenal.remote.errors import RemoteConnectionError, RemoteHTTPError
from airsenal.squad import state


def test_a_missing_gameweek_scans_forward(monkeypatch):
    # The API 404s for gameweeks 1-3, then has picks for 4.
    monkeypatch.setattr(state, "next_gameweek", lambda *a, **k: 10)

    def picks(gameweek, fpl_team_id=None, fetcher=None):
        if gameweek < 4:
            msg = f"no picks for gw {gameweek}"
            raise RemoteHTTPError(msg, 404)
        return ["a player"]

    monkeypatch.setattr(state, "get_players_for_gameweek", picks)
    assert state.get_entry_start_gameweek(123, fetcher=object()) == 4


def test_an_unreachable_api_gives_up_and_assumes_gameweek_one(monkeypatch):
    monkeypatch.setattr(state, "next_gameweek", lambda *a, **k: 10)

    def unreachable(gameweek, fpl_team_id=None, fetcher=None):
        msg = "down"
        raise RemoteConnectionError(msg)

    monkeypatch.setattr(state, "get_players_for_gameweek", unreachable)
    assert state.get_entry_start_gameweek(123, fetcher=object()) == 1


def test_a_connection_failure_does_not_scan_the_whole_season(monkeypatch):
    # The distinction matters for cost too: merging the two would send one request
    # per gameweek to a host that is known to be unreachable.
    monkeypatch.setattr(state, "next_gameweek", lambda *a, **k: 38)
    calls = []

    def unreachable(gameweek, fpl_team_id=None, fetcher=None):
        calls.append(gameweek)
        msg = "down"
        raise RemoteConnectionError(msg)

    monkeypatch.setattr(state, "get_players_for_gameweek", unreachable)
    state.get_entry_start_gameweek(123, fetcher=object())
    assert calls == [1]
