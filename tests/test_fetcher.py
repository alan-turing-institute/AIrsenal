"""
The FPL API still returns what we expect.

Every test here makes a live request, so the whole module is marked `live` and is
deselected by default (see the addopts in pyproject.toml). Run it with `pytest -m live`
when you want to check whether the API has changed shape.

Each test names the keys the package actually reads off that endpoint, because a
renamed field is the drift this exists to catch and it passes any "is a non-empty
dict" check. The reader of each set is named beside it, so a failure here says
which module breaks. The fetcher's own caching and gameweek arithmetic are tested
offline in tests/remote/; nothing here needs to repeat them.
"""

import random

import pytest

from airsenal.remote.fpl_api import FPLDataFetcher

pytestmark = pytest.mark.live


def test_get_summary_data():
    """The bootstrap endpoint: `export/attributes.py`, `ingest/player_attributes.py`."""
    data = FPLDataFetcher().get_current_summary_data()
    assert {"elements", "teams", "events", "total_players"} <= data.keys()
    assert len(data["elements"]) > 0


def test_get_team_history_data():
    """Our entry's gameweek history: `squad/state.py` and `reporting/plots.py`."""
    data = FPLDataFetcher().get_fpl_team_history_data()
    assert "current" in data
    assert len(data["current"]) > 0
    assert {
        "event",
        "event_transfers",
        "bank",
        "points",
        "total_points",
        "rank",
        "overall_rank",
    } <= data["current"][0].keys()


def test_get_event_data():
    """Every gameweek's deadline and status: `get_last_finished_gameweek`."""
    data = FPLDataFetcher().get_event_data()
    assert len(data) == 38
    assert {"deadline", "is_finished"} <= data[1].keys()


def test_get_player_summary_data():
    """One row per player, keyed by api id: `ingest/players.py`, `squad/pricing.py`."""
    data = FPLDataFetcher().get_player_summary_data()
    assert len(data) > 0
    player = next(iter(data.values()))
    assert {
        "first_name",
        "second_name",
        "opta_code",
        "element_type",
        "team",
        "now_cost",
        "news",
        "chance_of_playing_next_round",
    } <= player.keys()


def test_get_current_team_data():
    """This season's teams, keyed by code: `export/player_details.py`."""
    data = FPLDataFetcher().get_current_team_data()
    assert len(data) == 20
    assert {"id", "name", "short_name"} <= next(iter(data.values())).keys()


def test_get_fpl_team_data_gw1():
    """An entry's picks for a gameweek: `get_players_for_gameweek`."""
    data = FPLDataFetcher().get_fpl_team_data(1)
    assert "picks" in data
    assert len(data["picks"]) == 15
    assert "element" in data["picks"][0]
    # `free_hit_used_in_gameweek` reads this, and `.get`s it, so it may be absent
    assert data.get("active_chip", None) != ""


def test_get_fpl_team_data_gw1_different_fpl_team_ids():
    """Two other entries' picks for gameweek 1."""
    fetcher = FPLDataFetcher()
    # assume that fpl_team_ids < 100 will all have squads for
    # gameweek 1, and that they will be different..
    team_id_1 = random.randint(1, 50)
    team_id_2 = random.randint(51, 100)
    data_1 = fetcher.get_fpl_team_data(1, fpl_team_id=team_id_1)
    players_1 = [p["element"] for p in data_1["picks"]]
    assert len(players_1) == 15
    data_2 = fetcher.get_fpl_team_data(1, fpl_team_id=team_id_2)
    players_2 = [p["element"] for p in data_2["picks"]]
    assert len(players_2) == 15
    # check they are different
    assert sorted(players_1) != sorted(players_2)


def test_get_detailed_player_data():
    """One player's per-gameweek data: `ingest/player_scores.py`."""
    data = FPLDataFetcher().get_gameweek_data_for_player(1)
    assert len(data) > 0
    # a list per gameweek, because of double gameweeks
    fixture = next(iter(data.values()))[0]
    assert {
        "round",
        "fixture",
        "opponent_team",
        "was_home",
        "kickoff_time",
        "minutes",
        "goals_scored",
        "assists",
        "bonus",
        "total_points",
        "goals_conceded",
    } <= fixture.keys()
