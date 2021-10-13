"""
test that we get valid responses from the API.
"""

import random

import pytest

from airsenal.framework.data_fetcher import FPLDataFetcher
from airsenal.framework.utils import NEXT_GAMEWEEK


def test_instantiate_fetchers():
    """
    check we can instantiate the classes
    """
    fpl = FPLDataFetcher()
    assert fpl


def test_get_summary_data():
    """
    get summary of all players' data for this season.
    """
    fetcher = FPLDataFetcher()
    data = fetcher.get_current_summary_data()
    assert isinstance(data, dict)
    assert len(data) > 0


@pytest.mark.skipif(NEXT_GAMEWEEK == 1, reason="No team data before start of season")
def test_get_team_data():
    """
    should give current list of players in our team
    """
    fetcher = FPLDataFetcher()
    data = fetcher.get_fpl_team_data(1)["picks"]
    assert isinstance(data, list)
    assert len(data) == 15


@pytest.mark.skipif(NEXT_GAMEWEEK == 1, reason="No team data before start of season")
def test_get_team_history_data():
    """
    gameweek history for our team id
    """
    fetcher = FPLDataFetcher()
    data = fetcher.get_fpl_team_history_data()
    assert isinstance(data, dict)
    assert len(data) > 0


def test_get_event_data():
    """
    gameweek list with deadlines and status
    """
    fetcher = FPLDataFetcher()
    data = fetcher.get_event_data()
    assert isinstance(data, dict)
    assert len(data) > 0


def test_get_player_summary_data():
    """
    summary for individual players
    """
    fetcher = FPLDataFetcher()
    data = fetcher.get_player_summary_data()
    assert isinstance(data, dict)
    assert len(data) > 0


def test_get_current_team_data():
    """
    summary for current teams
    """
    fetcher = FPLDataFetcher()
    data = fetcher.get_current_team_data()
    assert isinstance(data, dict)
    assert len(data) > 0


@pytest.mark.skipif(NEXT_GAMEWEEK == 1, reason="No data yet for gameweek 1")
def test_get_fpl_team_data_gw1():
    """
    which players are in our squad for gw1
    """
    fetcher = FPLDataFetcher()
    data = fetcher.get_fpl_team_data(1)
    assert isinstance(data, dict)
    assert "picks" in data.keys()
    players = [p["element"] for p in data["picks"]]
    assert len(players) == 15


@pytest.mark.skipif(NEXT_GAMEWEEK == 1, reason="No data yet for gameweek 1")
def test_get_fpl_team_data_gw1_different_fpl_team_ids():
    """
    which players are in a couple of different squads for gw 1
    """
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


@pytest.mark.skipif(NEXT_GAMEWEEK == 1, reason="No data yet for gameweek 1")
def test_get_detailed_player_data():
    """
    for player_id=1, list of gameweek data
    """
    fetcher = FPLDataFetcher()

    data = fetcher.get_gameweek_data_for_player(1)
    assert isinstance(data, dict)
    assert len(data) > 0
