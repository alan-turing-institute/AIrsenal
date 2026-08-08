"""
Tests for filling the team table at the start of a new season, when there is no
teams CSV for it and the API is the only source.
"""

import pytest
from sqlalchemy import select

from airsenal.framework.schema import Team
from airsenal.scripts.fill_team_table import fill_team_table_from_api
from airsenal.scripts.update_db import update_teams

API_TEAMS = [
    {"id": 1, "short_name": "ARS", "name": "Arsenal"},
    {"id": 2, "short_name": "COV", "name": "Coventry City"},
    {"id": 3, "short_name": "HUL", "name": "Hull City"},
]


class FakeFetcher:
    def __init__(self, teams=None):
        self.teams = API_TEAMS if teams is None else teams

    def get_current_summary_data(self):
        return {"teams": self.teams}


def test_fills_teams_from_the_api(isolated_session):
    ts = isolated_session
    added = fill_team_table_from_api("2627", dbsession=ts, apifetcher=FakeFetcher())
    assert added == 3

    teams = ts.scalars(select(Team).where(Team.season == "2627")).all()
    assert {t.name for t in teams} == {"ARS", "COV", "HUL"}
    assert {t.team_id for t in teams} == {1, 2, 3}


def test_rerunning_updates_rather_than_duplicates(isolated_session):
    """Team ids are reassigned alphabetically each season, so a second run must
    correct the existing rows instead of adding a parallel set."""
    ts = isolated_session
    fill_team_table_from_api("2728", dbsession=ts, apifetcher=FakeFetcher())
    renamed = [{"id": 1, "short_name": "ARS", "name": "Arsenal FC"}]
    added = fill_team_table_from_api(
        "2728", dbsession=ts, apifetcher=FakeFetcher(renamed)
    )

    assert added == 0
    teams = ts.scalars(select(Team).where(Team.season == "2728")).all()
    assert len(teams) == 3
    arsenal = next(t for t in teams if t.team_id == 1)
    assert arsenal.full_name == "Arsenal FC"


def test_empty_api_response_is_an_error(isolated_session):
    """Silently filling nothing would leave every fixture lookup failing later,
    a long way from the cause."""
    with pytest.raises(RuntimeError, match="No teams"):
        fill_team_table_from_api(
            "2829", dbsession=isolated_session, apifetcher=FakeFetcher([])
        )


def test_update_teams_leaves_a_populated_season_alone(isolated_session):
    ts = isolated_session
    team = Team()
    team.name = "ARS"
    team.full_name = "Arsenal"
    team.season = "2930"
    team.team_id = 1
    ts.add(team)
    ts.flush()

    # no fetcher is passed, so a call to the real API would fail the test
    assert update_teams("2930", ts) == 0
