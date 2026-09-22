"""
Which seasons an update is allowed to touch.

Every read in `update_db` is of the live FPL API, which serves the current
season only, so the season argument is not a choice of what to fetch - it is
only the label the fetched rows are filed under.
"""

import pytest
from typer.testing import CliRunner

from airsenal.cli.main import app
from airsenal.game.season import CURRENT_SEASON
from airsenal.ingest.update import update_db


def test_updating_a_past_season_is_refused():
    """
    It would file this season's prices, positions and fixtures under that one.

    Nothing downstream could tell them apart afterwards, and a past season is
    what the models are fitted on and what a replay is scored against.
    """
    with pytest.raises(ValueError, match="current season"):
        update_db("2223", True, 123, None)


def test_the_message_says_how_to_rebuild_a_past_season_instead():
    with pytest.raises(ValueError, match="airsenal db create"):
        update_db("2223", True, 123, None)


def test_db_update_does_not_offer_a_season():
    """Any season but the current one would be filed with the current one's data."""
    result = CliRunner().invoke(app, ["db", "update", "--help"])

    assert result.exit_code == 0
    assert "--season" not in result.stdout
    assert "--attributes" in result.stdout


def test_db_update_rejects_a_season_argument():
    result = CliRunner().invoke(app, ["db", "update", "--season", "2223"])

    assert result.exit_code != 0


def test_the_current_season_runs(monkeypatch):
    """The guard is on the season alone, and lets the current one through."""
    ran = []
    for name, result in (
        ("update_players", 0),
        ("update_attributes", None),
        ("fill_fixtures_from_api", None),
        ("clear_query_caches", None),
        ("update_results", None),
        ("update_transactions", None),
    ):
        monkeypatch.setattr(
            f"airsenal.ingest.update.{name}",
            lambda *a, _name=name, _result=result, **k: ran.append(_name) or _result,
        )

    assert update_db(CURRENT_SEASON, True, 123, None) is True
    assert "update_results" in ran
