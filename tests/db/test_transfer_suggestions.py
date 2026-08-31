"""
Reading back the suggestions of one optimisation run.

An isolated database rather than the shared dummy one: what is being tested is
which run `get_transfer_suggestions` decides is the current one, and that answer
depends on every row in the table.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from airsenal.db.models import Base, TransferSuggestion
from airsenal.db.queries.predictions import get_transfer_suggestions

SEASON = "2526"
PAST_SEASON = "2122"


@pytest.fixture
def dbsession():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def add_run(
    dbsession, timestamp, fpl_team_id, season=SEASON, gameweek=5, player_ids=(1, 2)
):
    """One optimisation run's rows: they share a timestamp, as a real run's do."""
    for player_id in player_ids:
        suggestion = TransferSuggestion()
        suggestion.player_id = player_id
        suggestion.in_or_out = 1
        suggestion.gameweek = gameweek
        suggestion.points_gain = 1.0
        suggestion.timestamp = timestamp
        suggestion.season = season
        suggestion.fpl_team_id = fpl_team_id
        suggestion.chip_played = None
        dbsession.add(suggestion)
    dbsession.commit()


def test_a_second_entrys_run_does_not_hide_the_first(dbsession):
    """
    Each entry gets its own latest run.

    Optimising two entries in turn wrote two runs a second apart. Picking the
    newest row in the whole table and only then filtering by fpl_team_id left the
    entry that was optimised first with nothing to apply.
    """
    add_run(dbsession, "2026-08-31 10:00:00.000000", fpl_team_id=111, player_ids=(1,))
    add_run(dbsession, "2026-08-31 10:00:01.000000", fpl_team_id=222, player_ids=(2,))

    for fpl_team_id, expected in ((111, 1), (222, 2)):
        rows = get_transfer_suggestions(
            gameweek=5, season=SEASON, fpl_team_id=fpl_team_id, dbsession=dbsession
        )
        assert [row.player_id for row in rows] == [expected]


def test_a_replay_does_not_hide_the_season_being_played(dbsession):
    """
    Replaying a past season leaves the current season's suggestions readable.

    A replay optimises every gameweek and writes suggestions for each, stamped
    with today's clock, so its rows are always the newest in the table.
    """
    add_run(dbsession, "2026-08-31 10:00:00.000000", fpl_team_id=111)
    add_run(
        dbsession,
        "2026-08-31 11:00:00.000000",
        fpl_team_id=-1,
        season=PAST_SEASON,
        gameweek=20,
    )

    rows = get_transfer_suggestions(
        gameweek=5, season=SEASON, fpl_team_id=111, dbsession=dbsession
    )
    assert [row.player_id for row in rows] == [1, 2]


def test_a_superseded_plan_is_not_offered_for_a_gameweek_the_latest_one_missed(
    dbsession,
):
    """
    The gameweek selects within the current run; it does not reach back past it.

    Otherwise asking for a gameweek the latest plan does not cover would answer
    with an older plan's transfers, which have already been superseded.
    """
    add_run(dbsession, "2026-08-31 10:00:00.000000", fpl_team_id=111, gameweek=5)
    add_run(dbsession, "2026-08-31 12:00:00.000000", fpl_team_id=111, gameweek=6)

    assert (
        get_transfer_suggestions(
            gameweek=5, season=SEASON, fpl_team_id=111, dbsession=dbsession
        )
        == []
    )
    assert (
        len(
            get_transfer_suggestions(
                gameweek=6, season=SEASON, fpl_team_id=111, dbsession=dbsession
            )
        )
        == 2
    )


def test_no_suggestions_at_all_is_not_an_error(dbsession):
    assert (
        get_transfer_suggestions(
            gameweek=5, season=SEASON, fpl_team_id=111, dbsession=dbsession
        )
        == []
    )
