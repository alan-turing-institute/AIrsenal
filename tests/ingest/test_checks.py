"""
The consistency checks `airsenal db check` runs over the ingested database.

The module promises to log what it found rather than raise, "so one bad season
does not hide the rest". These test the states that broke that promise: the
database is deliberately half-filled, because that is what the checks are for.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from airsenal.db.models import Base, Fixture, Player, PlayerScore, Result
from airsenal.ingest.checks import fixture_num_conceded, fixture_num_goals

SEASON = "2526"


@pytest.fixture
def dbsession():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def add_fixture(dbsession, home_score, away_score):
    """One played fixture, with a result and no player scores yet."""
    fixture = Fixture(
        date="2025-08-16T14:00:00Z",
        gameweek=1,
        home_team="ARS",
        away_team="CHE",
        season=SEASON,
        tag="test",
    )
    dbsession.add(fixture)
    dbsession.flush()
    dbsession.add(
        Result(
            fixture_id=fixture.fixture_id,
            home_score=home_score,
            away_score=away_score,
        )
    )
    dbsession.flush()
    return fixture


def add_score(dbsession, fixture, team, minutes, conceded, goals=0):
    player = Player(name=f"{team}-{minutes}-{conceded}-{goals}")
    dbsession.add(player)
    dbsession.flush()
    result = fixture.result
    dbsession.add(
        PlayerScore(
            player_team=team,
            opponent="CHE" if team == "ARS" else "ARS",
            points=0,
            goals=goals,
            assists=0,
            bonus=0,
            conceded=conceded,
            minutes=minutes,
            player_id=player.player_id,
            result_id=result.result_id,
            fixture_id=fixture.fixture_id,
        )
    )
    dbsession.flush()


def test_a_result_without_player_scores_is_reported_not_raised(dbsession):
    """
    A fixture whose result landed before its player scores does not end the run.

    That is exactly the state an interrupted update leaves behind, and the state
    this check exists to find - so an empty `max()` over the 90-minute players
    turned the check into a ValueError instead of a warning.
    """
    add_fixture(dbsession, home_score=1, away_score=0)
    dbsession.commit()

    # one per team: neither has a 90-minute player to read a conceded figure off
    assert fixture_num_conceded([SEASON], dbsession) == 2
    # the sibling check over the same fixture already logged and carried on
    assert fixture_num_goals([SEASON], dbsession) == 1


def test_conceded_matching_the_opponents_score_is_no_error(dbsession):
    fixture = add_fixture(dbsession, home_score=1, away_score=2)
    add_score(dbsession, fixture, "ARS", minutes=90, conceded=2, goals=1)
    add_score(dbsession, fixture, "CHE", minutes=90, conceded=1, goals=2)
    dbsession.commit()

    assert fixture_num_conceded([SEASON], dbsession) == 0


def test_conceded_disagreeing_with_the_result_is_one_error_per_team(dbsession):
    fixture = add_fixture(dbsession, home_score=1, away_score=2)
    add_score(dbsession, fixture, "ARS", minutes=90, conceded=0, goals=1)
    add_score(dbsession, fixture, "CHE", minutes=90, conceded=0, goals=2)
    dbsession.commit()

    assert fixture_num_conceded([SEASON], dbsession) == 2


def test_a_team_with_no_full_ninety_is_reported_on_its_own(dbsession):
    """One team short of a 90-minute player does not stop the other being checked."""
    fixture = add_fixture(dbsession, home_score=1, away_score=2)
    add_score(dbsession, fixture, "ARS", minutes=90, conceded=2, goals=1)
    add_score(dbsession, fixture, "CHE", minutes=64, conceded=1, goals=2)
    dbsession.commit()

    # the home team checks out; only the away team cannot be checked
    assert fixture_num_conceded([SEASON], dbsession) == 1


def test_a_fixture_with_no_result_is_skipped(dbsession):
    fixture = Fixture(
        date="2026-05-16T14:00:00Z",
        gameweek=38,
        home_team="ARS",
        away_team="CHE",
        season=SEASON,
        tag="test",
    )
    dbsession.add(fixture)
    dbsession.commit()

    assert fixture_num_conceded([SEASON], dbsession) == 0
