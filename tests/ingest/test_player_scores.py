"""
The availability recorded against a past performance is the one on match day.

`PlayerAttributes` records what was known at each gameweek's deadline, because
that is when a squad is picked. `PlayerScore` records what was known on the
morning of that particular kickoff, because that is what explains the minutes.
A double gameweek has one of the first and two of the second.
"""

import datetime
from contextlib import contextmanager

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from airsenal.core.caching import clear_query_caches
from airsenal.db.models import Base, Fixture, Player
from airsenal.ingest.player_scores import get_status_from_attributes_history

SEASON = "2526"
GAMEWEEK = 5
# A gameweek spread over three days, as a real one is.
KICKOFFS = ["2025-09-20T14:00:00Z", "2025-09-21T14:00:00Z", "2025-09-22T20:00:00Z"]


@contextmanager
def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    dbsession = sessionmaker(bind=engine)()
    try:
        yield dbsession
    finally:
        dbsession.close()


@pytest.fixture
def dbsession():
    clear_query_caches()
    with _session() as dbsession:
        for kickoff in KICKOFFS:
            fixture = Fixture()
            fixture.date = kickoff
            fixture.gameweek = GAMEWEEK
            fixture.home_team = "ARS"
            fixture.away_team = "CHE"
            fixture.season = SEASON
            fixture.tag = "test"
            dbsession.add(fixture)
        dbsession.commit()
        yield dbsession
    clear_query_caches()


def _player():
    player = Player()
    player.player_id = 1
    player.name = "Bob"
    player.opta_code = "p1"
    return player


def _history(rows):
    """This player's rows of the attributes history, as the caller filters them."""
    frame = pd.DataFrame(rows, columns=["day", "news", "chance_of_playing_next_round"])
    frame["chance_of_playing_next_round"] = frame[
        "chance_of_playing_next_round"
    ].astype("float64")
    return frame


def _fixture_on(dbsession, kickoff):
    return dbsession.query(Fixture).filter(Fixture.date == kickoff).one()


def test_the_status_is_the_one_on_the_day_of_that_kickoff(dbsession):
    """
    Not the gameweek's first day: a player can be ruled out between the two.

    The Monday-night fixture of a gameweek that began on Saturday is played two
    days after the deadline, and the squad that picked them cannot be unpicked -
    but what they were fit for on the Monday is what their minutes reflect.
    """
    history = _history(
        [
            (datetime.date(2025, 9, 20), None, 100),
            (datetime.date(2025, 9, 21), "Knock", 75),
            (datetime.date(2025, 9, 22), "Hamstring injury", 0),
        ]
    )

    monday = _fixture_on(dbsession, KICKOFFS[2])
    assert get_status_from_attributes_history(
        _player(), monday, history, dbsession
    ) == ("Hamstring injury", 0)

    saturday = _fixture_on(dbsession, KICKOFFS[0])
    assert get_status_from_attributes_history(
        _player(), saturday, history, dbsession
    ) == (None, 100)


def test_known_future_unavailability_is_read_from_the_deadline_instead(dbsession):
    """
    The API flags international duty on match day for the gameweek *after* it.

    Taking that at face value would record a player who played 90 minutes as
    having been unavailable for the match they played, so for the handful of news
    texts that mean "not this gameweek, the next one" the deadline-day row wins.
    """
    history = _history(
        [
            (datetime.date(2025, 9, 20), None, 100),
            (datetime.date(2025, 9, 22), "On international duty", 0),
        ]
    )

    monday = _fixture_on(dbsession, KICKOFFS[2])

    assert get_status_from_attributes_history(
        _player(), monday, history, dbsession
    ) == (None, 100)


def test_an_ordinary_injury_on_match_day_is_not_second_guessed(dbsession):
    """Only the news texts that name a future commitment get the deadline row."""
    history = _history(
        [
            (datetime.date(2025, 9, 20), None, 100),
            (datetime.date(2025, 9, 22), "Hamstring injury", 0),
        ]
    )

    monday = _fixture_on(dbsession, KICKOFFS[2])

    assert get_status_from_attributes_history(
        _player(), monday, history, dbsession
    ) == ("Hamstring injury", 0)


def test_a_match_before_the_daily_dump_started_has_no_status(dbsession):
    """The history begins on 12 September 2025; there is nothing to read before it."""
    fixture = Fixture()
    fixture.date = "2025-08-16T14:00:00Z"
    fixture.gameweek = 1
    fixture.home_team = "ARS"
    fixture.away_team = "CHE"
    fixture.season = SEASON
    fixture.tag = "test"
    dbsession.add(fixture)
    dbsession.commit()
    history = _history([(datetime.date(2025, 8, 16), "Knee injury", 0)])

    assert get_status_from_attributes_history(
        _player(), fixture, history, dbsession
    ) == (None, None)
