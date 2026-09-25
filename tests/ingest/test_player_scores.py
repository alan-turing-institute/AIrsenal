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
from airsenal.db.models import Base, Fixture, Player, PlayerAttributes
from airsenal.ingest import player_scores
from airsenal.ingest.player_scores import (
    fill_playerscores_from_api,
    get_availability_for_fixture,
    get_status_from_attributes_history,
)

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


# ------------------------- falling back to the gameweek's attributes row ---


def _attributes(dbsession, gameweek, news, chance, return_gameweek=None):
    row = PlayerAttributes()
    row.player_id = 1
    row.season = SEASON
    row.gameweek = gameweek
    row.price = 50
    row.team = "ARS"
    row.position = "MID"
    row.news = news
    row.chance_of_playing_next_round = chance
    row.return_gameweek = return_gameweek
    dbsession.add(row)
    dbsession.commit()


def test_a_season_before_the_history_falls_back_to_the_attributes_row(dbsession):
    """
    Which is where the Transfermarkt scrape lands, for every season before 2526.

    `exclude_unavailable` treats a null chance of playing as available, so without
    it a match a player missed injured would count towards their recent minutes.
    """
    player = _player()
    dbsession.add(player)
    _attributes(dbsession, GAMEWEEK, "Knee injury", 0, 8)
    history = _history([])
    monday = _fixture_on(dbsession, KICKOFFS[2])

    assert get_availability_for_fixture(player, monday, history, dbsession) == (
        "Knee injury",
        0,
    )


def test_the_history_wins_where_it_covers_the_day(dbsession):
    """It is a match-day answer, where the attributes row is a deadline-day one."""
    player = _player()
    dbsession.add(player)
    _attributes(dbsession, GAMEWEEK, "Scraped injury", 0, 8)
    history = _history([(datetime.date(2025, 9, 22), None, 100)])
    monday = _fixture_on(dbsession, KICKOFFS[2])

    assert get_availability_for_fixture(player, monday, history, dbsession) == (
        None,
        100,
    )


def test_a_day_the_history_covers_but_has_nothing_to_say_is_still_an_answer(dbsession):
    """
    A covered day on which the API reported nothing means the player was fine.

    That is not the same as the day being uncovered, and must not reach past it
    for a scrape that says otherwise.
    """
    player = _player()
    dbsession.add(player)
    _attributes(dbsession, GAMEWEEK, "Scraped injury", 0, 8)
    history = _history([(datetime.date(2025, 9, 22), None, None)])
    monday = _fixture_on(dbsession, KICKOFFS[2])

    assert get_availability_for_fixture(player, monday, history, dbsession) == (
        None,
        None,
    )


def test_no_attributes_row_for_the_gameweek_means_no_status(dbsession):
    player = _player()
    dbsession.add(player)
    dbsession.commit()
    monday = _fixture_on(dbsession, KICKOFFS[2])

    assert get_availability_for_fixture(player, monday, _history([]), dbsession) == (
        None,
        None,
    )


def test_an_opponent_from_the_api_is_named_from_that_season(monkeypatch, dbsession):
    """Team ids are per season, so the opponent is looked up in the season filled."""
    past_season = "2425"

    class Fetcher:
        def get_player_summary_data(self):
            return {1: {}}

        def get_gameweek_data_for_player(self, _player_api_id):
            return {GAMEWEEK: [{"opponent_team": 3}]}

    lookups = []

    def record_team_name(team_id, season="unset", dbsession="unset"):
        lookups.append((team_id, season, dbsession))

    monkeypatch.setattr(player_scores, "get_fetcher", Fetcher)
    monkeypatch.setattr(player_scores, "load_attributes_history", lambda _season: None)
    monkeypatch.setattr(
        player_scores, "get_player_from_api_id", lambda _api_id, **_kwargs: _player()
    )
    monkeypatch.setattr(
        player_scores, "_attributes_for_player", lambda _p, _attributes: None
    )
    monkeypatch.setattr(player_scores, "get_team_name", record_team_name)

    fill_playerscores_from_api(
        past_season, gameweek_end=GAMEWEEK + 1, dbsession=dbsession
    )

    assert lookups == [(3, past_season, dbsession)]
