"""
Test that the absences csv files written by save_expected_absences can be read back by
fill_absence_table. The two used to disagree about the file's columns, so anything
written by the exporter was unreadable by the importer.
"""

import warnings
from contextlib import contextmanager
from datetime import date

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from airsenal.core.data_files import absences_file
from airsenal.db.models import Absence, Base, Fixture, Player, PlayerAttributes
from airsenal.export.absences import (
    ABSENCE_CSV_COLUMNS,
    classify_reason,
    get_gameweek_start_date,
    player_attribute_to_row,
    save_absences,
)
from airsenal.ingest.absences import load_absences

TEST_SEASON = "2526"
TEAM = "ARS"
# One fixture per gameweek, a week apart.
GAMEWEEK_DATES = {
    1: "2025-08-16T14:00:00Z",
    2: "2025-08-23T14:00:00Z",
    3: "2025-08-30T14:00:00Z",
    4: "2025-09-06T14:00:00Z",
}


@contextmanager
def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/absences.db")
    Base.metadata.create_all(engine)
    dbsession = sessionmaker(bind=engine)()
    try:
        yield dbsession
    finally:
        dbsession.close()


def _add_player(dbsession, player_id, name, gameweek, news, return_gameweek):
    player = Player()
    player.player_id = player_id
    player.fpl_api_id = player_id
    player.name = name
    dbsession.add(player)

    attributes = PlayerAttributes()
    attributes.player = player
    attributes.player_id = player_id
    attributes.season = TEST_SEASON
    attributes.gameweek = gameweek
    attributes.price = 50
    attributes.team = TEAM
    attributes.position = "MID"
    attributes.chance_of_playing_next_round = 25
    attributes.news = news
    attributes.return_gameweek = return_gameweek
    dbsession.add(attributes)
    return attributes


@pytest.fixture
def dbsession(tmp_path):
    with _session(tmp_path) as dbsession:
        for gameweek, date in GAMEWEEK_DATES.items():
            fixture = Fixture()
            fixture.date = date
            fixture.gameweek = gameweek
            fixture.home_team = TEAM
            fixture.away_team = "CHE"
            fixture.season = TEST_SEASON
            fixture.tag = "test"
            dbsession.add(fixture)
        dbsession.commit()
        yield dbsession


@pytest.mark.parametrize(
    ("news", "expected"),
    [
        ("Knee injury - Expected back 25 Dec", "injury"),
        ("Suspended for three matches", "suspension"),
        ("Received a red card", "suspension"),
        ("Ill", "injury"),
        ("Joined on loan", "absence"),
        ("", "absence"),
        (None, "absence"),
    ],
)
def test_classify_reason(news, expected):
    assert classify_reason(news) == expected


def test_get_gameweek_start_date_uses_earliest_fixture(dbsession):
    assert (
        get_gameweek_start_date(2, TEST_SEASON, dbsession).isoformat() == "2025-08-23"
    )


def test_get_gameweek_start_date_returns_none_for_unscheduled_gameweek(dbsession):
    assert get_gameweek_start_date(38, TEST_SEASON, dbsession) is None


def test_row_has_the_columns_the_reader_expects(dbsession):
    attributes = _add_player(dbsession, 1, "Bob", 1, "Knee injury", 3)
    dbsession.commit()

    row = player_attribute_to_row(attributes, dbsession)

    assert tuple(row.keys()) == ABSENCE_CSV_COLUMNS
    assert row["player"] == "Bob"
    assert row["from"] == "2025-08-16"
    assert row["until"] == "2025-08-30"
    assert row["days"] == "14"
    assert row["games"] == "2"
    assert row["reason"] == "injury"


def test_row_is_none_when_gameweek_has_no_fixtures(dbsession):
    attributes = _add_player(dbsession, 1, "Bob", 38, "Knee injury", None)
    dbsession.commit()

    assert player_attribute_to_row(attributes, dbsession) is None


def test_open_ended_absence_has_blank_until(dbsession):
    attributes = _add_player(dbsession, 1, "Bob", 1, "Knee injury", None)
    dbsession.commit()

    row = player_attribute_to_row(attributes, dbsession)

    assert row["until"] == ""
    assert row["days"] == ""
    assert row["games"] == ""


def test_save_absences_round_trips_through_load_absences(dbsession, tmp_path):
    """
    The important one: what the exporter writes, the importer must be able to read.
    """
    _add_player(dbsession, 1, "Bob", 1, "Knee injury - Expected back 30 Aug", 3)
    # A comma in the news text used to corrupt the file, because rows were written by
    # joining fields with "," rather than by the csv module.
    _add_player(dbsession, 2, "Alice", 2, "Ankle injury, out indefinitely", None)
    dbsession.commit()

    attributes = dbsession.scalars(select(PlayerAttributes)).all()
    rows = [player_attribute_to_row(pa, dbsession) for pa in attributes]
    path = str(tmp_path / f"absences_{TEST_SEASON}.csv")
    assert save_absences(rows, TEST_SEASON, path) == 2

    load_absences(TEST_SEASON, dbsession, path)

    absences = dbsession.scalars(select(Absence).order_by(Absence.player_id)).all()
    assert len(absences) == 2

    bob, alice = absences
    assert bob.player.name == "Bob"
    assert bob.reason == "injury"
    assert bob.details == "Knee injury - Expected back 30 Aug"
    assert bob.date_from == "2025-08-16"
    assert bob.date_until == "2025-08-30"

    assert alice.player.name == "Alice"
    assert alice.details == "Ankle injury, out indefinitely"
    assert alice.date_until is None


def test_save_absences_does_not_write_duplicates(dbsession, tmp_path):
    _add_player(dbsession, 1, "Bob", 1, "Knee injury", 3)
    dbsession.commit()

    attributes = dbsession.scalars(select(PlayerAttributes)).all()
    rows = [player_attribute_to_row(pa, dbsession) for pa in attributes]
    path = str(tmp_path / f"absences_{TEST_SEASON}.csv")

    assert save_absences(rows, TEST_SEASON, path) == 1
    assert save_absences(rows, TEST_SEASON, path) == 0

    with open(path) as infile:
        assert len(infile.readlines()) == 2  # header + one row


def test_reader_and_writer_agree_on_the_path():
    """Both modules resolve the csv path through the same helper."""
    assert absences_file(TEST_SEASON).name == f"absences_{TEST_SEASON}.csv"


def test_absence_dates_are_stored_as_iso_strings(dbsession, tmp_path):
    """
    date_from/date_until are VARCHAR columns, so they must receive ISO-8601 text.

    Passing datetime.date objects worked only through sqlite3's default date adapter,
    which is deprecated in Python 3.12 and slated for removal - and would not work at
    all against the postgres backend, whose columns are also VARCHAR.
    """
    _add_player(dbsession, 1, "Bob", 1, "Knee injury", 3)
    _add_player(dbsession, 2, "Alice", 2, "Ankle injury", None)
    dbsession.commit()

    attributes = dbsession.scalars(select(PlayerAttributes)).all()
    rows = [player_attribute_to_row(pa, dbsession) for pa in attributes]
    path = str(tmp_path / f"absences_{TEST_SEASON}.csv")
    save_absences(rows, TEST_SEASON, path)
    load_absences(TEST_SEASON, dbsession, path)

    for absence in dbsession.scalars(select(Absence)).all():
        assert isinstance(absence.date_from, str), (
            f"date_from is {type(absence.date_from).__name__}, not str"
        )
        assert date.fromisoformat(absence.date_from)
        assert absence.date_until is None or isinstance(absence.date_until, str)
        if absence.date_until is not None:
            assert date.fromisoformat(absence.date_until)


def test_loading_absences_emits_no_deprecated_date_adapter_warning(dbsession, tmp_path):
    """Writing dates as text means sqlite3's deprecated adapter is never reached."""
    _add_player(dbsession, 1, "Bob", 1, "Knee injury", 3)
    dbsession.commit()

    attributes = dbsession.scalars(select(PlayerAttributes)).all()
    rows = [player_attribute_to_row(pa, dbsession) for pa in attributes]
    path = str(tmp_path / f"absences_{TEST_SEASON}.csv")
    save_absences(rows, TEST_SEASON, path)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        load_absences(TEST_SEASON, dbsession, path)
