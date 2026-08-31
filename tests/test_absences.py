"""
The absences CSV round-trips between the exporter and the importer.

What `save_expected_absences` writes, `fill_absence_table` reads. Nothing else
checks that the exporter and the importer agree about the columns.
"""

import csv
import warnings
from contextlib import contextmanager
from datetime import date

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from airsenal.core.caching import clear_query_caches
from airsenal.core.data_files import absences_file
from airsenal.db.models import Absence, Base, Fixture, Player, PlayerAttributes
from airsenal.db.queries.absences import was_historic_absence
from airsenal.export.absences import (
    ABSENCE_CSV_COLUMNS,
    classify_reason,
    get_gameweek_start_date,
    player_attribute_to_row,
    save_absences,
)
from airsenal.game.season import CURRENT_SEASON
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
    # A fresh database, so cached answers about gameweeks and dates from whichever
    # database ran before it are wrong for this one. The gameweek lookups are
    # cached on their arguments and not on the session - see core/caching.py - so
    # only clearing them keeps `load_absences` reading the fixtures below.
    clear_query_caches()
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
    clear_query_caches()


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
    """The round trip itself: what the exporter writes, the importer reads."""
    _add_player(dbsession, 1, "Bob", 1, "Knee injury - Expected back 30 Aug", 3)
    # A comma in the news text, to check rows go through the csv module rather
    # than a "," join.
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


def _write_absence_csv(path, date_from, date_until):
    """One absence row, in the columns `load_absences` reads."""
    with open(path, "w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=ABSENCE_CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "season": TEST_SEASON,
                "details": "Knee injury",
                "from": date_from,
                "until": date_until,
                "days": "",
                "games": "",
                "reason": "injury",
                "player": "Bob",
                "url": "",
            }
        )
    return path


@pytest.mark.parametrize(
    ("date_from", "expected_gw_from", "why"),
    [
        ("2025-08-22", 2, "the day before their team plays, so gameweek 2 is missed"),
        ("2025-08-23", 3, "their team's matchday, so they played it and miss from 3"),
        ("2025-08-24", 3, "the day after, so gameweek 2 was played and 3 is missed"),
    ],
)
def test_gw_from_is_the_first_gameweek_the_absence_could_have_stopped_them_playing(
    dbsession, tmp_path, date_from, expected_gw_from, why
):
    """
    An absence beginning on matchday did not stop the player playing that match.

    `date_from` is when the absence began, not the first match missed -
    Transfermarkt dates it to the day, and three quarters of those days are ones
    the player's team was not playing. So the first gameweek missed is the first
    one that kicks off *after* it: a player hurt during Saturday's match is
    available for Saturday's match, and one ruled out on the Friday is not.
    """
    _add_player(dbsession, 1, "Bob", 1, "Knee injury", None)
    dbsession.commit()
    path = _write_absence_csv(tmp_path / "a.csv", date_from, "2025-09-06")

    load_absences(TEST_SEASON, dbsession, path)

    absence = dbsession.scalars(select(Absence)).one()
    assert absence.gw_from == expected_gw_from, why
    # `date_from` itself is stored as given; only the gameweek is resolved
    assert absence.date_from == date_from


def test_an_absence_beginning_on_the_last_matchday_covers_nothing(dbsession, tmp_path):
    """
    Nothing after it kicks off, so there is no gameweek it could have stopped.

    The resolved `gw_from` lands past the end of the season, which the half-open
    range then covers nothing of - rather than reaching back to the match the
    player did play.
    """
    _add_player(dbsession, 1, "Bob", 1, "Knee injury", None)
    dbsession.commit()
    # gameweek 4 is the last one with a fixture in this database
    path = _write_absence_csv(tmp_path / "a.csv", "2025-09-06", "")

    load_absences(TEST_SEASON, dbsession, path)

    absence = dbsession.scalars(select(Absence)).one()
    assert absence.gw_from > 4


# ------------------------------------------------- reading absences back ---


def _add_absence(dbsession, player_id, gw_from, gw_until, season=TEST_SEASON):
    absence = Absence()
    absence.player_id = player_id
    absence.season = season
    absence.reason = "injury"
    absence.date_from = "2025-08-16"
    absence.gw_from = gw_from
    absence.gw_until = gw_until
    absence.timestamp = "2025-08-16"
    dbsession.add(absence)
    dbsession.commit()


@pytest.fixture
def absence_db(tmp_path):
    """A past-season database with one absent player, and no query cache."""
    clear_query_caches()
    with _session(tmp_path) as dbsession:
        player = Player()
        player.player_id = 1
        player.fpl_api_id = 1
        player.name = "Absent"
        dbsession.add(player)
        dbsession.commit()
        yield dbsession
    clear_query_caches()


def test_a_player_is_absent_from_the_first_gameweek_they_miss(absence_db):
    """
    `gw_from` is a gameweek the player missed, so it counts as an absence.

    `load_absences` resolves it to the team's next match on or after the day the
    absence began, which is the first gameweek missed rather than the last one
    played. Excluding it treated the opening week of every absence as available -
    and that is the week the other guard cannot catch either, because the recent
    minutes it reads are all from before the absence.
    """
    _add_absence(absence_db, 1, gw_from=1, gw_until=4)
    player = absence_db.get(Player, 1)

    assert was_historic_absence(player, 1, TEST_SEASON, dbsession=absence_db)
    assert was_historic_absence(player, 2, TEST_SEASON, dbsession=absence_db)
    assert was_historic_absence(player, 3, TEST_SEASON, dbsession=absence_db)
    # gw_until is the gameweek they returned in, so they are available again
    assert not was_historic_absence(player, 4, TEST_SEASON, dbsession=absence_db)


def test_an_absence_ending_the_week_it_began_covers_nothing(absence_db):
    """
    An absence whose end is its own start gameweek was never missed.

    `load_absences` gives both ends the same gameweek when a player is flagged
    and back before their team plays again, so the range has to be able to be
    empty even though `gw_from` itself now counts.
    """
    _add_absence(absence_db, 1, gw_from=2, gw_until=2)
    player = absence_db.get(Player, 1)

    assert not was_historic_absence(player, 2, TEST_SEASON, dbsession=absence_db)


def test_an_open_ended_absence_is_not_counted(absence_db):
    """
    A NULL `gw_until` never satisfied the SQL comparison it replaced.

    Keeping that is deliberate: an absence with no recorded end would otherwise
    zero out the rest of the player's season.
    """
    _add_absence(absence_db, 1, gw_from=1, gw_until=None)
    player = absence_db.get(Player, 1)

    assert not was_historic_absence(player, 2, TEST_SEASON, dbsession=absence_db)


def test_the_current_season_is_never_a_historic_absence(absence_db):
    """The Absence table only covers finished seasons; the API says who is out now."""
    _add_absence(absence_db, 1, gw_from=1, gw_until=4, season=CURRENT_SEASON)
    player = absence_db.get(Player, 1)

    assert not was_historic_absence(player, 2, CURRENT_SEASON, dbsession=absence_db)


def test_a_players_absences_are_read_once_per_season(absence_db):
    """
    One query per player per season, not one per fixture.

    This is read from the innermost loop of the points prediction, so over a
    replay the per-fixture version was tens of thousands of queries for an
    answer that cannot change within a season.
    """
    _add_absence(absence_db, 1, gw_from=1, gw_until=6)
    player = absence_db.get(Player, 1)

    statements = []
    event.listen(
        absence_db.get_bind(),
        "before_cursor_execute",
        lambda *args: statements.append(args[2]),
    )
    for gameweek in range(1, 6):
        was_historic_absence(player, gameweek, TEST_SEASON, dbsession=absence_db)

    assert sum("FROM absence" in s for s in statements) == 1
