"""
Absences from the packaged CSV become availability on the attributes table.

Each row gives a date range, which is resolved to a half-open range of gameweeks
and written onto those gameweeks' `PlayerAttributes` rows, so that
`Player.is_injured_or_suspended` answers for a past season the same way it does
for the current one.
"""

import csv
from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from airsenal.core.caching import clear_query_caches
from airsenal.core.data_files import absences_file
from airsenal.db.models import Base, Fixture, Player, PlayerAttributes
from airsenal.ingest.absences import get_availability_from_absences
from airsenal.ingest.attributes_history import Availability
from airsenal.ingest.player_attributes import set_availability

TEST_SEASON = "2425"
TEAM = "ARS"
# One fixture per gameweek, a week apart.
GAMEWEEK_DATES = {
    1: "2025-08-16T14:00:00Z",
    2: "2025-08-23T14:00:00Z",
    3: "2025-08-30T14:00:00Z",
    4: "2025-09-06T14:00:00Z",
}
# The header of the packaged absences_yyyy.csv files. "days" and "games" come
# from the Transfermarkt scrape and are not read back.
ABSENCE_CSV_COLUMNS = (
    "season",
    "details",
    "from",
    "until",
    "days",
    "games",
    "reason",
    "player",
    "url",
)


@contextmanager
def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/absences.db")
    Base.metadata.create_all(engine)
    dbsession = sessionmaker(bind=engine)()
    try:
        yield dbsession
    finally:
        dbsession.close()


def _add_player(dbsession, player_id, name, gameweeks=tuple(GAMEWEEK_DATES)):
    """A player with an unflagged attributes row in each of `gameweeks`."""
    player = Player()
    player.player_id = player_id
    player.fpl_api_id = player_id
    player.name = name
    dbsession.add(player)

    for gameweek in gameweeks:
        attributes = PlayerAttributes()
        attributes.player = player
        attributes.player_id = player_id
        attributes.season = TEST_SEASON
        attributes.gameweek = gameweek
        attributes.price = 50
        attributes.team = TEAM
        attributes.position = "MID"
        dbsession.add(attributes)
    return player


@pytest.fixture
def dbsession(tmp_path):
    # A fresh database, so cached answers about gameweeks and dates from whichever
    # database ran before it are wrong for this one. The gameweek lookups are
    # cached on their arguments and not on the session - see core/caching.py - so
    # only clearing them keeps the absence resolution reading the fixtures below.
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


def _write_absence_csv(path, date_from, date_until, player="Bob"):
    """One absence row, in the columns the importer reads."""
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
                "player": player,
                "url": "",
            }
        )
    return path


def _load(dbsession, path):
    """Resolve the csv and write what it says onto the attributes table."""
    availability = get_availability_from_absences(TEST_SEASON, dbsession, path)
    set_availability(availability, TEST_SEASON, dbsession)


def _unavailable_gameweeks(player, gameweeks=tuple(GAMEWEEK_DATES)):
    """The gameweeks this player is unavailable for, asked about from each."""
    return [
        gameweek
        for gameweek in gameweeks
        if player.is_injured_or_suspended(TEST_SEASON, gameweek, gameweek)
    ]


# ------------------------------------------------- resolving the csv row ---


@pytest.mark.parametrize(
    ("date_from", "expected_gameweek_from", "why"),
    [
        ("2025-08-22", 2, "the day before their team plays, so gameweek 2 is missed"),
        ("2025-08-23", 3, "their team's matchday, so they played it and miss from 3"),
        ("2025-08-24", 3, "the day after, so gameweek 2 was played and 3 is missed"),
    ],
)
def test_the_first_gameweek_missed_is_the_first_one_kicking_off_after_it_began(
    dbsession, tmp_path, date_from, expected_gameweek_from, why
):
    """
    An absence beginning on matchday did not stop the player playing that match.

    `date_from` is when the absence began, not the first match missed -
    Transfermarkt dates it to the day, and three quarters of those days are ones
    the player's team was not playing. So the first gameweek missed is the first
    one that kicks off *after* it: a player hurt during Saturday's match is
    available for Saturday's match, and one ruled out on the Friday is not.
    """
    player = _add_player(dbsession, 1, "Bob")
    dbsession.commit()
    path = _write_absence_csv(tmp_path / "a.csv", date_from, "2025-09-06")

    _load(dbsession, path)

    assert _unavailable_gameweeks(player) == list(range(expected_gameweek_from, 4)), why


def test_an_absence_beginning_on_the_last_matchday_covers_nothing(dbsession, tmp_path):
    """
    Nothing after it kicks off, so there is no gameweek it could have stopped.

    The resolved first gameweek lands past the end of the season, which the
    half-open range then covers nothing of - rather than reaching back to the
    match the player did play.
    """
    player = _add_player(dbsession, 1, "Bob")
    dbsession.commit()
    # gameweek 4 is the last one with a fixture in this database
    path = _write_absence_csv(tmp_path / "a.csv", "2025-09-06", "")

    _load(dbsession, path)

    assert _unavailable_gameweeks(player) == []


def test_an_absence_ending_after_the_season_covers_the_rest_of_it(dbsession, tmp_path):
    """
    An end date past the last fixture means they never came back that season.

    The half-open range therefore ends one past the last gameweek, so a long
    injury or a mid-season transfer out of the league covers the rest of the season.
    """
    player = _add_player(dbsession, 1, "Bob")
    dbsession.commit()
    # the last fixture in this database is gameweek 4, on 2025-09-06
    path = _write_absence_csv(tmp_path / "a.csv", "2025-08-17", "2026-05-24")

    _load(dbsession, path)

    assert _unavailable_gameweeks(player) == [2, 3, 4]
    attributes = dbsession.scalars(
        select(PlayerAttributes).where(PlayerAttributes.gameweek == 2)
    ).one()
    assert attributes.return_gameweek == 5


def test_an_absence_with_no_end_date_lasts_the_rest_of_the_season(dbsession, tmp_path):
    """
    A blank end date means the player did not come back.

    Every packaged file is a scrape of a season that has finished, so a blank
    `until` is a season they did not return in.
    """
    player = _add_player(dbsession, 1, "Bob")
    dbsession.commit()
    path = _write_absence_csv(tmp_path / "a.csv", "2025-08-17", "")

    _load(dbsession, path)

    assert _unavailable_gameweeks(player) == [2, 3, 4]


def test_a_player_is_found_under_a_differently_spelled_name(dbsession, tmp_path):
    """Transfermarkt drops the family names the FPL API keeps, among other things."""
    player = _add_player(dbsession, 1, "Matheus Santos Carneiro da Cunha")
    dbsession.commit()
    path = _write_absence_csv(
        tmp_path / "a.csv", "2025-08-17", "2025-09-06", player="Matheus Cunha"
    )

    _load(dbsession, path)

    assert _unavailable_gameweeks(player) == [2, 3]


def test_a_name_that_fits_two_players_is_not_guessed_at(dbsession, tmp_path):
    """Filing one player's absence against another is worse than filing neither."""
    ward = _add_player(dbsession, 1, "Danny Ward")
    daniel = _add_player(dbsession, 2, "Daniel Ward")
    dbsession.commit()
    path = _write_absence_csv(
        tmp_path / "a.csv", "2025-08-17", "2025-09-06", player="Dan Ward"
    )

    _load(dbsession, path)

    assert _unavailable_gameweeks(ward) == []
    assert _unavailable_gameweeks(daniel) == []


def test_the_importer_resolves_the_packaged_path():
    assert absences_file(TEST_SEASON).name == f"absences_{TEST_SEASON}.csv"


def test_overlapping_absences_keep_the_one_that_lasts_longest(dbsession, tmp_path):
    """Two absences over one gameweek: the player is out until the later return."""
    _add_player(dbsession, 1, "Bob")
    dbsession.commit()
    path = tmp_path / "a.csv"
    with open(path, "w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=ABSENCE_CSV_COLUMNS)
        writer.writeheader()
        for details, date_until in [("Knock", "2025-08-30"), ("Knee", "2026-05-24")]:
            writer.writerow(
                {
                    "season": TEST_SEASON,
                    "details": details,
                    "from": "2025-08-17",
                    "until": date_until,
                    "days": "",
                    "games": "",
                    "reason": "injury",
                    "player": "Bob",
                    "url": "",
                }
            )

    availability = get_availability_from_absences(TEST_SEASON, dbsession, path)

    assert availability[(1, 2)] == Availability(
        news="Knee", chance_of_playing_next_round=0, return_gameweek=5
    )


def test_a_missing_absences_file_is_not_an_error(dbsession, tmp_path):
    """A season only gets a file once it has been scraped."""
    assert (
        get_availability_from_absences(TEST_SEASON, dbsession, tmp_path / "nope.csv")
        == {}
    )


# ------------------------------------------- writing it onto the table ---


def _set_absent(dbsession, player_id, gameweek_from, gameweek_until):
    """Mark a player absent over a half-open gameweek range, as the csv would."""
    set_availability(
        {
            (player_id, gameweek): Availability(
                news="Knee injury",
                chance_of_playing_next_round=0,
                return_gameweek=gameweek_until,
            )
            for gameweek in range(gameweek_from, gameweek_until)
        },
        TEST_SEASON,
        dbsession,
    )


@pytest.fixture
def absence_db(dbsession):
    """A past-season database with one player who has a row in every gameweek."""
    _add_player(dbsession, 1, "Absent")
    dbsession.commit()
    return dbsession


def test_a_player_is_absent_from_the_first_gameweek_they_miss(absence_db):
    """
    The first gameweek missed counts as an absence, not the last one played.

    The recent-minutes guard cannot catch the opening week of an absence, because
    the minutes it reads are all from before the absence.
    """
    _set_absent(absence_db, 1, gameweek_from=1, gameweek_until=4)
    player = absence_db.get(Player, 1)

    assert player.is_injured_or_suspended(TEST_SEASON, 1, 1)
    assert player.is_injured_or_suspended(TEST_SEASON, 2, 2)
    assert player.is_injured_or_suspended(TEST_SEASON, 3, 3)
    # the return gameweek is the one they came back in, so they are available again
    assert not player.is_injured_or_suspended(TEST_SEASON, 4, 4)


def test_an_absence_ending_the_week_it_began_covers_nothing(absence_db):
    """
    An absence whose end is its own start gameweek was never missed.

    Both ends get the same gameweek when a player is flagged and back before
    their team plays again, so the range has to be able to be empty even though
    the first gameweek itself counts.
    """
    _set_absent(absence_db, 1, gameweek_from=2, gameweek_until=2)
    player = absence_db.get(Player, 1)

    assert not player.is_injured_or_suspended(TEST_SEASON, 2, 2)


def test_an_absence_that_has_not_begun_is_not_known_yet(absence_db):
    """
    A replay may not rule a player out of a fixture over an injury still to come.

    The flag lives on the gameweek's own attributes row, so planning from an
    earlier gameweek reads a row that knows nothing about it, as the real run
    would not have known either.
    """
    _set_absent(absence_db, 1, gameweek_from=3, gameweek_until=4)
    player = absence_db.get(Player, 1)

    # planning from gameweek 1: nothing has happened to them yet
    assert not player.is_injured_or_suspended(TEST_SEASON, 1, 3)


def test_an_absence_already_under_way_rules_out_the_gameweeks_it_covers(absence_db):
    """Flagged as at the gameweek being planned from, and not back by the fixture."""
    _set_absent(absence_db, 1, gameweek_from=2, gameweek_until=4)
    player = absence_db.get(Player, 1)

    assert player.is_injured_or_suspended(TEST_SEASON, 2, 2)
    assert player.is_injured_or_suspended(TEST_SEASON, 2, 3)
    # the return gameweek is the gameweek they came back in
    assert not player.is_injured_or_suspended(TEST_SEASON, 2, 4)


def test_an_absence_already_over_rules_out_nothing(absence_db):
    """The player came back before the gameweek being planned from."""
    _set_absent(absence_db, 1, gameweek_from=1, gameweek_until=3)
    player = absence_db.get(Player, 1)

    assert not player.is_injured_or_suspended(TEST_SEASON, 4, 4)


def test_a_gameweek_with_no_attributes_row_gets_one(dbsession):
    """
    An absence beginning before a player joined the league still lands somewhere.

    `player_details` has an entry only per gameweek the player was registered at
    a Premier League club, so a transfer absence covers gameweeks with no row at
    all. One row at the first gameweek missed is enough: the nearest-gameweek
    fallback in `Player.get_gameweek_attributes` covers the ones either side.
    """
    _add_player(dbsession, 1, "Arriving", gameweeks=(3, 4))
    dbsession.commit()

    _set_absent(dbsession, 1, gameweek_from=1, gameweek_until=3)
    player = dbsession.get(Player, 1)

    added = dbsession.scalars(
        select(PlayerAttributes).where(PlayerAttributes.gameweek == 1)
    ).one()
    assert added.chance_of_playing_next_round == 0
    assert added.return_gameweek == 3
    # price, team and position are carried from the nearest gameweek they have
    assert (added.price, added.team, added.position) == (50, TEAM, "MID")
    assert player.is_injured_or_suspended(TEST_SEASON, 1, 1)
    assert not player.is_injured_or_suspended(TEST_SEASON, 3, 3)


def test_a_player_with_no_attributes_at_all_is_skipped(dbsession):
    """There is no price, team or position to copy, so there is no row to make."""
    player = Player()
    player.player_id = 1
    player.name = "Never Here"
    dbsession.add(player)
    dbsession.commit()

    _set_absent(dbsession, 1, gameweek_from=1, gameweek_until=3)

    assert dbsession.scalars(select(PlayerAttributes)).all() == []
