"""
Filling the attributes table from the FPL API.

Every player gets one row for the gameweek being filled, and then a row per
gameweek they have already played. The two are written in the same loop, which is
what these cover: the gameweek being filled has to survive the walk back through a
player's history, for that player and for the one after them.
"""

from datetime import date

import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from airsenal.core.caching import clear_query_caches
from airsenal.db.models import Base, Fixture, Player, PlayerAttributes, PlayerMapping
from airsenal.ingest import player_attributes as attributes_module
from airsenal.ingest.attributes_history import Availability
from airsenal.ingest.player_attributes import (
    fill_attributes_table_from_api,
    fill_attributes_table_from_file,
    fill_availability_for_season,
)

SEASON = "2425"
GAMEWEEK_BEING_FILLED = 5
PLAYED_IN = [1, 2, 3, 4]


@pytest.fixture
def dbsession():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    # `autoflush=False`, as `db.session.create_session` does. It is not a detail:
    # with autoflush on, a query flushes whatever the ingest has just added and
    # so always sees it, which is exactly the thing the real session does not do.
    session = sessionmaker(bind=engine, autoflush=False)()
    for api_id in (101, 102):
        player = Player()
        player.player_id = api_id - 100
        player.fpl_api_id = api_id
        player.name = f"Player {api_id}"
        session.add(player)
    session.commit()
    yield session
    session.close()


def test_a_player_under_two_names_is_filled_once(dbsession):
    """A past season's file can list one person under an old spelling as well."""
    rows = [{"gameweek": "1", "value": "45", "played_for": "BRE", "position": "MID"}]
    mapping = PlayerMapping()
    mapping.player_id = 1
    mapping.alt_name = "Old Spelling"
    dbsession.add(mapping)
    dbsession.commit()

    fill_attributes_table_from_file(
        {"Player 101": rows, "Old Spelling": rows}, season=SEASON, dbsession=dbsession
    )
    assert len(dbsession.scalars(select(PlayerAttributes)).all()) == 1


class FakeFetcher:
    """The three calls `fill_attributes_table_from_api` makes, and nothing else."""

    def __init__(self, api_ids):
        self._api_ids = api_ids

    def get_current_summary_data(self):
        return {"total_players": 1_000_000}

    def get_player_summary_data(self):
        return {
            api_id: {
                "element_type": 3,
                "now_cost": 50 + api_id,
                "team": 1,
                "selected_by_percent": "10.0",
                "transfers_in": 100,
                "transfers_out": 50,
                "news": "",
                "chance_of_playing_next_round": None,
                "opta_code": f"p{api_id}",
            }
            for api_id in self._api_ids
        }

    def get_gameweek_data_for_player(self, player_api_id, gameweek=None):  # noqa: ARG002
        """As the real one: every gameweek the player has played, keyed by gameweek."""
        return {
            played_in: [
                {
                    "round": played_in,
                    "value": 50,
                    "opponent_team": 2,
                    "was_home": True,
                    "kickoff_time": "2024-09-01T14:00:00Z",
                    "transfers_balance": 0,
                    "selected": 1,
                    "transfers_in": 0,
                    "transfers_out": 0,
                }
            ]
            for played_in in PLAYED_IN
        }


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(
        attributes_module, "get_fetcher", lambda: FakeFetcher([101, 102])
    )
    monkeypatch.setattr(
        attributes_module, "next_gameweek", lambda *a, **k: GAMEWEEK_BEING_FILLED
    )
    monkeypatch.setattr(attributes_module, "get_team_name", lambda *a, **k: "ARS")
    # The history rows need a fixture to be attributable to a team; without one
    # the walk logs and moves on, which is all this needs it to do.
    monkeypatch.setattr(attributes_module, "find_fixture", lambda *a, **k: None)


def current_rows(dbsession):
    """The attributes written for the gameweek being filled, by player."""
    rows = dbsession.scalars(
        select(PlayerAttributes).where(
            PlayerAttributes.gameweek == GAMEWEEK_BEING_FILLED
        )
    ).all()
    return {row.player_id: row for row in rows}


def test_every_player_gets_a_row_for_the_gameweek_being_filled(dbsession, api):
    """Not for whichever gameweek the previous player's history happened to end on."""
    fill_attributes_table_from_api(SEASON, dbsession=dbsession)
    assert sorted(current_rows(dbsession)) == [1, 2]


def test_the_second_players_row_carries_that_players_own_price(dbsession, api):
    """A row filed under the wrong gameweek takes its player's data with it."""
    fill_attributes_table_from_api(SEASON, dbsession=dbsession)
    rows = current_rows(dbsession)
    assert rows[1].price == 151
    assert rows[2].price == 152


def test_nothing_is_written_for_a_gameweek_that_was_only_walked_over(dbsession, api):
    """
    The history walk writes rows of its own, but only where it found a fixture.

    With no fixture it writes none, so any row in a past gameweek here is the
    current-gameweek row having landed in the wrong place.
    """
    fill_attributes_table_from_api(SEASON, dbsession=dbsession)
    written = dbsession.scalars(
        select(PlayerAttributes.gameweek).where(
            PlayerAttributes.gameweek != GAMEWEEK_BEING_FILLED
        )
    ).all()
    assert written == []


# --------------------------------------- combining the two availability sources


AVAILABILITY_SEASON = "2526"
AVAILABILITY_GAMEWEEK_DATES = {
    1: "2025-08-16T14:00:00Z",
    2: "2025-08-23T14:00:00Z",
}


@pytest.fixture
def availability_db(monkeypatch):
    """One player with a row in each of two gameweeks, and a fixture for each."""
    clear_query_caches()
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    player = Player()
    player.player_id = 1
    player.name = "Bob"
    player.opta_code = "p1"
    session.add(player)
    for gameweek, kickoff in AVAILABILITY_GAMEWEEK_DATES.items():
        fixture = Fixture()
        fixture.date = kickoff
        fixture.gameweek = gameweek
        fixture.home_team = "ARS"
        fixture.away_team = "CHE"
        fixture.season = AVAILABILITY_SEASON
        fixture.tag = "test"
        session.add(fixture)

        attributes = PlayerAttributes()
        attributes.player = player
        attributes.player_id = 1
        attributes.season = AVAILABILITY_SEASON
        attributes.gameweek = gameweek
        attributes.price = 50
        attributes.team = "ARS"
        attributes.position = "MID"
        session.add(attributes)
    session.commit()
    yield session
    session.close()
    clear_query_caches()


def _fake_history(monkeypatch, rows):
    frame = pd.DataFrame(
        rows,
        columns=[
            "day",
            "opta_code",
            "player",
            "news",
            "chance_of_playing_next_round",
            "return_gameweek",
        ],
    )
    monkeypatch.setattr(
        attributes_module, "load_attributes_history", lambda *a, **k: frame
    )


def _absences(monkeypatch, availability):
    monkeypatch.setattr(
        attributes_module,
        "get_availability_from_absences",
        lambda *a, **k: dict(availability),
    )


def test_the_history_wins_where_it_has_something_to_say(availability_db, monkeypatch):
    """
    The history is what the FPL API reported at the time.

    The absences csv is a retrospective Transfermarkt scrape, so where the two
    disagree about a gameweek the history is the better answer - including when
    it says the player was fine and the scrape says they were not.
    """
    _absences(
        monkeypatch,
        {
            (1, 1): Availability("Scraped injury", 0, 3),
            (1, 2): Availability("Scraped injury", 0, 3),
        },
    )
    _fake_history(monkeypatch, [(date(2025, 8, 16), "p1", "Bob", None, 100, None)])

    fill_availability_for_season(AVAILABILITY_SEASON, availability_db)

    rows = {
        row.gameweek: row
        for row in availability_db.scalars(select(PlayerAttributes)).all()
    }
    assert rows[1].chance_of_playing_next_round == 100
    assert rows[1].news is None
    # gameweek 2 is not in the history, so the scrape stands
    assert rows[2].chance_of_playing_next_round == 0
    assert rows[2].news == "Scraped injury"


def test_the_absences_csv_fills_the_gameweeks_the_history_does_not_reach(
    availability_db, monkeypatch
):
    """The daily dump started partway through 25/26, so its first gameweeks need it."""
    _absences(monkeypatch, {(1, 1): Availability("Knee injury", 0, 2)})
    _fake_history(monkeypatch, [])

    fill_availability_for_season(AVAILABILITY_SEASON, availability_db)

    player = availability_db.get(Player, 1)
    assert player.is_injured_or_suspended(AVAILABILITY_SEASON, 1, 1)
    assert not player.is_injured_or_suspended(AVAILABILITY_SEASON, 2, 2)


# ------------- the window between a deadline passing and the first kickoff ---


class DeadlinePassedFetcher(FakeFetcher):
    """
    The API once a deadline has passed but before the gameweek has kicked off.

    It flips the gameweek to current and gives every player a history row for
    it, carrying their price and ownership as at the deadline. `next_gameweek`
    still answers with that gameweek, because none of its fixtures has started.
    """

    def get_gameweek_data_for_player(self, player_api_id, gameweek=None):  # noqa: ARG002
        return {
            played_in: [
                {
                    "round": played_in,
                    "value": 50,
                    "opponent_team": 2,
                    "was_home": True,
                    "kickoff_time": "2024-09-01T14:00:00Z",
                    "transfers_balance": 0,
                    "selected": 1,
                    "transfers_in": 0,
                    "transfers_out": 0,
                }
            ]
            for played_in in [*PLAYED_IN, GAMEWEEK_BEING_FILLED]
        }


@pytest.fixture
def api_mid_deadline(monkeypatch):
    """As `api`, but the history reaches the gameweek being filled and writes."""
    monkeypatch.setattr(
        attributes_module, "get_fetcher", lambda: DeadlinePassedFetcher([101, 102])
    )
    monkeypatch.setattr(
        attributes_module, "next_gameweek", lambda *a, **k: GAMEWEEK_BEING_FILLED
    )
    monkeypatch.setattr(attributes_module, "get_team_name", lambda *a, **k: "ARS")
    monkeypatch.setattr(attributes_module, "find_fixture", lambda *a, **k: object())
    monkeypatch.setattr(
        attributes_module, "get_player_team_from_fixture", lambda *a, **k: "ARS"
    )


def test_the_gameweek_being_filled_gets_one_row_not_two(dbsession, api_mid_deadline):
    """
    A deadline that has passed puts that gameweek in the player's history too.

    The session does not autoflush, so a query in the history walk cannot see the
    row the summary data has just added. A second row would break the unique
    constraint on (player, season, gameweek) for the whole window between a
    deadline and its first kickoff.
    """
    fill_attributes_table_from_api(SEASON, dbsession=dbsession)

    rows = dbsession.scalars(
        select(PlayerAttributes).where(
            PlayerAttributes.gameweek == GAMEWEEK_BEING_FILLED
        )
    ).all()
    assert sorted(row.player_id for row in rows) == [1, 2]


def test_the_summary_data_wins_for_the_gameweek_being_filled(
    dbsession, api_mid_deadline
):
    """
    It is the live one: the history row holds the price as at the deadline.

    It also holds no availability at all, so letting it through would drop the
    news and chance of playing the summary data had just written.
    """
    fill_attributes_table_from_api(SEASON, dbsession=dbsession)

    rows = current_rows(dbsession)
    assert rows[1].price == 151
    assert rows[2].price == 152
