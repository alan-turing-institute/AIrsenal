"""
Filling the attributes table from the FPL API.

Every player gets one row for the gameweek being filled, and then a row per
gameweek they have already played. The two are written in the same loop, which is
what these cover: the gameweek being filled has to survive the walk back through a
player's history, for that player and for the one after them.
"""

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from airsenal.db.models import Base, Player, PlayerAttributes
from airsenal.ingest import player_attributes as attributes_module
from airsenal.ingest.player_attributes import fill_attributes_table_from_api

SEASON = "2425"
GAMEWEEK_BEING_FILLED = 5
PLAYED_IN = [1, 2, 3, 4]


@pytest.fixture
def dbsession():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    for api_id in (101, 102):
        player = Player()
        player.player_id = api_id - 100
        player.fpl_api_id = api_id
        player.name = f"Player {api_id}"
        session.add(player)
    session.commit()
    yield session
    session.close()


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
    """
    Not for whichever gameweek the previous player's history happened to end on.

    The walk back through a player's history used to run in a variable named for
    the gameweek being filled, so after the first player every subsequent row was
    written against the last gameweek that player had played.
    """
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
