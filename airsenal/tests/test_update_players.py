"""
Tests for adding players the FPL API has but the database doesn't - which is what
every new season starts with.
"""

from sqlalchemy import select

from airsenal.framework.schema import Player, PlayerMapping
from airsenal.scripts.fill_player_mappings_table import mappings_data
from airsenal.scripts.update_db import add_players_to_db


def a_player_with_alternative_names() -> str:
    """A name from the alternative-names CSV, so add_mappings has something to do.

    The null player_id bug only fires for players that appear in that file, which is
    why it survived several seasons of new players being added.
    """
    for row in mappings_data:
        if len(row) > 1 and all(row):
            return row[0]
    msg = "no usable row in alternative_player_names.csv"
    raise AssertionError(msg)


def test_new_player_with_mappings_gets_a_player_id(isolated_session):
    """A new player whose name has known alternatives must get mappings that point
    at a real player_id - writing them before the flush violates NOT NULL."""
    ts = isolated_session
    name = a_player_with_alternative_names()
    first, _, second = name.partition(" ")
    api_id = 999999

    added = add_players_to_db(
        players_from_db=[],
        players_from_api=[api_id],
        player_data_from_api={api_id: {"first_name": first, "second_name": second}},
        dbsession=ts,
    )
    assert added == 1

    player = ts.scalars(select(Player).where(Player.fpl_api_id == api_id)).one()
    assert player.player_id is not None

    mappings = ts.scalars(
        select(PlayerMapping).where(PlayerMapping.player_id == player.player_id)
    ).all()
    assert mappings, "expected alternative names to be recorded"
    assert all(m.player_id == player.player_id for m in mappings)


def test_existing_player_is_updated_not_duplicated(isolated_session):
    """Matching on name means a player already in the db gets their api id filled
    in, rather than a second row appearing."""
    ts = isolated_session
    api_id = 999998

    existing = Player()
    existing.name = "Testy McTestface"
    ts.add(existing)
    ts.flush()
    player_id = existing.player_id

    add_players_to_db(
        players_from_db=[],
        players_from_api=[api_id],
        player_data_from_api={
            api_id: {"first_name": "Testy", "second_name": "McTestface"}
        },
        dbsession=ts,
    )

    matches = ts.scalars(select(Player).where(Player.name == "Testy McTestface")).all()
    assert len(matches) == 1
    assert matches[0].player_id == player_id
    assert matches[0].fpl_api_id == api_id
