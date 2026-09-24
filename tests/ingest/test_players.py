"""
Filling the player table from the packaged season files.

`test_the_packaged_file_still_has_the_keys_the_parser_reads` is the important one
- it is what stops the small fixtures below drifting away from the real data.
"""

import json

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from airsenal.core.data_files import data_file
from airsenal.db.models import Base, Player, PlayerMapping
from airsenal.ingest.players import (
    fill_player_table_from_file,
    find_player_in_table,
)

SEASON = "2425"


@pytest.fixture
def dbsession():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    # `autoflush=False`, as `db.session.create_session` does, so a query does not
    # see what the ingest has added and not yet committed.
    session = sessionmaker(bind=engine, autoflush=False)()
    yield session
    session.close()


@pytest.fixture
def players_file(tmp_path):
    """A handful of players in the shape of a packaged season file."""
    path = tmp_path / f"player_summary_{SEASON}.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "Bukayo Saka",
                    "team": "ARS",
                    "position": "MID",
                    "opta_code": "p1",
                },
                {
                    "name": "Cole Palmer",
                    "team": "CHE",
                    "position": "MID",
                    "opta_code": "p2",
                },
                {"name": "No Opta Code", "team": "BHA", "position": "DEF"},
            ]
        )
    )
    return path


def test_players_are_added(dbsession, players_file):
    fill_player_table_from_file(players_file, SEASON, dbsession)
    names = {p.name for p in dbsession.scalars(select(Player)).all()}
    assert names == {"Bukayo Saka", "Cole Palmer", "No Opta Code"}


def test_the_opta_code_is_kept(dbsession, players_file):
    fill_player_table_from_file(players_file, SEASON, dbsession)
    saka = dbsession.scalars(select(Player).where(Player.name == "Bukayo Saka")).one()
    assert saka.opta_code == "p1"


def test_a_player_without_an_opta_code_is_still_added(dbsession, players_file):
    """The key is optional: 15/16 and 16/17 have no dump to build one from."""
    fill_player_table_from_file(players_file, SEASON, dbsession)
    player = dbsession.scalars(
        select(Player).where(Player.name == "No Opta Code")
    ).one()
    assert player.opta_code is None


def test_running_it_twice_does_not_duplicate_anyone(dbsession, players_file):
    """Seasons overlap, so every player is offered to this more than once."""
    fill_player_table_from_file(players_file, SEASON, dbsession)
    fill_player_table_from_file(players_file, SEASON, dbsession)
    assert len(dbsession.scalars(select(Player)).all()) == 3


def test_a_renamed_player_matches_on_opta_code(dbsession, tmp_path):
    """
    The same person under two names is one row, found by opta code.

    This is the whole reason the opta code is stored.
    """
    first = tmp_path / "a.json"
    first.write_text(json.dumps([{"name": "Bukayo Saka", "opta_code": "p1"}]))
    second = tmp_path / "b.json"
    second.write_text(json.dumps([{"name": "B. Saka", "opta_code": "p1"}]))

    fill_player_table_from_file(first, SEASON, dbsession)
    fill_player_table_from_file(second, SEASON, dbsession)
    assert len(dbsession.scalars(select(Player)).all()) == 1


def test_a_renamed_player_keeps_the_other_name(dbsession, tmp_path):
    """The season's other files name the player the way its summary does."""
    first = tmp_path / "a.json"
    first.write_text(json.dumps([{"name": "Mitoma Kaoru", "opta_code": "p1"}]))
    second = tmp_path / "b.json"
    second.write_text(json.dumps([{"name": "Kaoru Mitoma", "opta_code": "p1"}]))

    fill_player_table_from_file(first, SEASON, dbsession)
    fill_player_table_from_file(second, SEASON, dbsession)
    fill_player_table_from_file(second, SEASON, dbsession)
    player = dbsession.scalars(select(Player)).one()
    assert find_player_in_table("Kaoru Mitoma", dbsession) is player
    assert len(dbsession.scalars(select(PlayerMapping)).all()) == 1


def test_two_people_with_one_name_stay_apart(dbsession, tmp_path):
    first = tmp_path / "a.json"
    first.write_text(json.dumps([{"name": "Ben Davies", "opta_code": "p1"}]))
    second = tmp_path / "b.json"
    second.write_text(json.dumps([{"name": "Ben Davies", "opta_code": "p2"}]))

    fill_player_table_from_file(first, SEASON, dbsession)
    fill_player_table_from_file(second, SEASON, dbsession)
    codes = {p.opta_code for p in dbsession.scalars(select(Player)).all()}
    assert codes == {"p1", "p2"}


def test_an_alternative_name_does_not_join_two_opta_codes(dbsession):
    player = Player()
    player.name = "Josh King"
    player.opta_code = "p1"
    dbsession.add(player)
    dbsession.commit()
    mapping = PlayerMapping()
    mapping.player_id = player.player_id
    mapping.alt_name = "Joshua King"
    dbsession.add(mapping)
    dbsession.commit()

    assert find_player_in_table("Joshua King", dbsession, opta_code="p2") is None
    assert find_player_in_table("Joshua King", dbsession, opta_code="p1") is player


def test_find_player_in_table_matches_an_alternative_name(dbsession):
    player = Player()
    player.name = "Bukayo Saka"
    dbsession.add(player)
    dbsession.commit()
    mapping = PlayerMapping()
    mapping.player_id = player.player_id
    mapping.alt_name = "Saka"
    dbsession.add(mapping)
    dbsession.commit()

    assert find_player_in_table("Saka", dbsession) is player


def test_find_player_in_table_returns_none_for_a_stranger(dbsession):
    assert find_player_in_table("Nobody At All", dbsession) is None


def test_the_packaged_file_still_has_the_keys_the_parser_reads():
    """
    The fixtures above are hand-written, so this checks them against reality.

    `fill_player_table_from_file` reads "name" and, optionally, "opta_code". If a
    future dump stops providing them, the tests above would keep passing while
    `airsenal db create` silently produced nameless players.
    """
    packaged = json.loads(data_file(f"player_summary_{SEASON}.json").read_text())
    assert packaged, "packaged season file is empty"
    assert all("name" in entry for entry in packaged)
    assert any(entry.get("opta_code") for entry in packaged)


@pytest.mark.parametrize("season", ["2324", "2425", "2526"])
def test_every_player_in_a_default_season_has_an_opta_code(season):
    """Without one, a renamed player becomes a second row. See `opta_code`."""
    packaged = json.loads(data_file(f"player_summary_{season}.json").read_text())
    assert all(entry.get("opta_code") for entry in packaged)
