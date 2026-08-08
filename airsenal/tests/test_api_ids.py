"""
FPL reassigns element ids every season. These cover the consequences of an id
being left on the player who held it last year.
"""

from sqlalchemy import select

from airsenal.conftest import session_scope
from airsenal.framework.schema import Player
from airsenal.scripts.update_db import add_players_to_db, sync_api_ids


class FakeFetcher:
    """Stands in for the module-level fetcher in update_db."""

    def __init__(self, summary):
        self.summary = summary

    def get_player_summary_data(self):
        return self.summary


def summary(**names):
    return {
        int(api_id): {"first_name": name.split()[0], "second_name": name.split()[1]}
        for api_id, name in names.items()
    }


def add_player(dbsession, name, api_id=None):
    player = Player()
    player.name = name
    player.fpl_api_id = api_id
    dbsession.add(player)
    dbsession.flush()
    return player


def test_id_moves_to_the_player_the_api_names(monkeypatch):
    """Two players holding one id is how this season's price and team ended up on
    a player who had left the league."""
    with session_scope() as ts:
        gone = add_player(ts, "Departed Striker", api_id=8001)
        arrived = add_player(ts, "New Striker", api_id=8001)

        monkeypatch.setattr(
            "airsenal.scripts.update_db.fetcher",
            FakeFetcher(summary(**{"8001": "New Striker"})),
        )
        assert sync_api_ids(ts) == 1

        ts.refresh(gone)
        ts.refresh(arrived)
        assert gone.fpl_api_id is None
        assert arrived.fpl_api_id == 8001


def test_id_absent_from_the_api_is_released(monkeypatch):
    with session_scope() as ts:
        retired = add_player(ts, "Retired Winger", api_id=8002)

        monkeypatch.setattr(
            "airsenal.scripts.update_db.fetcher",
            FakeFetcher(summary(**{"8003": "Someone Else"})),
        )
        sync_api_ids(ts)

        ts.refresh(retired)
        assert retired.fpl_api_id is None


def test_sole_holder_keeps_their_id(monkeypatch):
    with session_scope() as ts:
        player = add_player(ts, "Current Midfielder", api_id=8004)

        monkeypatch.setattr(
            "airsenal.scripts.update_db.fetcher",
            FakeFetcher(summary(**{"8004": "Current Midfielder"})),
        )
        assert sync_api_ids(ts) == 0

        ts.refresh(player)
        assert player.fpl_api_id == 8004


def test_adding_a_player_takes_the_id_off_last_years_holder():
    """The same thing at the point of assignment, so it cannot come back."""
    with session_scope() as ts:
        old = add_player(ts, "Last Season Player", api_id=8005)

        add_players_to_db(
            players_from_db=[],
            players_from_api=[8005],
            player_data_from_api={
                8005: {"first_name": "This", "second_name": "Season"}
            },
            dbsession=ts,
        )

        ts.refresh(old)
        assert old.fpl_api_id is None
        holders = ts.scalars(select(Player).where(Player.fpl_api_id == 8005)).all()
        assert len(holders) == 1
        assert holders[0].name == "This Season"
