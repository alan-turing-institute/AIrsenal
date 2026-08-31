"""
That recording an entry's transfers asks the API about *that* entry.

`update_squad` takes an fpl_team_id and resolves it up front, so everything it
then asks the FPL API has to be asked for that entry. Getting it wrong is not a
read error that goes away on the next run: the answer is written into
`Transaction.free_hit`, and `get_squad_from_transactions` filters those rows out
for good.
"""

import pytest

from airsenal.squad import history

OTHER_ENTRY = 4321
GAMEWEEK = 7


class _Fetcher:
    """Returns one transfer, for whichever entry is asked about."""

    FPL_TEAM_ID = 1111

    def get_fpl_transfer_data(self, fpl_team_id):  # noqa: ARG002
        return [
            {
                "event": GAMEWEEK,
                "element_out": 10,
                "element_out_cost": 55,
                "element_in": 20,
                "element_in_cost": 60,
                "time": "2026-01-01T00:00:00Z",
            }
        ]


class _Player:
    def __init__(self, player_id):
        self.player_id = player_id


@pytest.fixture
def asked_about(monkeypatch):
    """The fpl_team_ids the free-hit lookup was asked about, in order."""
    asked = []

    def record(gameweek, fpl_team_id=None, fetcher=None):
        asked.append(fpl_team_id)
        return 0

    monkeypatch.setattr(history, "free_hit_used_in_gameweek", record)
    monkeypatch.setattr(history, "get_fetcher", lambda *a, **k: _Fetcher())
    monkeypatch.setattr(
        history, "get_player_from_api_id", lambda api_id, **k: _Player(api_id)
    )
    monkeypatch.setattr(history, "transaction_exists", lambda *a, **k: False)
    monkeypatch.setattr(history, "add_transaction", lambda *a, **k: None)
    # An entry with transactions already recorded, so the initial-squad branch -
    # which does pass the id on - is not the one under test here.
    monkeypatch.setattr(
        history,
        "record_initial_squad_transactions",
        lambda **k: pytest.fail("should not be reached"),
    )
    return asked


def test_the_free_hit_flag_is_read_for_the_entry_being_updated(
    asked_about, monkeypatch
):
    class _Session:
        def scalars(self, *args, **kwargs):
            return self

        def all(self):
            return ["an existing transaction"]

    monkeypatch.setattr(history, "get_session", _Session)

    history.update_squad(fpl_team_id=OTHER_ENTRY, dbsession=_Session())

    assert asked_about == [OTHER_ENTRY]
