"""
That recording an entry's transfers asks the API about *that* entry.

`update_squad` takes an fpl_team_id and resolves it up front, so everything it
then asks the FPL API has to be asked for that entry. Getting it wrong is not a
read error that goes away on the next run: the answer is written into
`Transaction.free_hit` and `Transaction.counts_as_transfer`, and both are read
back for good - the first by `get_squad_from_transactions`, which filters those
rows out, and the second by `get_free_transfers`.
"""

import pytest

from airsenal.game.enums import Chip
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
def asked_about(monkeypatch, request):
    """The fpl_team_ids the chip lookup was asked about, in order."""
    asked = []
    chip = getattr(request, "param", None)

    def record(gameweek, fpl_team_id=None, fetcher=None):
        asked.append(fpl_team_id)
        return chip

    monkeypatch.setattr(history, "chip_used_in_gameweek", record)
    monkeypatch.setattr(history, "get_fetcher", lambda *a, **k: _Fetcher())
    monkeypatch.setattr(
        history, "require_player_from_api_id", lambda api_id, **k: _Player(api_id)
    )
    monkeypatch.setattr(history, "transaction_exists", lambda *a, **k: False)
    # An entry with transactions already recorded, so the initial-squad branch -
    # which does pass the id on - is not the one under test here.
    monkeypatch.setattr(
        history,
        "record_initial_squad_transactions",
        lambda **k: pytest.fail("should not be reached"),
    )
    return asked


class _Session:
    def scalars(self, *args, **kwargs):
        return self

    def first(self):
        return "an existing transaction"


@pytest.fixture
def recorded(monkeypatch):
    """The keyword arguments every add_transaction call was made with."""
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(history, "add_transaction", record)
    monkeypatch.setattr(history, "get_session", _Session)
    return calls


def test_the_chip_played_is_read_for_the_entry_being_updated(asked_about, recorded):
    history.update_squad(fpl_team_id=OTHER_ENTRY, dbsession=_Session())

    assert asked_about == [OTHER_ENTRY]


@pytest.mark.parametrize(
    ("asked_about", "counts", "free_hit"),
    [
        (None, 1, 0),
        (Chip.WILDCARD, 0, 0),
        (Chip.FREE_HIT, 0, 1),
        # The two lineup chips are played alongside ordinary transfers, which are
        # charged for as usual.
        (Chip.BENCH_BOOST, 1, 0),
        (Chip.TRIPLE_CAPTAIN, 1, 0),
    ],
    indirect=["asked_about"],
)
def test_only_a_squad_chip_makes_a_gameweeks_transfers_free(
    asked_about, recorded, counts, free_hit
):
    """
    `counts_as_transfer` is what stops a wildcard costing fifteen free transfers.

    `get_free_transfers` reads it back to work out what the following gameweek
    starts with.
    """
    history.update_squad(fpl_team_id=OTHER_ENTRY, dbsession=_Session())

    assert [call["counts_as_transfer"] for call in recorded] == [counts, counts]
    assert [call["free_hit"] for call in recorded] == [free_hit, free_hit]
