"""
How many free transfers an entry has.

The search accrues free transfers with `calc_free_transfers`; the count it starts
from comes from `get_free_transfers`. Both go through
`game.scoring.free_transfers_after`.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from airsenal.db.models import Base, Transaction
from airsenal.game.enums import Chip
from airsenal.game.scoring import MAX_FREE_TRANSFERS, free_transfers_after
from airsenal.game.season import CURRENT_SEASON
from airsenal.optimization.moves import GameweekMove, calc_free_transfers
from airsenal.remote.errors import RemoteError
from airsenal.squad import state
from airsenal.squad.state import get_free_transfers


@pytest.mark.parametrize(
    ("n_transfers", "previous", "expected"),
    [
        (0, 1, 2),  # an idle week accrues one
        (0, 4, 5),  # up to the cap
        (0, 5, 5),  # and no further
        (1, 3, 3),  # one transfer spends the one accrued
        (2, 3, 2),  # two spends one of the bank
        (5, 1, 1),  # never below one
    ],
)
def test_free_transfers_accrue_to_the_cap(n_transfers, previous, expected):
    assert free_transfers_after(n_transfers, previous) == expected


def test_idle_weeks_reach_the_documented_maximum():
    """
    Four idle gameweeks from one free transfer reach MAX_FREE_TRANSFERS.

    The estimate used to stop at 2, so the search was told it had 2 and then
    charged a points hit for moves FPL would have given away.
    """
    free_transfers = 1
    for _ in range(4):
        free_transfers = free_transfers_after(0, free_transfers)
    assert free_transfers == MAX_FREE_TRANSFERS


@pytest.mark.parametrize("chip", [Chip.WILDCARD, Chip.FREE_HIT])
def test_rebuilding_the_squad_leaves_the_count_alone(chip):
    # Changed in 24/25: playing a wildcard or free hit no longer resets you to 1.
    assert free_transfers_after(15, 3, rebuilds_squad=True) == 3
    assert calc_free_transfers(GameweekMove(chip=chip), 3) == 3


@pytest.mark.parametrize("max_free_transfers", [2, 5])
@pytest.mark.parametrize("n_transfers", range(6))
@pytest.mark.parametrize("prev_free_transfers", range(6))
def test_the_count_stays_between_one_and_the_cap(
    max_free_transfers, n_transfers, prev_free_transfers
):
    """Whatever the cap, and whatever is spent, the count never leaves the range."""
    got = calc_free_transfers(
        GameweekMove(n_transfers), prev_free_transfers, max_free_transfers
    )
    assert 1 <= got <= max_free_transfers


def test_the_move_shaped_wrapper_agrees_with_the_rule():
    """`calc_free_transfers` is a face on the same arithmetic, not a second copy."""
    for n_transfers in range(4):
        for previous in range(1, MAX_FREE_TRANSFERS + 1):
            assert calc_free_transfers(
                GameweekMove(n_transfers), previous
            ) == free_transfers_after(n_transfers, previous)


# --- the count read back out of the transactions table ---


@pytest.fixture
def transaction_db():
    """An empty database holding nothing but the transactions a test adds."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _bought(
    dbsession, season, gameweek, player_id, fpl_team_id=-1, counts_as_transfer=1
):
    transaction = Transaction()
    transaction.player_id = player_id
    transaction.gameweek = gameweek
    transaction.season = season
    transaction.bought_or_sold = 1
    transaction.price = 50
    transaction.free_hit = 0
    transaction.counts_as_transfer = counts_as_transfer
    transaction.fpl_team_id = fpl_team_id
    transaction.tag = "test"
    transaction.time = "2024-01-01T00:00:00"
    dbsession.add(transaction)


def _initial_squad(dbsession, season, gameweek=1, fpl_team_id=-1):
    """The fifteen an entry starts with, which cost it no transfers."""
    for player_id in range(1, 16):
        _bought(
            dbsession, season, gameweek, player_id, fpl_team_id, counts_as_transfer=0
        )


def _rebuilt_squad(dbsession, season, gameweek, fpl_team_id=-1):
    """Fifteen players in on a wildcard or free hit: unlimited, so free."""
    for player_id in range(100, 115):
        _bought(
            dbsession, season, gameweek, player_id, fpl_team_id, counts_as_transfer=0
        )


def _free_transfers(dbsession, gameweek, season="2223"):
    return get_free_transfers(
        gameweek=gameweek,
        season=season,
        fpl_team_id=-1,
        dbsession=dbsession,
        is_replay=True,
    )


def test_only_this_seasons_transfers_are_counted(transaction_db):
    """
    Two replays of different seasons share a dummy fpl_team_id.

    `replay.get_dummy_id` picks the lowest unused id *within the season*, so
    replaying 2223 and then 2324 files both under -1. Counting every row for the
    id charged the second replay for the first one's transfers: it had made none
    of its own by GW6 and should have banked the maximum.
    """
    _initial_squad(transaction_db, "2223")
    for gameweek in range(2, 6):
        _bought(transaction_db, "2223", gameweek, 100 + gameweek)
    _initial_squad(transaction_db, "2324")
    transaction_db.commit()

    assert (
        get_free_transfers(
            gameweek=6,
            season="2324",
            fpl_team_id=-1,
            dbsession=transaction_db,
            is_replay=True,
        )
        == MAX_FREE_TRANSFERS
    )


def test_a_transfer_every_gameweek_never_banks_one(transaction_db):
    """The counting itself, on a season with transfers in it."""
    _initial_squad(transaction_db, "2223")
    for gameweek in range(2, 6):
        _bought(transaction_db, "2223", gameweek, 100 + gameweek)
    transaction_db.commit()

    assert (
        get_free_transfers(
            gameweek=6,
            season="2223",
            fpl_team_id=-1,
            dbsession=transaction_db,
            is_replay=True,
        )
        == 1
    )


def test_a_wildcard_leaves_the_count_where_it_was(transaction_db):
    """
    Fifteen players in on a wildcard is not fifteen transfers, nor an idle week.

    The search plans with `calc_free_transfers`, which freezes the count across a
    chip that rebuilds the squad. Reading the plan back charged it for fifteen,
    and then, once the rows were skipped, credited it with the accrual an idle
    gameweek earns - so the read-back and the search disagreed either way.
    """
    _initial_squad(transaction_db, "2223")
    _rebuilt_squad(transaction_db, "2223", gameweek=3)
    transaction_db.commit()

    # GW2 and GW4 idle, so one accrual each; GW3's wildcard neither spends nor
    # accrues, leaving the two the count had reached.
    assert _free_transfers(transaction_db, gameweek=5) == 3


def test_the_read_back_agrees_with_the_search_across_a_wildcard(transaction_db):
    """The two halves of the rule, on the same wildcard, must give one answer."""
    _initial_squad(transaction_db, "2223")
    _rebuilt_squad(transaction_db, "2223", gameweek=3)
    transaction_db.commit()

    before = _free_transfers(transaction_db, gameweek=3)
    assert _free_transfers(transaction_db, gameweek=4) == calc_free_transfers(
        GameweekMove(chip=Chip.WILDCARD), before
    )


def test_the_opening_fifteen_cost_no_free_transfers(transaction_db):
    """An entry starts its second gameweek with one free transfer, not none."""
    _initial_squad(transaction_db, "2223")
    transaction_db.commit()

    assert _free_transfers(transaction_db, gameweek=2) == 1


def test_an_ordinary_transfer_still_costs_one(transaction_db):
    """The flag does not make everything free."""
    _initial_squad(transaction_db, "2223")
    _bought(transaction_db, "2223", 2, 101)
    _bought(transaction_db, "2223", 3, 102)
    transaction_db.commit()

    # one free transfer in GW2 and in GW3, both spent, so GW4 has one again
    assert _free_transfers(transaction_db, gameweek=4) == 1


# --- the count read from the FPL API ---


class StubFetcher:
    """
    Stands in for `FPLDataFetcher` over the two API paths.

    `live_count` is what a logged in fetcher reports; `None` makes that call fail
    the way an entry with no credentials does, so the history estimate is used.
    """

    FPL_TEAM_ID = 123

    def __init__(self, played, live_count=None, chips=None):
        self.played, self.live_count, self.chips = played, live_count, chips or {}

    def get_num_free_transfers(self, fpl_team_id=None):  # noqa: ARG002
        if self.live_count is None:
            msg = "not logged in"
            raise RemoteError(msg)
        return self.live_count

    def get_fpl_team_history_data(self, team_id=None):  # noqa: ARG002
        return {
            "current": [
                {"event": gameweek, "event_transfers": n, "bank": 0}
                for gameweek, n in sorted(self.played.items())
            ],
            "chips": [
                {"name": name, "event": gameweek}
                for gameweek, name in sorted(self.chips.items())
            ],
        }


@pytest.fixture
def live_season(monkeypatch):
    """The live season, up to gameweek 10, with the entry lookup stubbed out."""
    monkeypatch.setattr(state, "next_gameweek", lambda *a, **k: 10)

    def _entry_at(gameweek):
        monkeypatch.setattr(state, "get_entry_start_gameweek", lambda *a, **k: gameweek)

    return _entry_at


def _api_free_transfers(fetcher, gameweek, dbsession):
    return get_free_transfers(
        gameweek=gameweek,
        season=CURRENT_SEASON,
        fpl_team_id=123,
        fetcher=fetcher,
        dbsession=dbsession,
    )


def test_the_logged_in_count_wins_for_the_gameweek_being_played(
    live_season, transaction_db
):
    """What the game itself reports beats any estimate we could make."""
    live_season(1)
    fetcher = StubFetcher(played=dict.fromkeys(range(1, 10), 0), live_count=3)

    assert _api_free_transfers(fetcher, gameweek=10, dbsession=transaction_db) == 3


def test_the_logged_in_count_is_not_used_for_a_later_gameweek(
    live_season, transaction_db
):
    """
    It is the count for the gameweek being played, not for one further ahead.

    `--gameweek-start` can ask about a gameweek beyond the next one, and the
    logged in endpoint has no answer for that - so the estimate is used instead.
    """
    live_season(1)
    played = {1: 15} | dict.fromkeys(range(2, 12), 0)
    fetcher = StubFetcher(played=played, live_count=3)

    # ten idle gameweeks have banked the maximum, whatever today's count is
    got = _api_free_transfers(fetcher, gameweek=12, dbsession=transaction_db)
    assert got == MAX_FREE_TRANSFERS
    assert got != fetcher.live_count


def test_the_estimate_freezes_the_count_across_a_wildcard(live_season, transaction_db):
    """
    A wildcard's transfers are not charged, and do not accrue either.

    Without the chip list the twelve transfers looked ordinary and collapsed the
    count to one; skipping them alone would have credited an idle gameweek.
    """
    live_season(1)
    fetcher = StubFetcher(
        played={1: 15, 2: 0, 3: 12, 4: 0}, chips={3: "wildcard"}, live_count=None
    )

    # GW2 and GW4 accrue; GW3 leaves the count at the two it had reached
    assert _api_free_transfers(fetcher, gameweek=5, dbsession=transaction_db) == 3


def test_the_estimate_starts_from_the_gameweek_the_entry_joined(
    live_season, transaction_db
):
    """An entry joining in GW5 has one free transfer in GW6, not five."""
    live_season(5)
    fetcher = StubFetcher(played={5: 15, 6: 0, 7: 0}, live_count=None)

    assert _api_free_transfers(fetcher, gameweek=6, dbsession=transaction_db) == 1
    assert _api_free_transfers(fetcher, gameweek=8, dbsession=transaction_db) == 3
