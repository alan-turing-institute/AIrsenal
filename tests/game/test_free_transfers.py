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
from airsenal.optimization.moves import GameweekMove, calc_free_transfers
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


def test_a_wildcard_costs_no_free_transfers(transaction_db):
    """
    Fifteen players in on a wildcard is not fifteen transfers.

    The search plans with `calc_free_transfers`, which knows a chip that rebuilds
    the squad leaves the count alone. Reading the plan back out of the
    transactions table charged it for fifteen, so every gameweek after a replayed
    wildcard started on one free transfer.
    """
    _initial_squad(transaction_db, "2223")
    _rebuilt_squad(transaction_db, "2223", gameweek=3)
    transaction_db.commit()

    # GW2 idle, GW3 the wildcard, GW4 idle: three accruals from one
    assert _free_transfers(transaction_db, gameweek=5) == 4


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
