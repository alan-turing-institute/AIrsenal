"""
Rebuilding the squad from the transaction history.

Here rather than in a `tests/squad/` mirror because it needs the seeded league
from `tests/e2e/conftest.py` to supply the players; the transactions are written
per test, because their shape is what is being checked.
"""

import pytest
from sqlalchemy import delete

from airsenal.db.models import Transaction
from airsenal.game.enums import Position
from airsenal.squad.history import get_squad_from_transactions
from tests.e2e.conftest import PAST_SEASONS

SEASON = PAST_SEASONS[-1]
TEAM_ID = -99
GAMEWEEK = 5


def buy(session, player_id, gameweek, price, free_hit=0, team_id=TEAM_ID):
    session.add(
        Transaction(
            player_id=player_id,
            gameweek=gameweek,
            bought_or_sold=1,
            season=SEASON,
            tag="test",
            price=price,
            free_hit=free_hit,
            fpl_team_id=team_id,
            time="2024-08-01T00:00:00",
        )
    )


def sell(session, player_id, gameweek, price, free_hit=0, team_id=TEAM_ID):
    session.add(
        Transaction(
            player_id=player_id,
            gameweek=gameweek,
            bought_or_sold=-1,
            season=SEASON,
            tag="test",
            price=price,
            free_hit=free_hit,
            fpl_team_id=team_id,
            time="2024-08-01T00:00:00",
        )
    )


@pytest.fixture
def session(pipeline_db):
    """The seeded database, with the transaction table cleared between tests."""
    pipeline_db.execute(delete(Transaction))
    pipeline_db.commit()
    yield pipeline_db
    pipeline_db.execute(delete(Transaction))
    pipeline_db.commit()


# a legal starting fifteen from the seeded league: 2 GK, 5 DEF, 5 MID, 3 FWD
SQUAD_IDS = [0, 1, *range(8, 13), *range(24, 29), 40, 41, 42]


def fill_squad(session, gameweek=1, price=50):
    for player_id in SQUAD_IDS:
        buy(session, player_id, gameweek, price)
    session.commit()


def test_the_initial_squad_is_rebuilt(session):
    fill_squad(session)
    squad = get_squad_from_transactions(GAMEWEEK, SEASON, TEAM_ID, session)
    assert {p.player_id for p in squad.players} == set(SQUAD_IDS)


def test_a_sold_player_is_gone_and_the_bought_one_is_there(session):
    fill_squad(session)
    sell(session, 42, 3, 55)
    buy(session, 43, 3, 60)
    session.commit()

    squad = get_squad_from_transactions(GAMEWEEK, SEASON, TEAM_ID, session)
    ids = {p.player_id for p in squad.players}
    assert 42 not in ids
    assert 43 in ids
    assert len(squad.players) == len(SQUAD_IDS)


def test_transactions_at_or_after_the_gameweek_are_ignored(session):
    """
    The squad wanted is the one *before* the gameweek being optimised.

    Including the gameweek's own transfers would price a transfer against the
    squad it produces rather than the one it starts from.
    """
    fill_squad(session)
    sell(session, 42, GAMEWEEK, 55)
    buy(session, 43, GAMEWEEK, 60)
    session.commit()

    squad = get_squad_from_transactions(GAMEWEEK, SEASON, TEAM_ID, session)
    ids = {p.player_id for p in squad.players}
    assert 42 in ids
    assert 43 not in ids


def test_free_hit_transfers_are_skipped(session):
    """A free hit lasts one gameweek, so it never changes the standing squad."""
    fill_squad(session)
    sell(session, 42, 3, 55, free_hit=1)
    buy(session, 43, 3, 60, free_hit=1)
    session.commit()

    squad = get_squad_from_transactions(GAMEWEEK, SEASON, TEAM_ID, session)
    ids = {p.player_id for p in squad.players}
    assert 42 in ids
    assert 43 not in ids


def test_another_entrys_transactions_are_not_mixed_in(session):
    fill_squad(session)
    buy(session, 44, 2, 60, team_id=TEAM_ID - 1)
    session.commit()

    squad = get_squad_from_transactions(GAMEWEEK, SEASON, TEAM_ID, session)
    assert 44 not in {p.player_id for p in squad.players}


def test_the_purchase_price_is_carried_onto_the_player(session):
    """`get_sell_price` reads this back, so it is what a real sale is priced on."""
    fill_squad(session, price=50)
    sell(session, 42, 3, 55)
    buy(session, 43, 3, 71)
    session.commit()

    squad = get_squad_from_transactions(GAMEWEEK, SEASON, TEAM_ID, session)
    bought = next(p for p in squad.players if p.player_id == 43)
    assert bought.purchase_price == 71


def test_no_transactions_at_all_is_an_error_not_an_empty_squad(session):
    with pytest.raises(ValueError, match="No transactions in database"):
        get_squad_from_transactions(GAMEWEEK, SEASON, TEAM_ID, session)


def test_the_rebuilt_squad_has_a_legal_shape(session):
    fill_squad(session)
    squad = get_squad_from_transactions(GAMEWEEK, SEASON, TEAM_ID, session)
    counts = dict.fromkeys(Position, 0)
    for player in squad.players:
        counts[Position(player.position)] += 1
    assert counts == {Position.GK: 2, Position.DEF: 5, Position.MID: 5, Position.FWD: 3}
