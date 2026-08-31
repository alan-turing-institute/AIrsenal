"""Replaying a plan's transfers to find the price each was made at."""

from airsenal.game.enums import Chip
from airsenal.game.season import CURRENT_SEASON
from airsenal.optimization.moves import GameweekMove
from airsenal.optimization.plan import GameweekOutcome, Plan
from airsenal.optimization.run_transfers import transfer_rows
from airsenal.squad.squad import Squad
from tests.conftest import session_scope

STARTING_IDS = tuple(range(15))
FREE_HIT_IDS = tuple(range(15, 30))


def _squad(dbsession):
    squad = Squad(season=CURRENT_SEASON)
    for player_id in STARTING_IDS:
        squad.add_player(
            player_id, check_budget=False, check_team=False, dbsession=dbsession
        )
    return squad


def _outcome(gameweek, move, players_out, players_in):
    return GameweekOutcome(
        gameweek=gameweek,
        move=move,
        points=0.0,
        discount_factor=1.0,
        points_hit=0,
        free_transfers=1,
        players_in=players_in,
        players_out=players_out,
    )


def test_transfer_rows_reverts_a_free_hit(fill_players):
    """
    A free hit is undone before the next gameweek's transfers are priced.

    The search plans the gameweek after a free hit from the pre-free-hit squad
    (`GameweekMove.carry_forward`), so the walk has to do the same - otherwise it
    is asked to sell a player the free-hit squad does not hold.
    """
    with session_scope() as ts:
        plan = Plan(
            root_gameweek=1,
            outcomes=(
                _outcome(
                    1,
                    GameweekMove(chip=Chip.FREE_HIT),
                    STARTING_IDS,
                    FREE_HIT_IDS,
                ),
                _outcome(2, GameweekMove(1), (0,), (30,)),
            ),
        )
        rows = transfer_rows(
            plan, _squad(ts), season=CURRENT_SEASON, use_api=False, dbsession=ts
        )

    # 15 for the free hit, then the one real transfer made off the original squad
    assert len(rows) == 16
    assert rows[-1].gameweek == 2
    assert rows[-1].sale_price is not None


def test_transfer_rows_carries_a_wildcard_forward(fill_players):
    """A wildcard is kept, so the next gameweek transfers out of the new squad."""
    with session_scope() as ts:
        plan = Plan(
            root_gameweek=1,
            outcomes=(
                _outcome(
                    1,
                    GameweekMove(chip=Chip.WILDCARD),
                    STARTING_IDS,
                    FREE_HIT_IDS,
                ),
                _outcome(2, GameweekMove(1), (15,), (30,)),
            ),
        )
        rows = transfer_rows(
            plan, _squad(ts), season=CURRENT_SEASON, use_api=False, dbsession=ts
        )

    assert len(rows) == 16
    assert rows[-1].gameweek == 2
