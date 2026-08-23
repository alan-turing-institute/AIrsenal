"""
Choosing what to do in one gameweek.

The work of each option lives in `optimization/strategies/`; this module picks
the right one for a move and scores what it came back with.
"""

from airsenal.core.logging import get_logger
from airsenal.optimization.moves import GameweekMove
from airsenal.optimization.protocols import StepCounter, TransferRequest
from airsenal.optimization.squad_score import get_discounted_squad_score
from airsenal.optimization.strategies import DEFAULT_STRATEGIES, StrategySet
from airsenal.squad.squad import Squad

logger = get_logger(__name__)


def get_num_increments(
    move: GameweekMove,
    num_iterations: int = 100,
    strategies: StrategySet | None = None,
) -> int:
    """
    How many candidate squads the search will consider for this move.

    The total of the worker's progress bar, which the search then advances one
    candidate at a time. It comes from the strategy that does the searching, so
    it cannot drift away from what actually happens.
    """
    strategies = strategies if strategies is not None else DEFAULT_STRATEGIES
    return strategies.create(move).num_increments(move, num_iterations)


def make_best_transfers(
    move: GameweekMove,
    squad: Squad,
    tag: str,
    gameweeks: list[int],
    root_gw: int,
    season: str,
    num_iter: int = 100,
    on_step: StepCounter | None = None,
    strategies: StrategySet | None = None,
) -> tuple[Squad, dict[str, list[int]], float]:
    """
    Make this gameweek's move, returning the resulting squad, the transfers made
    as {"in": [player_ids], "out": [player_ids]}, and the points it is expected
    to score next gameweek.
    """
    strategies = strategies if strategies is not None else DEFAULT_STRATEGIES
    request = TransferRequest(
        move=move,
        squad=squad,
        tag=tag,
        gameweeks=gameweeks,
        root_gw=root_gw,
        season=season,
        num_iterations=num_iter,
        progress=on_step,
    )
    plan = strategies.create(move).propose(request)

    points = get_discounted_squad_score(
        plan.squad,
        [request.transfer_gameweek],
        tag,
        root_gw=root_gw,
        bench_boost_gw=request.bench_boost_gw,
        triple_captain_gw=request.triple_captain_gw,
    )

    # A free hit is reverted after the gameweek it is played in, so the squad
    # that carries on to the next gameweek is the one we started with.
    resulting_squad = plan.squad if move.carry_forward else squad
    return resulting_squad, plan.as_transfer_dict(), points
