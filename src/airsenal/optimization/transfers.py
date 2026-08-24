"""
Choosing what to do in one gameweek.

The work of each option lives in `optimization/strategies/`; this module scores
what the chosen one came back with.
"""

from airsenal.core.logging import get_logger
from airsenal.optimization.protocols import TransferRequest, TransferStrategy
from airsenal.optimization.squad_score import get_discounted_squad_score
from airsenal.squad.squad import Squad

logger = get_logger(__name__)


def make_best_transfers(
    request: TransferRequest, strategy: TransferStrategy
) -> tuple[Squad, dict[str, list[int]], float]:
    """
    Make this gameweek's move, returning the resulting squad, the transfers made
    as {"in": [player_ids], "out": [player_ids]}, and the points it is expected
    to score next gameweek.
    """
    proposal = strategy.propose(request)

    points = get_discounted_squad_score(
        proposal.squad,
        [request.transfer_gameweek],
        request.tag,
        root_gw=request.root_gw,
        bench_boost_gw=request.bench_boost_gw,
        triple_captain_gw=request.triple_captain_gw,
    )

    # A free hit is reverted after the gameweek it is played in, so the squad
    # that carries on to the next gameweek is the one we started with.
    resulting_squad = proposal.squad if request.move.carry_forward else request.squad
    return resulting_squad, proposal.as_transfer_dict(), points
