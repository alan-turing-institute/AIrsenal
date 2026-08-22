"""
Transfer strategies, one per module, each registering itself by name.

`select_strategy` is the single place that decides which one a move needs.
That decision used to be an if/elif inside `make_best_transfers`, duplicated
in the progress-bar sizing, and the two could disagree.
"""

from airsenal.optimization.moves import GameweekMove
from airsenal.optimization.protocols import (
    TransferPlan,
    TransferRequest,
    TransferStrategy,
)

# importing each module runs its registration
from airsenal.optimization.strategies import (
    double,
    full_squad,
    none,
    random_search,
    single,
)
from airsenal.optimization.strategies.registry import TRANSFER_STRATEGIES, NoOptions

# How many players change decides how the search is done: one or two can be
# enumerated exhaustively, more has to be sampled, and a chip that rebuilds the
# squad goes to the genetic algorithm.
_BY_TRANSFER_COUNT = {0: "none", 1: "single", 2: "double"}
_MANY_TRANSFERS = "random"
_REBUILD = "full_squad"


def strategy_name_for(move: GameweekMove) -> str:
    """The name of the strategy that handles this move."""
    if move.rebuilds_squad:
        return _REBUILD
    return _BY_TRANSFER_COUNT.get(move.n_transfers, _MANY_TRANSFERS)


def select_strategy(move: GameweekMove) -> TransferStrategy:
    """The strategy that handles this move, with its default settings."""
    return TRANSFER_STRATEGIES.create(strategy_name_for(move))


__all__ = [
    "TRANSFER_STRATEGIES",
    "NoOptions",
    "TransferPlan",
    "TransferRequest",
    "TransferStrategy",
    "select_strategy",
    "strategy_name_for",
]
