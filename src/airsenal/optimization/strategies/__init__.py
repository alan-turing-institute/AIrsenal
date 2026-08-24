"""
Transfer strategies: one module per way of choosing a gameweek's transfers.

`StrategySet` is the single place that decides which one a move needs. That
decision used to be an if/elif inside `make_best_transfers`, duplicated in the
progress-bar sizing, and the two could disagree.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from airsenal.core.lookup import lookup
from airsenal.optimization.moves import GameweekMove
from airsenal.optimization.protocols import (
    Proposal,
    TransferRequest,
    TransferStrategy,
)
from airsenal.optimization.strategies.double import DoubleTransferStrategy
from airsenal.optimization.strategies.full_squad import FullSquadStrategy
from airsenal.optimization.strategies.none import NoTransfersStrategy
from airsenal.optimization.strategies.random_search import RandomTransferStrategy
from airsenal.optimization.strategies.single import SingleTransferStrategy

TRANSFER_STRATEGIES: dict[str, Callable[[], TransferStrategy]] = {
    "double": DoubleTransferStrategy,
    "full_squad": FullSquadStrategy,
    "none": NoTransfersStrategy,
    "random": RandomTransferStrategy,
    "single": SingleTransferStrategy,
}

# How many players change decides how the search is done: one or two can be
# enumerated exhaustively, more has to be sampled, and a chip that rebuilds the
# squad goes to a whole-squad optimizer.
_BY_TRANSFER_COUNT = {0: "none", 1: "single", 2: "double"}
_MANY_TRANSFERS = "random"
_REBUILD = "full_squad"


@dataclass(frozen=True)
class StrategySet:
    """
    Which strategy handles which move.

    Held as names rather than as built strategies so that the whole set survives
    a fork or a pickle - the transfer search hands it to worker processes, which
    build their own strategies from it.
    """

    by_transfer_count: Mapping[int, str] = field(
        default_factory=lambda: dict(_BY_TRANSFER_COUNT)
    )
    many_transfers: str = _MANY_TRANSFERS
    rebuild: str = _REBUILD

    def name_for(self, move: GameweekMove) -> str:
        """The name of the strategy that handles this move."""
        if move.rebuilds_squad:
            return self.rebuild
        return self.by_transfer_count.get(move.n_transfers, self.many_transfers)

    def create(self, move: GameweekMove) -> TransferStrategy:
        """The strategy that handles this move."""
        return lookup(TRANSFER_STRATEGIES, self.name_for(move), "transfer strategy")()


DEFAULT_STRATEGIES = StrategySet()


__all__ = [
    "DEFAULT_STRATEGIES",
    "TRANSFER_STRATEGIES",
    "Proposal",
    "StrategySet",
    "TransferRequest",
    "TransferStrategy",
]
