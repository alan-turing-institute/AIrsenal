"""
Transfer strategies, one per module, each registering itself by name.

`StrategySet` is the single place that decides which one a move needs. That
decision used to be an if/elif inside `make_best_transfers`, duplicated in the
progress-bar sizing, and the two could disagree. It then became a module-level
constant, which no caller could influence - which is why `--set-ga` could tune
the genetic algorithm behind `optimize squad` but not the identical one behind a
wildcard or free hit.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

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
# squad goes to a whole-squad optimizer.
_BY_TRANSFER_COUNT = {0: "none", 1: "single", 2: "double"}
_MANY_TRANSFERS = "random"
_REBUILD = "full_squad"


@dataclass(frozen=True)
class StrategySet:
    """
    Which strategy handles which move, and how each one is configured.

    Held as names and `key=value` strings rather than as built strategies so that
    the whole set survives a fork or a pickle - the transfer search hands it to
    worker processes, which build their own strategies from it. It also means a
    strategy's options reach it through the same `config_from_overrides` path as
    every other registry option, rather than a second mechanism.
    """

    by_transfer_count: Mapping[int, str] = field(
        default_factory=lambda: dict(_BY_TRANSFER_COUNT)
    )
    many_transfers: str = _MANY_TRANSFERS
    rebuild: str = _REBUILD
    options: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    def name_for(self, move: GameweekMove) -> str:
        """The name of the strategy that handles this move."""
        if move.rebuilds_squad:
            return self.rebuild
        return self.by_transfer_count.get(move.n_transfers, self.many_transfers)

    def create(self, move: GameweekMove) -> TransferStrategy:
        """The strategy that handles this move, with this set's settings."""
        name = self.name_for(move)
        # create_with(name, {}) is exactly create(name) - it builds the config
        # class with no overrides - so there is one path, not two.
        return TRANSFER_STRATEGIES.create_with(name, self.options.get(name, {}))


DEFAULT_STRATEGIES = StrategySet()


def strategy_name_for(move: GameweekMove) -> str:
    """The name of the strategy the default set uses for this move."""
    return DEFAULT_STRATEGIES.name_for(move)


def select_strategy(move: GameweekMove) -> TransferStrategy:
    """The strategy the default set uses for this move, with its default settings."""
    return DEFAULT_STRATEGIES.create(move)


__all__ = [
    "DEFAULT_STRATEGIES",
    "TRANSFER_STRATEGIES",
    "NoOptions",
    "StrategySet",
    "TransferPlan",
    "TransferRequest",
    "TransferStrategy",
    "select_strategy",
    "strategy_name_for",
]
