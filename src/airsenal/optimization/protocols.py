"""
The contract a transfer strategy has to satisfy.

`make_best_transfers` used to be a single if/elif over the number of transfers,
with the arguments for each branch assembled inline and the progress-bar step
count maintained in a separate function that had to be kept in step with it.
Splitting the branches into strategies puts each one's search and its cost in
the same place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeAlias

from airsenal.core.enums import Chip

if TYPE_CHECKING:
    from collections.abc import Callable

    from airsenal.optimization.moves import GameweekMove
    from airsenal.squad.squad import Squad


class ProgressUpdater(Protocol):
    """
    Counts off one step of the optimizer's progress display.

    Called with no arguments for a finished strategy on the overall bar, or with
    a worker index for one candidate squad on that worker's own bar.
    """

    def __call__(self, index: int | None = ...) -> None: ...


# (worker index, strategy label, candidate squads to consider) - restarts one
# worker's bar for a new strategy
ProgressResetter: TypeAlias = "Callable[[int, str, int], None]"

# Called once per candidate squad a strategy considers. The strategy counts what
# it does; how many there will be is `num_increments`, and turning the two into a
# percentage is the progress bar's job.
StepCounter: TypeAlias = "Callable[[], None]"


@dataclass(frozen=True)
class TransferRequest:
    """Everything a strategy needs to pick this gameweek's transfers."""

    move: GameweekMove
    squad: Squad
    tag: str
    gameweeks: list[int]
    root_gw: int
    season: str
    num_iterations: int = 100
    # called once per candidate squad considered, if anything is watching
    progress: StepCounter | None = None

    @property
    def chip(self) -> Chip | None:
        return self.move.chip

    @property
    def transfer_gameweek(self) -> int:
        """The gameweek the transfers are made for."""
        return self.gameweeks[0]

    @property
    def bench_boost_gw(self) -> int | None:
        """The gameweek to score with a boosted bench, if any."""
        return self.transfer_gameweek if self.chip is Chip.BENCH_BOOST else None

    @property
    def triple_captain_gw(self) -> int | None:
        """The gameweek to score with a tripled captain, if any."""
        return self.transfer_gameweek if self.chip is Chip.TRIPLE_CAPTAIN else None

    def advance_progress(self) -> None:
        """Count off one of the candidate squads this strategy considers."""
        if self.progress is not None:
            self.progress()


@dataclass(frozen=True)
class TransferPlan:
    """What a strategy decided to do."""

    squad: Squad
    players_in: list[int]
    players_out: list[int]

    def as_transfer_dict(self) -> dict[str, list[int]]:
        """The {"in": [...], "out": [...]} shape the suggestion table stores."""
        return {"in": self.players_in, "out": self.players_out}


class TransferStrategy(Protocol):
    """One way of choosing a gameweek's transfers."""

    def num_increments(self, move: GameweekMove, num_iterations: int) -> int:
        """
        How many candidate squads this strategy will consider.

        This is the total of the worker's progress bar, and `propose` advances it
        by one per candidate, so the two have to agree.
        """
        ...

    def propose(self, request: TransferRequest) -> TransferPlan: ...
