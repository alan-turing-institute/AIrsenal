"""
The contracts the optimisation algorithms have to satisfy.

There are three of them, at three different scales, and the names are close
enough to be worth stating plainly:

- `SquadOptimizer` picks fifteen players from nothing, for one gameweek window.
- `TransferStrategy` picks one gameweek's move, starting from a squad.
- `TransferOptimizer` picks a move for every gameweek in a range.

`make_best_transfers` used to be a single if/elif over the number of transfers,
with the arguments for each branch assembled inline and the progress-bar step
count maintained in a separate function that had to be kept in step with it.
Splitting the branches into strategies puts each one's search and its cost in
the same place.

(Note the separate trap that predates all this: `optimization/strategy.py` holds
the *result* of a search, while `optimization/strategies/` holds the algorithms.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypeAlias

from airsenal.core.enums import Chip
from airsenal.optimization.config import SquadScoringConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.orm.session import Session

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


# Called once per step of a whole-squad search, with the best score so far. What a
# step is depends on the optimizer - a generation, for the genetic algorithm - and
# there are `num_increments()` of them.
SquadProgress: TypeAlias = "Callable[[float], None]"


@dataclass(frozen=True)
class SquadRequest:
    """Everything an optimizer needs to pick a whole squad."""

    gameweeks: list[int]
    tag: str
    season: str
    scoring: SquadScoringConfig = field(default_factory=SquadScoringConfig)
    bench_boost_gw: int | None = None
    triple_captain_gw: int | None = None
    remove_zero: bool = True
    progress: SquadProgress | None = None
    # Only ever set by a caller in the parent process. A Session cannot cross a
    # process boundary, so a request built inside a search worker leaves this None
    # and a request must never be put on a queue - see tests/test_pickling.py.
    dbsession: Session | None = None

    @property
    def budget(self) -> int:
        return self.scoring.budget

    def advance_progress(self, best_score: float) -> None:
        """Report one step of the search, if anything is watching."""
        if self.progress is not None:
            self.progress(best_score)


class SquadOptimizer(Protocol):
    """One way of picking a whole squad from scratch."""

    def num_increments(self) -> int:
        """
        How many times a request's `progress` will be called.

        Takes no request because neither caller has one yet: `fill_initial_squad`
        creates its progress bar around the search, and `FullSquadStrategy` is
        asked for its cost by `TransferStrategy.num_increments`, which is given
        only a move. The count is a property of the optimizer's settings rather
        than of the problem.
        """
        ...

    def scaled(self, num_iterations: int) -> SquadOptimizer:
        """
        A copy of this optimizer sized from the transfer search's one effort knob.

        The wildcard and free-hit path has a single --num-iterations, while the
        standalone squad build has the full config and must not be scaled. Making
        that an explicit transformation the caller applies is what keeps the two
        apart - and what stops `FullSquadStrategy` having to know that its
        optimizer is a genetic algorithm in order to resize it.
        """
        ...

    def optimize(self, request: SquadRequest) -> Squad:
        """
        The best squad this optimizer can find for the request.

        Returns the squad alone: neither caller uses a score, and requiring one in
        `get_discounted_squad_score` units is not something every kind of optimizer
        could honestly report.
        """
        ...
