"""
The contracts the optimisation algorithms have to satisfy.

There are three of them, at three different scales, and the names are close
enough to be worth stating plainly:

- `SquadOptimizer` picks fifteen players from nothing, for one gameweek window.
- `TransferStrategy` picks one gameweek's move, starting from a squad.
- `TransferOptimizer` picks a move for every gameweek in a range.

Each declares only the method that does the work; see `progress_total` for the
optional method a component can add to size its own progress bar.

(Note the separate trap that predates all this: `optimization/strategy.py` holds
the *result* of a search, while `optimization/strategies/` holds the algorithms.)
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from sqlalchemy.orm.session import Session

from airsenal.core.enums import Chip
from airsenal.optimization.config import SquadScoringConfig
from airsenal.optimization.moves import (
    ChipSchedule,
    GameweekMove,
    TransferConstraints,
)
from airsenal.optimization.strategy import TransferSearchResult
from airsenal.squad.squad import Squad


class ProgressUpdater(Protocol):
    """
    Counts off one step of the optimizer's progress display.

    Called with no arguments for a finished strategy on the overall bar, or with
    a worker index for one candidate squad on that worker's own bar.
    """

    def __call__(self, index: int | None = ...) -> None: ...


# (worker index, strategy label, candidate squads to consider) - restarts one
# worker's bar for a new strategy. None candidates means the strategy could not
# say, and that worker's bar runs indeterminate.
type ProgressResetter = Callable[[int, str, int | None], None]

# Called once per candidate squad a strategy considers. The strategy counts what
# it does; how many there will be is what `Sized` reports, and turning the two
# into a percentage is the progress bar's job.
type StepCounter = Callable[[], None]


def progress_total(optimizer: object) -> int | None:
    """
    How many steps `optimizer` will take, if it is able to say.

    Sizing a progress bar is not part of doing the work, so it is not in the
    protocols below. Give a component a `num_increments()` method and its bar is
    exact - as every optimizer shipped here does; leave it out and the bar runs
    indeterminate rather than the component being unwritable.
    """
    num_increments = getattr(optimizer, "num_increments", None)
    return num_increments() if callable(num_increments) else None


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

    def propose(self, request: TransferRequest) -> TransferPlan: ...


def strategy_total(
    strategy: TransferStrategy, move: GameweekMove, num_iterations: int
) -> int | None:
    """
    How many candidate squads `strategy` will consider, if it is able to say.

    As `progress_total`, but a strategy's cost depends on the move and the
    iteration count, so its `num_increments` takes both. `propose` advances the
    bar once per candidate, so the two have to agree.
    """
    num_increments = getattr(strategy, "num_increments", None)
    return num_increments(move, num_iterations) if callable(num_increments) else None


# Called once per step of a whole-squad search, with the best score so far. What a
# step is depends on the optimizer - a generation, for the genetic algorithm.
type SquadProgress = Callable[[float], None]


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

    def optimize(self, request: SquadRequest) -> Squad:
        """
        The best squad this optimizer can find for the request.

        Returns the squad alone: neither caller uses a score, and requiring one in
        `get_discounted_squad_score` units is not something every kind of optimizer
        could honestly report.
        """
        ...


# Builds a squad optimizer sized from the transfer search's one effort knob. The
# wildcard and free-hit path has a single --num-iterations, while the standalone
# squad build has the full config and must not be scaled; making the difference a
# factory the caller supplies is what keeps the two apart.
type SquadOptimizerFactory = Callable[[int], SquadOptimizer]


@dataclass(frozen=True)
class TransferSearchRequest:
    """The problem a transfer optimizer is asked to solve."""

    starting_squad: Squad
    gameweeks: list[int]
    tag: str
    season: str
    chip_schedule: ChipSchedule
    num_free_transfers: int
    constraints: TransferConstraints = field(default_factory=TransferConstraints)

    @property
    def num_gameweeks(self) -> int:
        return len(self.gameweeks)


class TransferOptimizer(Protocol):
    """One way of choosing what to do across a range of gameweeks."""

    def search(self, request: TransferSearchRequest) -> TransferSearchResult:
        """
        The best plan this optimizer can find, and the baseline to judge it against.

        `search` rather than `propose` because a `TransferStrategy` proposes too,
        at the scale of a single gameweek, and the two are called within a few
        lines of each other inside the search.
        """
        ...
