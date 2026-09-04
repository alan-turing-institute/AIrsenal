"""
The contracts the optimisation algorithms have to satisfy.

There are three of them, at three different scales, and the names are close
enough to be worth stating plainly:

- `SquadOptimizer` picks fifteen players from nothing, for one gameweek window.
- `TransferStrategy` picks one gameweek's move, starting from a squad.
- `TransferOptimizer` picks a move for every gameweek in a range.

Each declares only the method that does the work; see `progress_total` for the
optional method a component can add to size its own progress bar.

What they produce lives in `optimization/plan.py`: a `Proposal` for one gameweek,
a `Plan` for a whole window.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from sqlalchemy.orm.session import Session

from airsenal.game.enums import Chip
from airsenal.game.scoring import MAX_FREE_TRANSFERS
from airsenal.optimization.moves import ChipSchedule, GameweekMove
from airsenal.optimization.plan import TransferSearchResult
from airsenal.optimization.squad_score import SquadScoringConfig
from airsenal.squad.squad import Squad

# How many candidate squads to consider when a move rebuilds the whole squad,
# how many transfers a strategy may make in one gameweek, and how many points a
# plan may spend on them, when nothing says otherwise. Beside the settings they
# are the defaults for: `TransferRequest` and `TransferConstraints` below, and
# the tree search's own signatures.
DEFAULT_NUM_ITERATIONS = 100
DEFAULT_MAX_OPT_TRANSFERS = 2
DEFAULT_MAX_TOTAL_HIT = 8


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


def progress_total(optimizer: object, effort: int | None = None) -> int | None:
    """
    How many steps `optimizer` will take, if it is able to say.

    Sizing a progress bar is not part of doing the work, so it is not in the
    protocols below. Give a component a `num_increments()` method and its bar is
    exact - as every optimizer shipped here does; leave it out and the bar runs
    indeterminate rather than the component being unwritable.

    An optimizer sized by an effort budget takes it as `num_increments`' one
    optional argument, and cannot answer without it.
    """
    num_increments = getattr(optimizer, "num_increments", None)
    if not callable(num_increments):
        return None
    total = num_increments(effort) if effort is not None else num_increments()
    return int(total) if total is not None else None


@dataclass(frozen=True)
class TransferRequest:
    """Everything a strategy needs to pick this gameweek's transfers."""

    move: GameweekMove
    squad: Squad
    tag: str
    gameweeks: list[int]
    root_gw: int
    season: str
    num_iterations: int = DEFAULT_NUM_ITERATIONS
    # How a squad is scored, so that a strategy weighs the bench the same way the
    # squad builder does. Must be set on every transfer path, or a flag like
    # --no-subs reaches one optimizer and not the other.
    scoring: SquadScoringConfig = field(default_factory=SquadScoringConfig)
    # Set only for a move that rebuilds the whole squad, which is the one kind of
    # move a strategy cannot answer by enumerating swaps. None means the default
    # whole-squad optimizer.
    squad_optimizer: "SquadOptimizer | None" = None
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
class Proposal:
    """What a strategy decided to do for one gameweek."""

    squad: Squad
    players_in: list[int]
    players_out: list[int]

    def as_transfer_dict(self) -> dict[str, list[int]]:
        """The {"in": [...], "out": [...]} shape the suggestion table stores."""
        return {"in": self.players_in, "out": self.players_out}


class TransferStrategy(Protocol):
    """One way of choosing a gameweek's transfers."""

    def propose(self, request: TransferRequest) -> Proposal: ...


def strategy_total(strategy: TransferStrategy, request: TransferRequest) -> int | None:
    """
    How many candidate squads `strategy` will consider, if it is able to say.

    As `progress_total`, but a strategy's cost depends on what it is being asked,
    so its `num_increments` takes the same request `propose` does. `propose`
    advances the bar once per candidate, so the two have to agree - and asking
    the same object both questions is what keeps them in step.
    """
    num_increments = getattr(strategy, "num_increments", None)
    if not callable(num_increments):
        return None
    total = num_increments(request)
    return int(total) if total is not None else None


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
    # How hard to search, in whatever unit the optimizer sizes itself by. None
    # means "use your own configuration", which is what a standalone squad build
    # wants; the wildcard and free-hit rebuilds inside a transfer search have one
    # --num-iterations knob and pass it here.
    effort: int | None = None
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


@dataclass(frozen=True, slots=True)
class TransferConstraints:
    """
    What a transfer search is allowed to consider.

    A field of `TransferSearchRequest`, and here beside it: exactly the knobs
    the tree search's branch enumeration takes. One frozen object rather than
    loose arguments, so nothing can be dropped on the way to a worker process.
    """

    # None is no cap at all, which a search has to be asked for explicitly.
    max_total_hit: int | None = DEFAULT_MAX_TOTAL_HIT
    allow_unused_transfers: bool = False
    max_opt_transfers: int = DEFAULT_MAX_OPT_TRANSFERS
    max_free_transfers: int = MAX_FREE_TRANSFERS


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
    # How a squad is scored, handed on to every strategy the search runs.
    scoring: SquadScoringConfig = field(default_factory=SquadScoringConfig)
    # A chip that rebuilds the squad delegates to this, so any optimizer that
    # handles wildcards and free hits needs one. None means the default whole-squad
    # optimizer; `FullSquadStrategy` is the single place that resolves it.
    squad_optimizer: "SquadOptimizer | None" = None

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
