"""
What a transfer search decided to do: a move per gameweek, and what it scores.

A `Plan` is the *result* of a search. The algorithms that produce one live in
`optimization/strategies/` (a gameweek at a time) and
`optimization/transfer_optimizers/` (a whole window); nothing here searches.

Plans are frozen, so a worker extending one cannot disturb the copy its siblings
were handed.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from airsenal.game.enums import Chip
from airsenal.optimization.moves import GameweekMove
from airsenal.optimization.squad_score import (
    get_discount_factor,
    get_discounted_squad_score,
)
from airsenal.squad.squad import Squad, SubWeights


@dataclass(frozen=True, slots=True)
class GameweekOutcome:
    """What one gameweek of a plan does, and what it is expected to score."""

    gameweek: int
    move: GameweekMove
    # discounted, and already net of any points hit
    points: float
    discount_factor: float
    points_hit: int
    free_transfers: int
    players_in: tuple[int, ...] = ()
    players_out: tuple[int, ...] = ()
    bank: int = 0

    @property
    def chip(self) -> Chip | None:
        return self.move.chip

    @property
    def undiscounted_points(self) -> float:
        """The score before the future-gameweek discount is applied."""
        if not self.discount_factor:
            return self.points
        return self.points / self.discount_factor

    def to_dict(self) -> dict[str, Any]:
        return {
            "gameweek": self.gameweek,
            "num_transfers": self.move.label(),
            "chip_played": str(self.chip) if self.chip else None,
            "points": self.points,
            "discount_factor": self.discount_factor,
            "points_hit": self.points_hit,
            "free_transfers": self.free_transfers,
            "players_in": list(self.players_in),
            "players_out": list(self.players_out),
            "bank": self.bank,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameweekOutcome":
        return cls(
            gameweek=int(data["gameweek"]),
            move=GameweekMove.parse(data["num_transfers"]),
            points=float(data["points"]),
            discount_factor=float(data["discount_factor"]),
            points_hit=int(data["points_hit"]),
            free_transfers=int(data["free_transfers"]),
            players_in=tuple(data["players_in"]),
            players_out=tuple(data["players_out"]),
            bank=int(data["bank"]),
        )


@dataclass(frozen=True, slots=True)
class Plan:
    """A sequence of gameweek moves and the score they are expected to produce."""

    root_gameweek: int
    outcomes: tuple[GameweekOutcome, ...] = field(default_factory=tuple)

    @property
    def total_score(self) -> float:
        return sum(outcome.points for outcome in self.outcomes)

    @property
    def total_points_hit(self) -> int:
        return sum(outcome.points_hit for outcome in self.outcomes)

    @property
    def gameweeks(self) -> tuple[int, ...]:
        return tuple(outcome.gameweek for outcome in self.outcomes)

    @property
    def chips_played(self) -> tuple[Chip | None, ...]:
        return tuple(outcome.chip for outcome in self.outcomes)

    def __len__(self) -> int:
        return len(self.outcomes)

    def outcome(self, gameweek: int) -> GameweekOutcome:
        """The outcome for one gameweek."""
        for outcome in self.outcomes:
            if outcome.gameweek == gameweek:
                return outcome
        msg = (
            f"Plan covers gameweeks {list(self.gameweeks)}, so it has nothing "
            f"for gameweek {gameweek}"
        )
        raise KeyError(msg)

    def extend(self, outcome: GameweekOutcome) -> "Plan":
        """A new plan with one more gameweek on the end."""
        return replace(self, outcomes=(*self.outcomes, outcome))

    @property
    def is_baseline(self) -> bool:
        """Whether this plan makes no transfers and plays no chips."""
        return all(outcome.move == GameweekMove() for outcome in self.outcomes)

    def label(self) -> str:
        """
        The per-gameweek moves joined with dashes, e.g. "0-1-W".

        A display and debugging aid; nothing identifies a plan by it.
        """
        return "-".join(outcome.move.label() for outcome in self.outcomes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_gameweek": self.root_gameweek,
            "total_score": self.total_score,
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Plan":
        return cls(
            root_gameweek=int(data["root_gameweek"]),
            outcomes=tuple(
                GameweekOutcome.from_dict(outcome) for outcome in data["outcomes"]
            ),
        )


def baseline_plan(
    squad: Squad,
    gameweeks: list[int],
    tag: str,
    root_gw: int | None = None,
    sub_weights: SubWeights | None = None,
) -> Plan:
    """
    The do-nothing plan, which every other plan is compared against.

    Scored with the same bench weighting as the plans it is compared against, or
    the comparison is between two different scoring functions.
    """
    root_gw = root_gw if root_gw is not None else gameweeks[0]
    outcomes = []
    for gw in gameweeks:
        discount_factor = get_discount_factor(root_gw, gw)
        outcomes.append(
            GameweekOutcome(
                gameweek=gw,
                move=GameweekMove(),
                points=get_discounted_squad_score(
                    squad, [gw], tag, root_gw=root_gw, sub_weights=sub_weights
                ),
                discount_factor=discount_factor,
                points_hit=0,
                free_transfers=0,
                bank=squad.budget,
            )
        )
    return Plan(root_gameweek=root_gw, outcomes=tuple(outcomes))


@dataclass(frozen=True)
class TransferSearchResult:
    """What an optimizer chose, and the do-nothing plan it is judged against."""

    best: Plan
    baseline: Plan | None = None
    # Every plan evaluated, for --save-plans. Empty for an optimizer that solves
    # rather than enumerates: the dump is a debugging aid, not a promise the
    # interface makes.
    considered: tuple[Plan, ...] = ()

    @property
    def baseline_score(self) -> float:
        """What doing nothing would have scored, or zero if it was never evaluated."""
        return self.baseline.total_score if self.baseline is not None else 0.0

    @classmethod
    def from_plans(cls, plans: Sequence[Plan]) -> "TransferSearchResult":
        """
        Read the answer off an exhaustive search.

        For an optimizer that evaluates every plan, the best and the baseline are
        both just entries in the list it produced.
        """
        if not plans:
            msg = "Failed to find a plan!"
            raise ValueError(msg)
        return cls(
            best=max(plans, key=lambda p: p.total_score),
            baseline=next((p for p in plans if p.is_baseline), None),
            considered=tuple(plans),
        )
