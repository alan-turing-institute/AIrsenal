"""
The result of searching one branch of the transfer tree.

Strategies used to be untyped dicts written to `strategy_{tag}_{sid}.json` in a
temporary directory, and the parent process found the best one by listing that
directory and comparing filenames. Everything about that was load-bearing and
none of it was checked: the gameweek was a dict key, built as an int and read
back as `str(gw)` after the JSON round trip, so an int lookup silently missed.

Here a gameweek is a field of a list element, which makes that whole class of
mistake unrepresentable. Strategies are frozen, so a worker extending one
cannot disturb the copy its siblings were handed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from airsenal.core.enums import Chip
from airsenal.optimization.moves import GameweekMove


@dataclass(frozen=True, slots=True)
class GameweekOutcome:
    """What one gameweek of a strategy does, and what it is expected to score."""

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
    def from_dict(cls, data: dict[str, Any]) -> GameweekOutcome:
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
class Strategy:
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
            f"Strategy covers gameweeks {list(self.gameweeks)}, so it has nothing "
            f"for gameweek {gameweek}"
        )
        raise KeyError(msg)

    def extend(self, outcome: GameweekOutcome) -> Strategy:
        """A new strategy with one more gameweek on the end."""
        return replace(self, outcomes=(*self.outcomes, outcome))

    def label(self) -> str:
        """
        The per-gameweek moves joined with dashes, e.g. "0-1-W".

        This was the filename the search used to identify a strategy by; it is
        now only a display and debugging aid.
        """
        return "-".join(outcome.move.label() for outcome in self.outcomes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_gameweek": self.root_gameweek,
            "total_score": self.total_score,
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Strategy:
        return cls(
            root_gameweek=int(data["root_gameweek"]),
            outcomes=tuple(
                GameweekOutcome.from_dict(outcome) for outcome in data["outcomes"]
            ),
        )
