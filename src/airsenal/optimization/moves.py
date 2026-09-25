"""
What we can do in a single gameweek, and which chips are available when.

`GameweekMove` carries how many transfers and which chip as fields. Moves are
also written in a compact string form (``int | "W" | "F" | "T0".."T2" |
"B0".."B2"``) for display and in plan and replay JSON.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace

from airsenal.game.enums import Chip
from airsenal.game.scoring import (
    MAX_FREE_TRANSFERS,
    POINTS_HIT_COST,
    free_transfers_after,
)

# Chips that are given a letter of their own because they replace the squad
# outright, so the number of transfers is not a meaningful part of the move.
_SQUAD_CHIP_LABELS = {Chip.WILDCARD: "W", Chip.FREE_HIT: "F"}
# Chips that are played alongside an ordinary number of transfers.
_TRANSFER_CHIP_LABELS = {Chip.TRIPLE_CAPTAIN: "T", Chip.BENCH_BOOST: "B"}

_LABEL_TO_SQUAD_CHIP = {v: k for k, v in _SQUAD_CHIP_LABELS.items()}
_LABEL_TO_TRANSFER_CHIP = {v: k for k, v in _TRANSFER_CHIP_LABELS.items()}


@dataclass(frozen=True, slots=True)
class GameweekMove:
    """The transfers and chip played in a single gameweek."""

    n_transfers: int = 0
    chip: Chip | None = None

    def __post_init__(self) -> None:
        if self.n_transfers < 0:
            msg = f"n_transfers must not be negative, got {self.n_transfers}"
            raise ValueError(msg)
        if self.rebuilds_squad and self.n_transfers:
            msg = (
                f"{self.chip} replaces the whole squad, so n_transfers is not "
                f"meaningful (got {self.n_transfers})"
            )
            raise ValueError(msg)

    @property
    def rebuilds_squad(self) -> bool:
        """Whether this move replaces the squad rather than transferring players."""
        return self.chip is not None and self.chip.rebuilds_squad

    @property
    def carry_forward(self) -> bool:
        """
        Whether the resulting squad is kept for the following gameweek.

        A free hit is reverted after the gameweek it is played in.
        """
        return self.chip is not Chip.FREE_HIT

    def label(self) -> str:
        """The short form used in plan labels, tables and JSON output."""
        if self.chip is None:
            return str(self.n_transfers)
        if self.chip in _SQUAD_CHIP_LABELS:
            return _SQUAD_CHIP_LABELS[self.chip]
        return f"{_TRANSFER_CHIP_LABELS[self.chip]}{self.n_transfers}"

    @classmethod
    def parse(cls, label: str | int) -> "GameweekMove":
        """Read back a `label()`."""
        if isinstance(label, int):
            return cls(label)
        if label in _LABEL_TO_SQUAD_CHIP:
            return cls(chip=_LABEL_TO_SQUAD_CHIP[label])
        if len(label) > 1 and label[0] in _LABEL_TO_TRANSFER_CHIP:
            return cls(int(label[1:]), _LABEL_TO_TRANSFER_CHIP[label[0]])
        try:
            return cls(int(label))
        except ValueError:
            msg = f"Unrecognised move label: {label!r}"
            raise ValueError(msg) from None

    def __str__(self) -> str:
        return self.label()


@dataclass(frozen=True, slots=True)
class GameweekChips:
    """Which chips may or must be played in one gameweek."""

    chip_to_play: Chip | None = None
    chips_allowed: tuple[Chip, ...] = ()

    def __post_init__(self) -> None:
        if self.chip_to_play is not None and self.chips_allowed:
            msg = (
                f"Cannot allow {[str(c) for c in self.chips_allowed]} in the same "
                f"gameweek as we play {self.chip_to_play}"
            )
            raise ValueError(msg)

    def allows(self, chip: Chip, already_played: Iterable[Chip | None]) -> bool:
        """Whether `chip` may optionally be played, given the chips used so far."""
        return chip in self.chips_allowed and chip not in already_played


NO_CHIPS = GameweekChips()


@dataclass(frozen=True, slots=True)
class ChipSchedule:
    """When each chip may or must be played, over a range of gameweeks."""

    per_gameweek: Mapping[int, GameweekChips] = field(default_factory=dict)

    def for_gameweek(self, gameweek: int) -> GameweekChips:
        return self.per_gameweek.get(gameweek, NO_CHIPS)

    @classmethod
    def from_gameweeks(
        cls,
        gameweeks: Iterable[int],
        chip_gameweeks: "ChipGameweeks | Mapping[str | Chip, int]",
    ) -> "ChipSchedule":
        """
        Build a schedule from the per-chip gameweek numbers the CLI takes.

        `chip_gameweeks` maps a chip to -1 (never play it), 0 (consider it in any
        gameweek), or a gameweek number (definitely play it then).
        """
        gameweeks = list(gameweeks)
        pairs = chip_gameweeks.items()
        allowed = tuple(Chip(chip) for chip, gameweek in pairs if int(gameweek) == 0)
        schedule = dict.fromkeys(gameweeks, GameweekChips(chips_allowed=allowed))

        for chip, gameweek in pairs:
            if int(gameweek) <= 0 or int(gameweek) not in gameweeks:
                continue
            existing = schedule[int(gameweek)].chip_to_play
            if existing is not None:
                msg = f"Cannot play {existing} and {Chip(chip)} in the same gameweek"
                raise ValueError(msg)
            # A definite chip displaces the optional ones for that gameweek.
            schedule[int(gameweek)] = GameweekChips(chip_to_play=Chip(chip))

        return cls(schedule)


def calc_points_hit(
    move: GameweekMove, free_transfers: int, cost: int = POINTS_HIT_COST
) -> int:
    """
    Points lost for making more transfers than we have free.

    Wildcard and free hit rebuild the squad without a hit; the other two chips
    are played alongside ordinary transfers and are charged as usual.
    """
    if move.rebuilds_squad:
        return 0
    return max(0, cost * (move.n_transfers - free_transfers))


def calc_free_transfers(
    move: GameweekMove,
    prev_free_transfers: int,
    max_free_transfers: int = MAX_FREE_TRANSFERS,
) -> int:
    """How many free transfers are available the gameweek after `move`."""
    return free_transfers_after(
        move.n_transfers,
        prev_free_transfers,
        max_free_transfers,
        rebuilds_squad=move.rebuilds_squad,
    )


@dataclass(frozen=True)
class ChipGameweeks:
    """
    Which gameweek to play each chip in, as the CLI takes it.

    -1 never, 0 any gameweek the search likes, n that gameweek. `played` is
    (gameweek, chip) for each chip already spent, which a replay accumulates as
    it goes; a live run leaves it empty and says -1 for a spent chip instead.
    """

    wildcard: int = -1
    free_hit: int = -1
    triple_captain: int = -1
    bench_boost: int = -1
    played: tuple[tuple[int, Chip], ...] = ()

    def after_playing(self, chip: Chip, gameweek: int) -> "ChipGameweeks":
        """The same chip gameweeks, with `chip` spent in `gameweek`."""
        return replace(self, played=(*self.played, (gameweek, chip)))

    def items(self) -> list[tuple[Chip, int]]:
        """Each chip with the gameweek it is wanted in."""
        return [
            (Chip.WILDCARD, self.wildcard),
            (Chip.FREE_HIT, self.free_hit),
            (Chip.TRIPLE_CAPTAIN, self.triple_captain),
            (Chip.BENCH_BOOST, self.bench_boost),
        ]

    def chip_in(self, gameweek: int) -> Chip | None:
        """The chip pinned to this gameweek, if any."""
        return next(
            (chip for chip, chip_gameweek in self.items() if chip_gameweek == gameweek),
            None,
        )
