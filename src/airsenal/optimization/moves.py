"""
What we can do in a single gameweek, and which chips are available when.

The transfer search used to describe a gameweek's move as ``int | "W" | "F" |
"T0".."T2" | "B0".."B2"``, and every consumer re-parsed that encoding for itself -
six copies of the same "does it start with a T?" logic, none of which agreed on
what an unrecognised value should do. `GameweekMove` carries the two things the
encoding was hiding (how many transfers, and which chip) as fields, so the parsing
happens once, on the way in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from airsenal.core.enums import Chip
from airsenal.db.queries.gameweeks import next_gameweek

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

# Chips that are given a letter of their own because they replace the squad
# outright, so the number of transfers is not a meaningful part of the move.
_SQUAD_CHIP_LABELS = {Chip.WILDCARD: "W", Chip.FREE_HIT: "F"}
# Chips that are played alongside an ordinary number of transfers.
_TRANSFER_CHIP_LABELS = {Chip.TRIPLE_CAPTAIN: "T", Chip.BENCH_BOOST: "B"}

_LABEL_TO_SQUAD_CHIP = {v: k for k, v in _SQUAD_CHIP_LABELS.items()}
_LABEL_TO_TRANSFER_CHIP = {v: k for k, v in _TRANSFER_CHIP_LABELS.items()}

# A wildcard or free hit is scored as if the whole squad were transferred in.
SQUAD_SIZE = 15


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
    def n_players_in(self) -> int:
        """How many players come in - the whole squad for a wildcard or free hit."""
        return SQUAD_SIZE if self.rebuilds_squad else self.n_transfers

    @property
    def carry_forward(self) -> bool:
        """
        Whether the resulting squad is kept for the following gameweek.

        A free hit is reverted after the gameweek it is played in.
        """
        return self.chip is not Chip.FREE_HIT

    def label(self) -> str:
        """
        The short form used in strategy ids and in the suggestion table.

        This is a wire format: it is written into `TransferSuggestion` rows and
        displayed to users, so the letters cannot change.
        """
        if self.chip is None:
            return str(self.n_transfers)
        if self.chip in _SQUAD_CHIP_LABELS:
            return _SQUAD_CHIP_LABELS[self.chip]
        return f"{_TRANSFER_CHIP_LABELS[self.chip]}{self.n_transfers}"

    @classmethod
    def parse(cls, label: str | int) -> GameweekMove:
        """
        Read back a `label()`.

        Only needed for tests and for old suggestion rows - the search itself
        passes `GameweekMove` objects around and never round-trips through text.
        """
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
                f"week as we play {self.chip_to_play}"
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
    def from_weeks(
        cls, gameweeks: Iterable[int], chip_weeks: Mapping[str | Chip, int]
    ) -> ChipSchedule:
        """
        Build a schedule from the per-chip week numbers the CLI takes.

        `chip_weeks` maps a chip to -1 (never play it), 0 (consider it in any
        gameweek), or a gameweek number (definitely play it then).
        """
        gameweeks = list(gameweeks)
        allowed = tuple(
            Chip(chip) for chip, week in chip_weeks.items() if int(week) == 0
        )
        schedule = dict.fromkeys(gameweeks, GameweekChips(chips_allowed=allowed))

        for chip, week in chip_weeks.items():
            if int(week) <= 0 or int(week) not in gameweeks:
                continue
            existing = schedule[int(week)].chip_to_play
            if existing is not None:
                msg = f"Cannot play {existing} and {Chip(chip)} in the same week"
                raise ValueError(msg)
            # A definite chip displaces the optional ones for that gameweek.
            schedule[int(week)] = GameweekChips(chip_to_play=Chip(chip))

        return cls(schedule)

    def with_chip_to_play(self, gameweek: int, chip: Chip) -> ChipSchedule:
        """Return a copy that definitely plays `chip` in `gameweek`."""
        return ChipSchedule({**self.per_gameweek, gameweek: GameweekChips(chip)})


MAX_FREE_TRANSFERS = 5  # changed in 24/25 season (not accounted for in replay season)


POINTS_HIT_COST = 4  # points lost per transfer beyond the free ones


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
    """
    We get one extra free transfer per week, unless we use a wildcard or
    free hit, but we can't have more than max_free_transfers. So we should only
    be able to return 1 to max_free_transfers.
    """
    if move.rebuilds_squad:
        return prev_free_transfers  # changed in 24/25 season, previously 1
    return max(1, min(max_free_transfers, 1 + prev_free_transfers - move.n_transfers))


def next_week_transfers(
    free_transfers: int,
    hit_so_far: int,
    chips_played: Iterable[Chip | None] = (),
    max_total_hit: int | None = None,
    allow_unused_transfers: bool = True,
    max_opt_transfers: int = 2,
    chips: GameweekChips | None = None,
    max_free_transfers: int = MAX_FREE_TRANSFERS,
) -> list[tuple[GameweekMove, int, int, int]]:
    """Given where a strategy has got to and some optimisation constraints, determine
    the valid moves (transfers, and any chip played) for the following gameweek.

    free_transfers - free transfers available going into the gameweek
    hit_so_far - points hit taken by this strategy up to but not including this gameweek
    chips_played - the chips this strategy has already used, so they are not offered
    again

    max_opt_transfers - maximum number of transfers to play each week as part of
    strategy in optimisation

    max_free_transfers - maximum number of free transfers saved in the game rules
    (2 before 2024/25, 5 from 2024/25 season)

    Returns (move, new_ft_available, total_points_hit, hit_this_gw) tuples.
        - total_points_hit is the total points hit so far including this gw
        - hit_this_gw is the points hit incurred this gameweek
    """
    chips = chips if chips is not None else NO_CHIPS
    chips_played = list(chips_played)

    if not allow_unused_transfers and free_transfers == max_free_transfers:
        # Force at least 1 free transfer if a free transfer will be lost otherwise.
        # NOTE: This can cause the baseline strategy to be excluded. Re-add it outside
        # this function in that case.
        ft_choices = list(range(1, max_opt_transfers + 1))
    else:
        ft_choices = list(range(max_opt_transfers + 1))

    if max_total_hit is not None:
        ft_choices = [
            nt
            for nt in ft_choices
            if hit_so_far + calc_points_hit(GameweekMove(nt), free_transfers)
            <= max_total_hit
        ]

    # if we are definitely going to play a wildcard or free_hit deal with that first
    if chips.chip_to_play is not None and chips.chip_to_play.rebuilds_squad:
        moves = [GameweekMove(chip=chips.chip_to_play)]
    elif chips.chip_to_play is not None:
        # triple captain or bench boost - we can still do ft_choices transfers
        moves = [GameweekMove(nt, chips.chip_to_play) for nt in ft_choices]
    else:
        # no chip definitely played, but some might be allowed
        moves = [GameweekMove(nt) for nt in ft_choices]
        for chip in (Chip.WILDCARD, Chip.FREE_HIT):
            if chips.allows(chip, chips_played):
                moves.append(GameweekMove(chip=chip))
        for chip in (Chip.BENCH_BOOST, Chip.TRIPLE_CAPTAIN):
            if chips.allows(chip, chips_played):
                moves += [GameweekMove(nt, chip) for nt in ft_choices]

    hit_this_gw = [calc_points_hit(move, free_transfers) for move in moves]
    total_points_hit = [hit_so_far + hit for hit in hit_this_gw]
    new_ft_available = [
        calc_free_transfers(move, free_transfers, max_free_transfers) for move in moves
    ]

    return list(
        zip(moves, new_ft_available, total_points_hit, hit_this_gw, strict=True)
    )


def count_expected_outputs(
    gw_ahead: int,
    next_gw: int | None = None,
    free_transfers: int = 1,
    max_total_hit: int | None = None,
    allow_unused_transfers: bool = True,
    max_opt_transfers: int = 2,
    chip_schedule: ChipSchedule | None = None,
    max_free_transfers: int = MAX_FREE_TRANSFERS,
) -> tuple[int, bool]:
    """
    Count the number of possible transfer and chip strategies for gw_ahead gameweeks
    ahead, subject to:
    * Start with free_transfers free transfers.
    * Spend a max of max_total_hit points on transfers across whole period
    (None for no limit)
    * Allow playing the chips permitted by chip_schedule
    * Exclude strategies that waste free transfers (make 0 transfers if 2 free tramsfers
    are available), if allow_unused_transfers is False.
    * Make a maximum of max_opt_transfers transfers each gameweek.
    * Each chip only allowed once.

    Returns
    -------
        Tuple of int: number of strategies that will be computed, and bool: whether the
        baseline strategy will be excluded from the main optimization tree and will need
        to be computed separately (this can be the case if allow_unused_transfers is
        False). Either way, the total count of strategies will include the baseline.
    """
    next_gw = next_gameweek() if next_gw is None else next_gw
    chip_schedule = chip_schedule if chip_schedule is not None else ChipSchedule()

    # (free transfers, points hit so far, moves made) - the moves are all that is
    # needed to count branches and to spot the do-nothing baseline among them
    branches: list[tuple[int, int, tuple[GameweekMove, ...]]] = [
        (free_transfers, 0, ())
    ]

    for gw in range(next_gw, next_gw + gw_ahead):
        new_branches = []
        for ft, hit, moves in branches:
            possibilities = next_week_transfers(
                ft,
                hit,
                [move.chip for move in moves],
                max_total_hit=max_total_hit,
                max_opt_transfers=max_opt_transfers,
                allow_unused_transfers=allow_unused_transfers,
                chips=chip_schedule.for_gameweek(gw),
                max_free_transfers=max_free_transfers,
            )
            new_branches += [
                (new_ft, new_hit, (*moves, move))
                for move, new_ft, new_hit, _ in possibilities
            ]
        branches = new_branches

    # if allow_unused_transfers is False the baseline of no transfers can be removed
    # above. Check whether the 1st strategy is the baseline and if not add it back in.
    baseline_moves = (GameweekMove(),) * gw_ahead
    baseline_excluded = branches[0][2] != baseline_moves
    if baseline_excluded:
        branches.insert(0, (max_free_transfers, 0, baseline_moves))

    return len(branches), baseline_excluded
