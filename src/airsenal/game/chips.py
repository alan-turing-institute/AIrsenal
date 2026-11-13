"""
How many of each chip a season gives, and when each one can be played.

A season's chips come in two halves, split after `FIRST_HALF_LAST_GAMEWEEK`: a
chip from the first half has to be played by then, and the second half gives a
new one. Every season has had two wildcards split that way. From
`EVERY_CHIP_TWICE_FROM` every other chip is split the same way; before it, each
of them came once, to be played at any point in the season.
"""

from collections.abc import Iterable

from airsenal.game.enums import Chip
from airsenal.game.season import season_str_to_year

FIRST_HALF_LAST_GAMEWEEK = 19
EVERY_CHIP_TWICE_FROM = "2526"


def comes_twice(chip: Chip, season: str) -> bool:
    """Whether the season gives one of `chip` for each half."""
    return chip is Chip.WILDCARD or season_str_to_year(season) >= season_str_to_year(
        EVERY_CHIP_TWICE_FROM
    )


def season_half(gameweek: int) -> int:
    """1 for the first half of the season, 2 for the second."""
    return 1 if gameweek <= FIRST_HALF_LAST_GAMEWEEK else 2


def chips_used_up(
    chips_played: Iterable[tuple[int, Chip | None]],
    gameweek: int,
    season: str,
) -> tuple[Chip, ...]:
    """
    The chips `chips_played` leaves nothing of to play in `gameweek`.

    Args:
        chips_played: (gameweek, chip) for each gameweek played so far, with None
            for a gameweek that played none.
    """
    return tuple(
        chip
        for played_in, chip in chips_played
        if chip is not None
        and (
            not comes_twice(chip, season)
            or season_half(played_in) == season_half(gameweek)
        )
    )
