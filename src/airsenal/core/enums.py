"""
Positions and chips.

Both are currently bare strings compared literally in a dozen places, and the chip
names are additionally encoded into single letters for the transfer search. Making
them enums means a typo is an error rather than a silently unmatched branch.
"""

from enum import Enum


class Position(str, Enum):
    """A player's position."""

    GK = "GK"
    DEF = "DEF"
    MID = "MID"
    FWD = "FWD"

    # Python 3.11 changed how mixin enums format themselves, so without this an
    # f-string would write "Position.GK" into logs and database columns.
    __str__ = str.__str__

    @classmethod
    def back_to_front(cls) -> tuple["Position", ...]:
        return (cls.GK, cls.DEF, cls.MID, cls.FWD)

    @classmethod
    def front_to_back(cls) -> tuple["Position", ...]:
        return tuple(reversed(cls.back_to_front()))


class Chip(str, Enum):
    """An FPL chip."""

    WILDCARD = "wildcard"
    FREE_HIT = "free_hit"
    BENCH_BOOST = "bench_boost"
    TRIPLE_CAPTAIN = "triple_captain"

    __str__ = str.__str__

    @property
    def rebuilds_squad(self) -> bool:
        """Whether playing this chip replaces the squad rather than transferring."""
        return self in (Chip.WILDCARD, Chip.FREE_HIT)
