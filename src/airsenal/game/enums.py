"""
Positions and chips.

Both subclass str, so a value read off a database row compares and indexes as
the enum. Nothing therefore fails when a bare literal is written instead;
`tests/test_naming_conventions.py` is what catches that. This module and
`mappings.py` are the boundary and keep their literals.
"""

from enum import StrEnum


class Position(StrEnum):
    """A player's position."""

    GK = "GK"
    DEF = "DEF"
    MID = "MID"
    FWD = "FWD"

    @classmethod
    def back_to_front(cls) -> tuple["Position", ...]:
        return (cls.GK, cls.DEF, cls.MID, cls.FWD)

    @classmethod
    def front_to_back(cls) -> tuple["Position", ...]:
        return tuple(reversed(cls.back_to_front()))


class Chip(StrEnum):
    """An FPL chip."""

    WILDCARD = "wildcard"
    FREE_HIT = "free_hit"
    BENCH_BOOST = "bench_boost"
    TRIPLE_CAPTAIN = "triple_captain"

    @property
    def rebuilds_squad(self) -> bool:
        """Whether playing this chip replaces the squad rather than transferring."""
        return self in (Chip.WILDCARD, Chip.FREE_HIT)
