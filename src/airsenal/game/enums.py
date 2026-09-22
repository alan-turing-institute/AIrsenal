"""Positions and chips."""

from enum import StrEnum


class Position(StrEnum):
    """A position AIrsenal models a player in."""

    GK = "GK"
    DEF = "DEF"
    MID = "MID"
    FWD = "FWD"

    @classmethod
    def is_modelled(cls, position: str | None) -> bool:
        """
        Whether AIrsenal models a player in this position at all.

        Managers (position "MNG", from season 2425) are not: no model predicts
        them and no squad can contain one, so they are left out of fitting and
        skipped rather than predicted.
        """
        return position in cls

    @classmethod
    def modelled(cls) -> tuple[str, ...]:
        """Every position `is_modelled` accepts, for a `WHERE position IN (...)`."""
        return tuple(str(position) for position in cls)

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
