"""
Tests for the gameweek window covered by an Absence.

gw_from is the first gameweek the player misses and gw_until the gameweek they
return for, so the absence covers gw_from <= gameweek < gw_until.
"""

import pytest
from sqlalchemy import select

from airsenal.conftest import session_scope
from airsenal.framework.schema import Absence


def make_absence(gw_from, gw_until):
    return Absence(
        player_id=1,
        season="1920",
        reason="injury",
        details="test",
        date_from="2019-09-01",
        date_until=None,
        gw_from=gw_from,
        gw_until=gw_until,
        url=None,
        timestamp="2019-09-01T00:00:00",
    )


@pytest.mark.parametrize(
    ("gameweek", "expected"),
    [
        (4, False),  # week before the absence starts
        (5, True),  # first gameweek missed - was previously excluded
        (6, True),
        (7, True),  # last gameweek missed
        (8, False),  # player returns in gw_until
        (9, False),
    ],
)
def test_covers_gameweek_window(gameweek, expected):
    assert make_absence(5, 8).covers_gameweek(gameweek) is expected


@pytest.mark.parametrize("gameweek", [5, 6, 38])
def test_covers_gameweek_open_ended(gameweek):
    """A null gw_until means the return date was unknown, so the player is absent
    from gw_from until the end of the season."""
    assert make_absence(5, None).covers_gameweek(gameweek) is True


@pytest.mark.parametrize("gameweek", [1, 4])
def test_covers_gameweek_open_ended_before_start(gameweek):
    assert make_absence(5, None).covers_gameweek(gameweek) is False


def test_single_gameweek_absence():
    """An absence where the player returns the following week covers one gameweek."""
    absence = make_absence(5, 6)
    assert absence.covers_gameweek(5) is True
    assert absence.covers_gameweek(6) is False


def test_covers_gameweek_clause_matches_python():
    """The SQL clause and the Python predicate must agree."""
    absences = [make_absence(5, 8), make_absence(10, None)]
    with session_scope() as ts:
        for absence in absences:
            ts.add(absence)
        ts.flush()

        matched_any = False
        for gameweek in range(1, 15):
            from_sql = {
                a.id
                for a in ts.scalars(
                    select(Absence).where(
                        Absence.season == "1920",
                        Absence.covers_gameweek_clause(gameweek),
                    )
                ).all()
            }
            from_python = {a.id for a in absences if a.covers_gameweek(gameweek)}
            assert from_sql == from_python, f"mismatch in gameweek {gameweek}"
            matched_any = matched_any or bool(from_sql)

        # guard against the comparison passing because both sides are always empty
        assert matched_any

        ts.rollback()
