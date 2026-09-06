"""
Breaking a realised score into the components that predict it.

The ground truth a component's expected points are scored against. Every case
here checks the components sum to the points FPL actually awarded, which is the
guardrail: if AIrsenal's scoring rules ever drift from FPL's, this is where it
shows.
"""

import pytest

from airsenal.db.models import PlayerScore
from airsenal.game.enums import Position
from airsenal.prediction.point_components import RESIDUAL, actual_component_points


def performance(**kwargs) -> PlayerScore:
    """A performance with everything at zero except what a test sets."""
    fields = {
        "player_id": 1,
        "minutes": 90,
        "points": 0,
        "goals": 0,
        "assists": 0,
        "bonus": 0,
        "conceded": 0,
        "clean_sheets": 0,
        "saves": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "own_goals": 0,
        "penalties_saved": 0,
        "penalties_missed": 0,
        "defensive_contribution": None,
    }
    return PlayerScore(**{**fields, **kwargs})


def check_reconciles(score: PlayerScore, position: str) -> dict[str, float]:
    components = actual_component_points(score, position)
    assert sum(components.values()) == pytest.approx(float(score.points))
    return components


def test_a_blank_ninety_minutes_is_two_points():
    components = check_reconciles(performance(points=2, conceded=2), Position.MID)
    assert components["appearance"] == 2
    assert components["attacking"] == 0


def test_coming_on_for_ten_minutes_is_one_point():
    components = check_reconciles(
        performance(minutes=10, points=1, conceded=2), Position.MID
    )
    assert components["appearance"] == 1


def test_a_midfielder_who_scored_and_kept_a_clean_sheet():
    """2 appearance + 5 goal + 1 clean sheet + 3 bonus."""
    components = check_reconciles(
        performance(points=11, goals=1, clean_sheets=1, bonus=3), Position.MID
    )
    assert components["attacking"] == 5
    assert components["defending"] == 1
    assert components["bonus"] == 3


def test_a_defender_conceding_three_loses_a_point_per_two():
    """2 appearance - 1 for three conceded."""
    components = check_reconciles(performance(points=1, conceded=3), Position.DEF)
    assert components["defending"] == -1


def test_a_clean_sheet_needs_the_hour():
    """Under 60 minutes there is no clean sheet, whatever the result."""
    components = check_reconciles(
        performance(minutes=45, points=1, clean_sheets=1), Position.DEF
    )
    assert components["defending"] == 0


def test_a_keeper_gets_a_point_per_three_saves():
    """2 appearance + 4 clean sheet + 2 for seven saves."""
    components = check_reconciles(
        performance(points=8, clean_sheets=1, saves=7), Position.GK
    )
    assert components["saves"] == 2
    assert components["defending"] == 4


def test_cards_come_off():
    components = check_reconciles(
        performance(points=1, conceded=1, yellow_cards=1), Position.MID
    )
    assert components["cards"] == -1


def test_a_sending_off_costs_three():
    components = check_reconciles(
        performance(minutes=30, points=-2, conceded=1, red_cards=1), Position.FWD
    )
    assert components["cards"] == -3


@pytest.mark.parametrize(
    ("field", "total", "expected"),
    [("own_goals", 0, -2), ("penalties_saved", 7, 5), ("penalties_missed", 0, -2)],
)
def test_the_events_no_component_predicts_go_to_the_residual(field, total, expected):
    """
    Own goals and penalties are real points that nothing here models.

    About 0.5% of all points awarded, so they are reported rather than blamed on
    a component that did not earn them.
    """
    # two appearance points, nothing for conceding one, plus the event itself
    components = check_reconciles(
        performance(points=total, conceded=1, **{field: 1}), Position.GK
    )
    assert components[RESIDUAL] == expected


def test_a_defensive_contribution_counts_only_when_it_was_recorded():
    """Before 25/26 there is no such component, rather than one worth zero."""
    before = actual_component_points(performance(points=2, conceded=1), Position.DEF)
    assert "def_con" not in before

    cleared = check_reconciles(
        performance(points=4, conceded=1, defensive_contribution=12), Position.DEF
    )
    assert cleared["def_con"] == 2

    short = check_reconciles(
        performance(points=2, conceded=1, defensive_contribution=3), Position.DEF
    )
    assert short["def_con"] == 0


def test_a_position_with_no_scoring_rules_is_refused():
    """`MNG` was a real FPL position in 24/25, and nothing here scores it."""
    with pytest.raises(ValueError, match="not a position"):
        actual_component_points(performance(points=2), "MNG")
