"""What FPL awards, for the events that do not need a model to predict."""

from airsenal.game.scoring import get_appearance_points


def test_appearance_points():
    """Points for appearances alone."""
    assert get_appearance_points(0) == 0
    assert get_appearance_points(45) == 1
    assert get_appearance_points(60) == 2
    assert get_appearance_points(90) == 2
