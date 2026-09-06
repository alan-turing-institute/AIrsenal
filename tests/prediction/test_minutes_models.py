"""
The minutes distribution, and the model that reproduces the old behaviour.

What used to be a list of recent appearances averaged inside the points
calculation is now a distribution that owns its weights. These check the
distribution's arithmetic and its invariants; `tests/e2e/test_minutes_models.py`
fits the models against a real database.
"""

import math
from itertools import pairwise

import pytest

from airsenal.prediction.evaluation import MINUTES_BANDS, minutes_band
from airsenal.prediction.minutes_models import (
    MINUTES_MODELS,
    RecentMinutesConfig,
    RecentMinutesModel,
    build_minutes_model,
)
from airsenal.prediction.protocols import MinutesDistribution


def test_a_uniform_distribution_weights_every_value_equally():
    distribution = MinutesDistribution.uniform([90.0, 45.0, 0.0])
    assert distribution.weights == (1 / 3, 1 / 3, 1 / 3)
    assert distribution.expected_minutes == pytest.approx(45.0)


def test_the_expectation_is_the_weighted_average():
    """The one thing the points calculation does with a distribution."""
    distribution = MinutesDistribution(minutes=(0.0, 90.0), weights=(0.25, 0.75))
    assert distribution.expectation(lambda minutes: minutes) == pytest.approx(67.5)
    # a quantity that is not the minutes themselves - two points for an hour
    assert distribution.expectation(
        lambda minutes: 2.0 if minutes >= 60 else 0.0
    ) == pytest.approx(1.5)


def test_a_uniform_expectation_is_the_old_average_over_recent_minutes():
    """
    The behaviour phase 1 had to preserve.

    The points calculation used to sum over recent appearances and divide by
    how many there were; a uniform distribution says the same thing.
    """
    recent = [90.0, 62.0, 13.0]
    distribution = MinutesDistribution.uniform(recent)

    def points(minutes: float) -> float:
        return 1.0 + (1.0 if minutes >= 60 else 0.0)

    assert distribution.expectation(points) == pytest.approx(
        sum(points(minutes) for minutes in recent) / len(recent)
    )


@pytest.mark.parametrize(
    ("minutes", "weights"),
    [
        ((), ()),
        ((90.0,), (0.5,)),
        ((90.0, 45.0), (1.0,)),
        ((-1.0,), (1.0,)),
        ((90.0, 45.0), (1.5, -0.5)),
    ],
)
def test_an_impossible_distribution_is_refused(minutes, weights):
    """
    A model that returns nonsense fails where it is returned, not later.

    An empty distribution used to be checked for in the points calculation; the
    invariant belongs to the type, so every model gets it.
    """
    with pytest.raises(ValueError, match=r"minutes|weight|one"):
        MinutesDistribution(minutes=minutes, weights=weights)


def test_probability_between_reads_the_weight_in_a_band():
    distribution = MinutesDistribution(
        minutes=(0.0, 30.0, 90.0), weights=(0.5, 0.2, 0.3)
    )
    assert distribution.probability_between(0.0, 1.0) == pytest.approx(0.5)
    assert distribution.probability_between(1.0, 60.0) == pytest.approx(0.2)
    assert distribution.probability_between(60.0, math.inf) == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("minutes", "band"), [(0.0, 0), (1.0, 1), (59.0, 1), (60.0, 2), (90.0, 2)]
)
def test_the_bands_are_the_ones_the_scoring_rules_use(minutes, band):
    """Nothing for not appearing, a point for appearing, two from the hour."""
    assert minutes_band(minutes) == band


def test_the_bands_cover_every_possible_number_of_minutes():
    assert MINUTES_BANDS[0][0] == 0.0
    assert MINUTES_BANDS[-1][1] == math.inf
    for (_, high), (low, _) in pairwise(MINUTES_BANDS):
        assert high == low


def test_the_table_builds_the_default_model():
    assert isinstance(build_minutes_model(), RecentMinutesModel)
    assert set(MINUTES_MODELS) == {"recent"}


def test_the_lookback_is_the_models_own_business():
    """
    How far back to look is configuration, not a caller's problem.

    `fixtures_behind` and `min_fixtures_behind` used to be arguments of the
    points calculation that no caller ever passed.
    """
    assert RecentMinutesConfig().min_fixtures_behind == 3
    assert (
        RecentMinutesModel(
            RecentMinutesConfig(min_fixtures_behind=8)
        ).config.min_fixtures_behind
        == 8
    )
