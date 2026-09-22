"""
A team model that predicts only a mean, made usable by the points calculation.

These are the seam that lets a model with no distribution over goal counts - an
xG model fitted to a continuous quantity - be a `TEAM_MODELS` entry.
`PoissonScorelines` supplies the standard distribution and
`ConwayMaxwellScorelines` one whose spread is a measured number instead.
`tests/e2e/test_team_models.py` puts one through a real fit and prediction.
"""

import math
from typing import Any

import numpy as np
import pytest
from scipy.stats import poisson

from airsenal.game.scoring import MAX_GOALS
from airsenal.prediction.protocols import TeamFitData
from airsenal.prediction.team_models.scorelines import (
    DEFAULT_GOAL_DISPERSION,
    POISSON_DISPERSION,
    ConwayMaxwellScorelines,
    PoissonScorelines,
    conway_maxwell_pmf,
    outcome_proba_from_scores,
)

TEAMS = ["AAA", "BBB"]


class FlatExpectedGoals:
    """The smallest `ExpectedGoalsTeamModel`: one rate, whoever is playing."""

    def __init__(self, rate: float = 1.4, home_advantage: float = 0.3):
        self.rate = rate
        self.home_advantage = home_advantage
        self.teams: list[str] | None = None
        self.fitted_with: TeamFitData | None = None

    def fit(self, training_data: TeamFitData) -> "FlatExpectedGoals":
        self.fitted_with = training_data
        self.teams = sorted(TEAMS)
        return self

    def add_new_team(self, team_name: str, **kwargs: Any) -> None:
        del kwargs
        self.teams = sorted({*(self.teams or []), team_name})

    def predict_expected_goals(
        self, team: str, opponent: str, home: bool = True, **kwargs: Any
    ) -> float:
        del team, opponent, kwargs
        return self.rate + (self.home_advantage if home else 0.0)


@pytest.fixture
def wrapped():
    return PoissonScorelines(FlatExpectedGoals())


def test_the_distribution_sums_to_one(wrapped):
    """
    Truncating the support must not lose probability.

    `score_team_model` scores a freak result against the last bin, so that bin
    has to carry the whole tail rather than only its own count.
    """
    goals = np.arange(MAX_GOALS + 1)
    probabilities = wrapped.predict_score_n_proba(goals, TEAMS[0], TEAMS[1])
    assert probabilities.sum() == pytest.approx(1.0)
    assert all(0.0 <= float(p) <= 1.0 for p in probabilities)


def test_the_last_count_carries_the_tail(wrapped):
    """So a 10-0 is unlikely rather than impossible."""
    rate = wrapped.predict_expected_goals(TEAMS[0], TEAMS[1])
    last = wrapped.predict_score_n_proba(np.array([MAX_GOALS]), TEAMS[0], TEAMS[1])
    assert float(last[0]) > float(poisson.pmf(MAX_GOALS, rate))
    assert float(last[0]) == pytest.approx(float(poisson.sf(MAX_GOALS - 1, rate)))


def test_the_mean_survives_the_wrapping(wrapped):
    """The point of a Poisson: its mean is the rate it was given."""
    goals = np.arange(MAX_GOALS + 1)
    probabilities = wrapped.predict_score_n_proba(goals, TEAMS[0], TEAMS[1])
    expected = float((goals * probabilities).sum())
    assert expected == pytest.approx(
        wrapped.predict_expected_goals(TEAMS[0], TEAMS[1]), rel=1e-3
    )


def test_negative_goal_counts_have_no_probability(wrapped):
    probabilities = wrapped.predict_score_n_proba(
        np.array([-1, 0, 1]), TEAMS[0], TEAMS[1]
    )
    assert float(probabilities[0]) == 0.0
    assert float(probabilities[1]) > 0.0


def test_home_advantage_reaches_the_distribution(wrapped):
    """The wrapper must not flatten what the inner model says about the fixture."""
    at_home = wrapped.predict_score_n_proba(
        np.arange(MAX_GOALS + 1), TEAMS[0], TEAMS[1], home=True
    )
    away = wrapped.predict_score_n_proba(
        np.arange(MAX_GOALS + 1), TEAMS[0], TEAMS[1], home=False
    )
    goals = np.arange(MAX_GOALS + 1)
    assert float((goals * at_home).sum()) > float((goals * away).sum())


def test_fitting_and_new_teams_reach_the_wrapped_model(wrapped):
    """The wrapper is transparent for everything except what it adds."""
    training_data = {"home_team": np.array(TEAMS), "away_team": np.array(TEAMS[::-1])}
    assert wrapped.fit(training_data) is wrapped
    assert wrapped.model.fitted_with is training_data
    assert wrapped.teams == sorted(TEAMS)
    wrapped.add_new_team("CCC")
    assert "CCC" in wrapped.teams


def test_outcome_probabilities_are_derived_not_required(wrapped):
    """
    A mean-only model gets win/draw/loss for free once it has a distribution.

    Which is why `predict_outcome_proba` is on the protocol rather than fetched
    with getattr.
    """
    outcomes = wrapped.predict_outcome_proba([TEAMS[0]], [TEAMS[1]])
    total = outcomes["home_win"][0] + outcomes["draw"][0] + outcomes["away_win"][0]
    assert total == pytest.approx(1.0, abs=1e-6)
    # the home side is the one with the advantage
    assert outcomes["home_win"][0] > outcomes["away_win"][0]


def test_the_helper_and_the_method_agree(wrapped):
    direct = outcome_proba_from_scores(wrapped, [TEAMS[0]], [TEAMS[1]])
    through = wrapped.predict_outcome_proba([TEAMS[0]], [TEAMS[1]])
    for key in direct:
        assert math.isclose(float(direct[key][0]), float(through[key][0]))


class TestConwayMaxwellPmf:
    """
    The distribution itself: a Poisson whose spread is a parameter.

    Parameterised by the mean rather than the rate, so that a dispersion means
    the same thing wherever it is used and changing it does not silently move
    the number of goals the model predicts.
    """

    MEANS = np.array([0.4, 1.45, 2.9])

    @pytest.mark.parametrize("dispersion", [0.7, 1.0, 1.3, 2.0])
    def test_every_row_is_a_distribution(self, dispersion):
        pmf = conway_maxwell_pmf(self.MEANS, dispersion)
        assert pmf.shape == (len(self.MEANS), MAX_GOALS + 1)
        assert pmf.sum(axis=1) == pytest.approx(1.0)
        assert (pmf >= 0.0).all()

    @pytest.mark.parametrize("dispersion", [0.7, 1.0, 1.3, 2.0])
    def test_the_mean_is_the_mean_whatever_the_dispersion(self, dispersion):
        """What makes the dispersion a shape parameter and not a second rate."""
        counts = np.arange(MAX_GOALS + 1)
        pmf = conway_maxwell_pmf(self.MEANS, dispersion)
        assert pmf @ counts == pytest.approx(self.MEANS)

    def test_a_dispersion_of_one_is_a_poisson(self):
        """
        Where the two families overlap they agree.

        Up to the truncated support, which is where the two parameterisations
        differ: this one solves for the rate that puts the mean where asked, and
        a Poisson truncated at ten goals has a mean slightly below its rate.
        """
        counts = np.arange(MAX_GOALS + 1)
        got = conway_maxwell_pmf(self.MEANS, POISSON_DISPERSION)
        want = poisson.pmf(counts[None, :], self.MEANS[:, None])
        want = want / want.sum(axis=1, keepdims=True)
        assert got == pytest.approx(want, abs=1e-3)

    def test_a_higher_dispersion_is_narrower(self):
        """The whole reason for the family: measured spread, not assumed."""
        counts = np.arange(MAX_GOALS + 1)
        mean = 1.45
        variances = [
            float(conway_maxwell_pmf(mean, dispersion)[0] @ (counts - mean) ** 2)
            for dispersion in (0.7, 1.0, 1.4, 2.0)
        ]
        assert variances == sorted(variances, reverse=True)
        # and a dispersion of one is the Poisson's variance-equals-mean
        assert variances[1] == pytest.approx(mean, rel=1e-3)

    def test_a_scalar_mean_is_accepted(self):
        assert conway_maxwell_pmf(1.45, 1.1).shape == (1, MAX_GOALS + 1)

    def test_a_mean_of_zero_does_not_blow_up(self):
        """No side is predicted to create nothing, but nothing should raise."""
        pmf = conway_maxwell_pmf(0.0, 1.1)
        assert pmf.sum() == pytest.approx(1.0)
        assert float(pmf[0][0]) == pytest.approx(1.0)


class TestConwayMaxwellScorelines:
    """The wrapper, which is `PoissonScorelines` with the spread let loose."""

    @pytest.fixture
    def wrapped(self):
        return ConwayMaxwellScorelines(FlatExpectedGoals())

    def test_it_is_narrower_than_a_poisson_by_default(self, wrapped):
        """
        Because that is what Premier League goals measure as.

        Fewer clean sheets and fewer routs than a Poisson would give, which is
        what reaches the defensive point components.
        """
        assert wrapped.dispersion == DEFAULT_GOAL_DISPERSION
        assert wrapped.dispersion > POISSON_DISPERSION
        goals = np.arange(MAX_GOALS + 1)
        narrow = wrapped.predict_score_n_proba(goals, TEAMS[0], TEAMS[1])
        wide = PoissonScorelines(FlatExpectedGoals()).predict_score_n_proba(
            goals, TEAMS[0], TEAMS[1]
        )
        assert float(narrow[0]) < float(wide[0])
        assert float(narrow[1]) > float(wide[1])

    def test_the_distribution_sums_to_one(self, wrapped):
        probabilities = wrapped.predict_score_n_proba(
            np.arange(MAX_GOALS + 1), TEAMS[0], TEAMS[1]
        )
        assert probabilities.sum() == pytest.approx(1.0)

    def test_the_mean_survives_the_wrapping(self, wrapped):
        """A different spread must not move the number of goals predicted."""
        goals = np.arange(MAX_GOALS + 1)
        probabilities = wrapped.predict_score_n_proba(goals, TEAMS[0], TEAMS[1])
        assert float((goals * probabilities).sum()) == pytest.approx(
            wrapped.predict_expected_goals(TEAMS[0], TEAMS[1]), rel=1e-6
        )

    def test_negative_goal_counts_have_no_probability(self, wrapped):
        probabilities = wrapped.predict_score_n_proba(
            np.array([-1, 0, 1]), TEAMS[0], TEAMS[1]
        )
        assert float(probabilities[0]) == 0.0
        assert float(probabilities[1]) > 0.0

    def test_counts_above_the_support_read_the_last_bin(self, wrapped):
        """Which carries the tail, so a freak result is unlikely not impossible."""
        probabilities = wrapped.predict_score_n_proba(
            np.array([MAX_GOALS, MAX_GOALS + 3]), TEAMS[0], TEAMS[1]
        )
        assert float(probabilities[0]) > 0.0
        assert float(probabilities[0]) == float(probabilities[1])

    def test_a_dispersion_of_one_is_the_poisson_wrapper(self):
        """The families agree where they overlap, so the sweep is a real sweep."""
        poisson_like = ConwayMaxwellScorelines(
            FlatExpectedGoals(), dispersion=POISSON_DISPERSION
        )
        goals = np.arange(MAX_GOALS + 1)
        assert poisson_like.predict_score_n_proba(
            goals, TEAMS[0], TEAMS[1]
        ) == pytest.approx(
            PoissonScorelines(FlatExpectedGoals()).predict_score_n_proba(
                goals, TEAMS[0], TEAMS[1]
            ),
            abs=1e-3,
        )

    def test_everything_else_is_inherited(self, wrapped):
        """It adds a spread and changes nothing else about the seam."""
        assert isinstance(wrapped, PoissonScorelines)
        outcomes = wrapped.predict_outcome_proba([TEAMS[0]], [TEAMS[1]])
        total = outcomes["home_win"][0] + outcomes["draw"][0] + outcomes["away_win"][0]
        assert total == pytest.approx(1.0, abs=1e-6)
        assert outcomes["home_win"][0] > outcomes["away_win"][0]


def test_a_wrapper_names_the_model_inside_it(wrapped):
    """
    A record of a run has to say which model produced the mean.

    The wrapper's own class name says which distribution was put over the goal
    counts and nothing about which model was wrapped, and the wrapper is the
    part two runs being compared would usually share.
    """
    assert wrapped.describe_component() == "PoissonScorelines(FlatExpectedGoals)"
    assert (
        ConwayMaxwellScorelines(FlatExpectedGoals()).describe_component()
        == "ConwayMaxwellScorelines(FlatExpectedGoals)"
    )


def test_the_wrapped_model_is_reachable(wrapped):
    """
    `.model` is the interface, not an implementation detail.

    `tools/team_ratings.py` reads attack and defence ratings off it without
    knowing what class it wrapped, so a wrapper that renamed it would break a
    tool rather than fail to type-check.
    """
    assert isinstance(wrapped.model, FlatExpectedGoals)
