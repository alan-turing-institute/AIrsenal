"""
Turning what a team model says about goals into what the points calculation needs.

Two directions. `outcome_proba_from_scores` goes from per-team goal
distributions to win/draw/loss, for a model that treats the two teams'
scorelines as independent - bpl does this itself, so `DixonColesTeamModel` does
not come through here. `PoissonScorelines` goes the other way, from a model
that predicts only an expected number of goals to one that has a distribution
over goal counts, and `ConwayMaxwellScorelines` does the same with a
distribution whose spread is measured rather than assumed.
"""

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from scipy.special import gammaln
from scipy.stats import poisson

from airsenal.game.scoring import MAX_GOALS
from airsenal.prediction.protocols import (
    ExpectedGoalsTeamModel,
    ScorelineTeamModel,
    TeamFitData,
)


def outcome_proba_from_scores(
    model: ScorelineTeamModel,
    home_team: Sequence[str],
    away_team: Sequence[str],
    max_goals: int = MAX_GOALS,
) -> dict[str, np.ndarray]:
    """Win/draw/loss probabilities per fixture, from independent goal counts."""
    goals = np.arange(max_goals + 1)
    home_win, draw, away_win = [], [], []
    for home, away in zip(home_team, away_team, strict=True):
        # outer[h, a] = P(home scores h) * P(away scores a)
        outer = np.outer(
            model.predict_score_n_proba(goals, home, away, home=True),
            model.predict_score_n_proba(goals, away, home, home=False),
        )
        home_win.append(float(np.tril(outer, -1).sum()))
        draw.append(float(np.trace(outer)))
        away_win.append(float(np.triu(outer, 1).sum()))
    return {
        "home_win": np.array(home_win),
        "draw": np.array(draw),
        "away_win": np.array(away_win),
    }


class PoissonScorelines:
    """
    A team model that predicts only a mean, read as a Poisson over goal counts.

    An expected-goals model has no natural distribution over *counts* - it is
    fitted to a continuous quantity - so this supplies the standard one rather
    than leaving every such model to invent it. Wrap the model in its
    `TEAM_MODELS` entry, so the table still promises a `ScorelineTeamModel` and
    nothing downstream has to ask which kind it was given.

    The support is truncated at `max_goals`, with the whole tail above it piled
    onto the last count, so the probabilities still sum to one - a freak result
    is unlikely rather than impossible.

    `model` is the wrapped model, and is public because a wrapper is otherwise
    a dead end: `tools/team_ratings.py` reads the attack and defence ratings off
    it, and `describe_component` names it.
    """

    def __init__(
        self, model: ExpectedGoalsTeamModel, max_goals: int = MAX_GOALS
    ) -> None:
        self.model = model
        self.max_goals = max_goals

    @property
    def teams(self) -> list[str] | None:
        return self.model.teams

    def describe_component(self) -> str:
        """
        This wrapper and the model inside it, for a run's record of its parts.

        The wrapper's own class name says which distribution was put over the
        goal counts and nothing about which model produced the mean, and the
        mean is the part a comparison is usually about.
        """
        return f"{type(self).__name__}({type(self.model).__name__})"

    def fit(self, training_data: TeamFitData) -> "PoissonScorelines":
        self.model.fit(training_data)
        return self

    def add_new_team(self, team_name: str, **kwargs: Any) -> None:
        self.model.add_new_team(team_name, **kwargs)

    def predict_expected_goals(
        self, team: str, opponent: str, home: bool = True, **kwargs: Any
    ) -> float:
        """The wrapped model's own prediction, passed straight through."""
        return self.model.predict_expected_goals(team, opponent, home, **kwargs)

    def predict_score_n_proba(
        self,
        n: int | Iterable[int],
        team: str,
        opponent: str,
        home: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        rate = self.model.predict_expected_goals(team, opponent, home, **kwargs)
        goals = np.atleast_1d(np.asarray(n))
        probability = np.where(
            goals < self.max_goals,
            poisson.pmf(np.clip(goals, 0, self.max_goals), rate),
            # everything from max_goals upwards, so the support sums to one
            poisson.sf(self.max_goals - 1, rate),
        )
        return np.where(goals >= 0, probability, 0.0).astype(float)

    def predict_outcome_proba(
        self, home_team: Sequence[str], away_team: Sequence[str]
    ) -> dict[str, np.ndarray]:
        return outcome_proba_from_scores(self, home_team, away_team, self.max_goals)


# A team model that predicts a mean needs a distribution over counts supplied,
# and the Poisson is an assumption about how goals scatter around a rate, not a
# measurement of it. Conway-Maxwell-Poisson is the Poisson with that assumption
# let go: the pmf is proportional to `rate ** n / factorial(n) ** dispersion`,
# so a dispersion of one is exactly a Poisson, above one is narrower and below
# one is wider.
POISSON_DISPERSION = 1.0
# Premier League goals are narrower than Poisson given a good estimate of the
# rate. This is the held-out optimum pooled over 2324, 2425 and 2526, swept with
# tools/tune_goal_dispersion.py; each season's own optimum is also above one
# (1.09, 1.14, 1.31), and every one of the three is better here than at a
# Poisson. Two decimal places because the surface is flat: anything from 1.15 to
# 1.20 is within a ten-thousandth of a nat of the peak. Choosing it this way is
# worth +0.0017 nats a side with a season held out, and it makes a clean sheet
# against an average side about 8% less likely than a Poisson would.
DEFAULT_GOAL_DISPERSION = 1.17
# A mean of zero has no rate to speak of, and no side is ever predicted to
# create nothing, so a mean is floored rather than special-cased.
MIN_MEAN = 1e-6


def conway_maxwell_pmf(
    mean: np.ndarray | float, dispersion: float, max_goals: int = MAX_GOALS
) -> np.ndarray:
    """
    Conway-Maxwell-Poisson probabilities of 0 to `max_goals` goals, per mean.

    Parameterised by the mean and not by the rate, which is both what a team
    model predicts and what makes a dispersion mean the same thing across
    models: changing the dispersion changes the shape and leaves the mean where
    it was. The rate that produces a given mean is solved for.

    The support is truncated at `max_goals` and renormalised, so `mean` is the
    mean of the truncated distribution. At league scoring rates that is a
    correction of order 1e-7.

    Returns:
        One row per mean, each summing to one, of shape (n_means, max_goals + 1).
    """
    means = np.atleast_1d(np.asarray(mean, dtype=float)).ravel()
    counts = np.arange(max_goals + 1, dtype=float)
    log_factorial = gammaln(counts + 1.0)
    target = np.clip(means, MIN_MEAN, float(max_goals))

    def pmf_of(log_rate: np.ndarray) -> np.ndarray:
        logits = log_rate[:, None] * counts[None, :] - dispersion * log_factorial
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        return np.asarray(weights / weights.sum(axis=1, keepdims=True), dtype=float)

    # Newton on the log rate. The mean rises with the rate and its derivative
    # with respect to the log rate is the variance, so starting from the Poisson
    # answer this converges in a few steps at any dispersion.
    log_rate = np.log(target)
    pmf = pmf_of(log_rate)
    for _ in range(32):
        predicted = pmf @ counts
        if np.abs(target - predicted).max() < 1e-12:
            break
        variance = pmf @ counts**2 - predicted**2
        log_rate = log_rate + np.clip(
            (target - predicted) / np.maximum(variance, 1e-12), -2.0, 2.0
        )
        pmf = pmf_of(log_rate)
    return pmf


class ConwayMaxwellScorelines(PoissonScorelines):
    """
    `PoissonScorelines` with the spread of the goal counts a measured number.

    A Poisson says a team's goals have a variance equal to their mean. Measured
    against what the xG model predicts, Premier League goals are narrower than
    that, so a family that can be narrower has something to say. `dispersion` is
    one number for the whole league, on the argument that how goals scatter
    around a rate is a property of football rather than of a team, and it is
    swept over held-out seasons rather than fitted while the model is: fitted in
    sample it comes out too narrow, because the ratings have already been fitted
    to the same matches.

    `POISSON_DISPERSION` recovers the Poisson - up to the truncated support,
    which this parameterises by the mean and `PoissonScorelines` by the rate.
    """

    def __init__(
        self,
        model: ExpectedGoalsTeamModel,
        max_goals: int = MAX_GOALS,
        dispersion: float = DEFAULT_GOAL_DISPERSION,
    ) -> None:
        super().__init__(model, max_goals)
        self.dispersion = dispersion

    def predict_score_n_proba(
        self,
        n: int | Iterable[int],
        team: str,
        opponent: str,
        home: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        mean = self.model.predict_expected_goals(team, opponent, home, **kwargs)
        pmf = conway_maxwell_pmf(mean, self.dispersion, self.max_goals)[0]
        goals = np.atleast_1d(np.asarray(n))
        # Anything at or above max_goals reads the last count, which carries the
        # whole of the tail, as `PoissonScorelines` does.
        probability = pmf[np.clip(goals, 0, self.max_goals)]
        return np.where(goals >= 0, probability, 0.0).astype(float)
