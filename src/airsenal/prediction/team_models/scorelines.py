"""
Turning what a team model says about goals into what the points calculation needs.

Two directions. `outcome_proba_from_scores` goes from per-team goal
distributions to win/draw/loss, for a model that treats the two teams'
scorelines as independent - bpl does this itself, so `DixonColesTeamModel` does
not come through here. `PoissonScorelines` goes the other way, from a model
that predicts only an expected number of goals to one that has a distribution
over goal counts.
"""

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
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
    """

    def __init__(
        self, model: ExpectedGoalsTeamModel, max_goals: int = MAX_GOALS
    ) -> None:
        self.model = model
        self.max_goals = max_goals

    @property
    def teams(self) -> list[str] | None:
        return self.model.teams

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
