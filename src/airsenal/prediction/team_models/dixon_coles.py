"""
Interface to the NumPyro team model in bpl-next.

https://github.com/anguswilliams91/bpl-next
"""

from typing import Any

import numpy as np
from bpl import ExtendedDixonColesMatchPredictor, NeutralDixonColesMatchPredictor

from airsenal.core.logging import get_logger
from airsenal.prediction.protocols import TeamFitData

logger = get_logger(__name__)

# Default time weighting, calculated as best on average across 20/21 to 24/25,
# assuming 3 seasons of history before the current season in the DB and
# predicting 5 weeks ahead.
DEFAULT_TEAM_EPSILON = 0.9
# Rescale weights to sum to the number of matches in the training data (what they
# would sum to with no time weighting). The epsilon above is optimal for True.
DEFAULT_RESCALE_WEIGHTS = True


class DixonColesTeamModel:
    """bpl's Dixon-Coles predictor, holding the arguments it is fitted with."""

    def __init__(
        self,
        *,
        neutral: bool = False,
        epsilon: float | None = None,
        rescale_weights: bool = DEFAULT_RESCALE_WEIGHTS,
    ) -> None:
        self.neutral = neutral
        self.epsilon = DEFAULT_TEAM_EPSILON if epsilon is None else epsilon
        self.rescale_weights = rescale_weights
        self.model = (
            NeutralDixonColesMatchPredictor()
            if neutral
            else ExtendedDixonColesMatchPredictor()
        )

    @property
    def teams(self) -> list[str] | None:
        return self.model.teams

    def fit(self, training_data: TeamFitData) -> "DixonColesTeamModel":
        logger.info(
            "Using %s model with epsilon=%s, rescale_weights=%s",
            type(self.model).__name__,
            self.epsilon,
            self.rescale_weights,
        )
        self.model.fit(
            training_data=training_data,
            epsilon=self.epsilon,
            rescale_weights=self.rescale_weights,
        )
        return self

    def add_new_team(self, team_name: str, **kwargs: Any) -> None:
        self.model.add_new_team(team_name, **kwargs)

    def predict_score_n_proba(
        self, n: np.ndarray, team: str, opponent: str, home: bool = True, **kwargs: Any
    ) -> np.ndarray:
        return self.model.predict_score_n_proba(n, team, opponent, home, **kwargs)

    def predict_outcome_proba(
        self, home_team: Any, away_team: Any
    ) -> dict[str, np.ndarray]:
        """Home win, draw and away win probabilities for each fixture."""
        if self.neutral:
            return self.model.predict_outcome_proba(
                home_team, away_team, neutral_venue=np.zeros(len(home_team))
            )
        return self.model.predict_outcome_proba(home_team, away_team)
