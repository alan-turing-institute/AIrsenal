"""The conjugate Bayesian player model: a Dirichlet prior updated in closed form."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from airsenal.core.logging import get_logger
from airsenal.prediction.player_models.scaling import (
    FloatArray,
    scale_goals_by_minutes,
)

logger = get_logger(__name__)

# Named constants rather than inline literals so each value has one home.
DEFAULT_PLAYER_EPSILON = 0.2
DEFAULT_N_GOALS_PRIOR = 35


@dataclass(frozen=True)
class ConjugatePlayerConfig:
    """Settings for the conjugate Bayesian player model."""

    # None disables time weighting entirely.
    epsilon: float | None = DEFAULT_PLAYER_EPSILON
    n_goals_prior: int = DEFAULT_N_GOALS_PRIOR
    rescale_weights: bool = True


class ConjugatePlayerModel:
    """Exact implementation of player model:
    - Prior: Dirichlet(alpha)
    - Posterior: Dirichlet(alpha + n)
    where n is the result of scale_goals_by_minutes for each player (i.e. total
    number of goal involvements for player weighted by amount of time on pitch).
    Strength of prior controlled by sum(alpha), by default 13 which is roughly the
    average no. of goals a team's expected to score in 10 matches. alpha values come
    from average goal involvements for all players in that position.
    """

    def __init__(self, config: ConjugatePlayerConfig | None = None):
        self.config = config or ConjugatePlayerConfig()
        self.player_ids: np.ndarray | None = None
        self.prior: np.ndarray | None = None
        self.posterior: np.ndarray | None = None
        self.mean_probabilities: np.ndarray | None = None
        self.time_diff: np.ndarray | None = None

    @property
    def epsilon(self) -> float | None:
        return self.config.epsilon

    @property
    def rescale_weights(self) -> bool:
        return self.config.rescale_weights

    def fit(self, data: dict[str, Any]) -> "ConjugatePlayerModel":
        logger.info(
            "Fitting ConjugatePlayerModel with epsilon=%s, rescale_weights=%s, "
            "n_goals_prior=%s",
            self.config.epsilon,
            self.config.rescale_weights,
            self.config.n_goals_prior,
        )
        goals = data["y"]
        minutes = data["minutes"]
        time_diff = data.get("time_diff")
        self.player_ids = data["player_ids"]

        scaled_goals = scale_goals_by_minutes(
            goals=goals,
            minutes=minutes,
            time_diff=time_diff,
            epsilon=self.config.epsilon,
            rescale_weights=self.config.rescale_weights,
        )
        self.prior = self.get_prior(
            scaled_goals, n_goals_prior=self.config.n_goals_prior
        )
        posterior = self.get_posterior(self.prior, scaled_goals)
        self.posterior = posterior
        self.mean_probabilities = self.posterior / self.posterior.sum(axis=1)[:, None]

        return self

    @staticmethod
    def get_prior(scaled_goals: FloatArray, n_goals_prior: int) -> FloatArray:
        """Compute alpha parameters for Dirichlet prior. Calculated by summing
        up all player goal involvements, then normalise to sum to n_goals_prior.
        """
        alpha = scaled_goals.sum(axis=0)
        return n_goals_prior * alpha / alpha.sum()

    @staticmethod
    def get_posterior(prior_alpha: FloatArray, scaled_goals: FloatArray) -> FloatArray:
        """Compute parameters of Dirichlet posterior, which is the sum of the prior
        and scaled goal involvements.
        """
        return prior_alpha + scaled_goals

    def get_probs(self) -> dict[str, np.ndarray]:
        if self.player_ids is None or self.mean_probabilities is None:
            msg = "Model player_ids or mean_probabilities have not been set yet."
            raise RuntimeError(msg)
        return {
            "player_id": self.player_ids,
            "prob_score": self.mean_probabilities[:, 0],
            "prob_assist": self.mean_probabilities[:, 1],
            "prob_neither": self.mean_probabilities[:, 2],
        }
