"""The conjugate Bayesian player model: a Dirichlet prior updated in closed form."""

from dataclasses import dataclass

from airsenal.core.logging import get_logger
from airsenal.prediction.player_models.scaling import (
    FloatArray,
    scale_goals_by_minutes,
)
from airsenal.prediction.protocols import PlayerFitData, PlayerInvolvement

logger = get_logger(__name__)

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
    """
    A Dirichlet prior updated in closed form, with no sampling.

    The posterior is Dirichlet(alpha + n), where n is `scale_goals_by_minutes`
    per player - goal involvements weighted by time on the pitch. The prior
    pools every player in the data it is fitted to, normalised so that
    `sum(alpha)` is `n_goals_prior`, which is therefore what sets how strongly
    the prior pulls an individual player towards the pool.
    """

    def __init__(self, config: ConjugatePlayerConfig | None = None):
        self.config = config or ConjugatePlayerConfig()
        self.player_ids: FloatArray | None = None
        self.prior: FloatArray | None = None
        self.posterior: FloatArray | None = None
        self.mean_probabilities: FloatArray | None = None

    def fit(self, data: PlayerFitData) -> "ConjugatePlayerModel":
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
        """Sum every player's goal involvements, normalised to sum to n_goals_prior."""
        alpha = scaled_goals.sum(axis=0)
        return n_goals_prior * alpha / alpha.sum()

    @staticmethod
    def get_posterior(prior_alpha: FloatArray, scaled_goals: FloatArray) -> FloatArray:
        """The Dirichlet posterior: the prior plus the scaled goal involvements."""
        return prior_alpha + scaled_goals

    def predict_involvement(self) -> PlayerInvolvement:
        if self.player_ids is None or self.mean_probabilities is None:
            msg = "Model player_ids or mean_probabilities have not been set yet."
            raise RuntimeError(msg)
        return PlayerInvolvement(
            player_ids=self.player_ids,
            prob_score=self.mean_probabilities[:, 0],
            prob_assist=self.mean_probabilities[:, 1],
            prob_neither=self.mean_probabilities[:, 2],
        )
