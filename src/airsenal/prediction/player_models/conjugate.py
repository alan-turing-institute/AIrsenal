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
        self.prior, self.posterior, self.mean_probabilities = dirichlet_update(
            scaled_goals, self.config.n_goals_prior
        )
        return self

    @staticmethod
    def get_prior(scaled_goals: FloatArray, n_goals_prior: int) -> FloatArray:
        """Sum every player's goal involvements, normalised to sum to n_goals_prior."""
        alpha = scaled_goals.sum(axis=0)
        return n_goals_prior * alpha / alpha.sum()

    def predict_involvement(self) -> PlayerInvolvement:
        return involvement_from_mean(
            self.player_ids,
            self.mean_probabilities,
            "Model player_ids or mean_probabilities have not been set yet.",
        )


def dirichlet_update(
    scaled_goals: FloatArray, n_goals_prior: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    The pooled prior, the posterior, and the posterior mean per player.

    The posterior is the prior plus each player's scaled goal involvements.
    """
    prior = ConjugatePlayerModel.get_prior(scaled_goals, n_goals_prior=n_goals_prior)
    posterior = prior + scaled_goals
    return prior, posterior, posterior / posterior.sum(axis=1)[:, None]


def involvement_from_mean(
    player_ids: FloatArray | None,
    mean_probabilities: FloatArray | None,
    not_fitted: str,
) -> PlayerInvolvement:
    """
    The involvement a fitted Dirichlet mean gives each player.

    Raises:
        RuntimeError: With `not_fitted` as its message, if either is None.
    """
    if player_ids is None or mean_probabilities is None:
        raise RuntimeError(not_fitted)
    return PlayerInvolvement(
        player_ids=player_ids,
        prob_score=mean_probabilities[:, 0],
        prob_assist=mean_probabilities[:, 1],
        prob_neither=mean_probabilities[:, 2],
    )
