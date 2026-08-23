from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from airsenal.core.logging import get_logger
from airsenal.core.types import FloatArray
from airsenal.prediction.config import (
    ConjugatePlayerConfig,
    ConstantPlayerConfig,
    NumpyroPlayerConfig,
)

if TYPE_CHECKING:
    import jax.numpy as jnp

logger = get_logger(__name__)


def get_empirical_bayes_estimates(
    df_emp: pd.DataFrame, prior_goals: float | None = None
) -> FloatArray:
    """
    Get values to use either for Dirichlet prior alphas in the original Stan and numpyro
    player models. Returns number of goals, assists and neither scaled by the
    proportion of minutes & no. matches a player is involved in. If df_emp contains more
    than one player, result is average across all players.

    If prior_goals is not None, normalise the returned alpha values to sum to
    prior_goals.
    """
    # for compatibility with models we zero pad data so all players have
    # the same number of rows (matches). Remove the dummy matches:
    df = df_emp.copy()
    df = df[df["match_id"] != 0]

    player_goals = df["goals"].sum()
    player_assists = df["assists"].sum()
    player_neither = df["neither"].sum()
    player_minutes = df["minutes"].sum()
    team_goals = df["team_goals"].sum()
    total_minutes = 90 * len(df)
    n_matches = df.groupby("player_name").count()["goals"].mean()

    # Total no. of player goals, assists, neither:
    # no. matches played * fraction goals scored * (1 / fraction mins played)
    a0 = n_matches * (player_goals / team_goals) * (total_minutes / player_minutes)
    a1 = n_matches * (player_assists / team_goals) * (total_minutes / player_minutes)
    a2 = (
        n_matches
        * (
            (player_neither / team_goals)
            - (total_minutes - player_minutes) / total_minutes
        )
        * (total_minutes / player_minutes)
    )
    alpha = np.array([a0, a1, a2])
    if prior_goals is not None:
        alpha = prior_goals * (alpha / alpha.sum())
    return alpha


def scale_goals_by_minutes(
    goals: np.ndarray,
    minutes: np.ndarray,
    time_diff: np.ndarray | None = None,
    epsilon: float | None = None,
    rescale_weights: bool = True,
) -> FloatArray:
    """
    Scale player goal involvements by the proportion of minutes they played
    (specifically: reduce the number of "neither" goals where the player is said
    to have had no involvement.
    goals: np.array with shape (n_players, n_matches, 3) where last axis is no. goals,
    no. assists, and no. goals not involved in
    minutes: np.array with shape (n_players, m_matches)
    time_diff: np.array with shape (n_players, m_matches)
    epsilon: float for weight decay rate with time
    rescale_weights: bool indicating whether to rescale weights to sum to n_matches for
    each player (n_matches the player appeared in where a goal was scored)
    """
    if epsilon is not None and time_diff is None:
        msg = "time_diff must be provided if using time weighting."
        raise ValueError(msg)
    if time_diff is not None and epsilon is not None:
        weights = np.exp(-epsilon * time_diff)
    else:
        weights = np.ones_like(minutes)
    select_matches = (goals.sum(axis=2) > 0) & (minutes > 0)
    n_players, _, _ = goals.shape
    scaled_goals = np.zeros((n_players, 3))
    for p in range(n_players):
        if select_matches[p, :].sum() == 0:
            # player not involved in any matches with goals
            scaled_goals[p, :] = [0, 0, 0]
            continue

        match_weights = weights[p, select_matches[p, :]]
        if rescale_weights:
            match_weights = (
                select_matches[p, :].sum() * match_weights / match_weights.sum()
            )
        team_goals = (
            goals[p, select_matches[p, :], :].sum(axis=1) * match_weights
        ).sum()
        team_mins = (90 * match_weights).sum()
        player_mins = (minutes[p, select_matches[p, :]] * match_weights).sum()
        player_goals = (goals[p, select_matches[p, :], 0] * match_weights).sum()
        player_assists = (goals[p, select_matches[p, :], 1] * match_weights).sum()
        player_neither = (
            team_goals * (player_mins / team_mins) - player_goals - player_assists
        )
        scaled_goals[p, :] = [player_goals, player_assists, player_neither]

    # players with high goal involvements in few matches may end up with a scaled
    # neither count less than 0 - set these to zero
    scaled_goals[scaled_goals < 0] = 0

    return scaled_goals


class BasePlayerModel(ABC):
    """
    Base class for player models
    """

    @abstractmethod
    def fit(self, data: dict[str, Any]) -> "BasePlayerModel":
        """Fit model, using the hyperparameters this model was constructed with.

        Deliberately takes no **kwargs: it used to, and NumpyroPlayerModel silently
        swallowed the epsilon and n_goals_prior the caller passed.

        Data must have the following keys (at minimum):
        - "y": np.ndarray of shape (n_players, n_matches, 3) with player goal
        involvements in each match. Last axis is (no. goals, no. assists, no. neither)
        - "player_ids": np.ndarray of shape (n_players,) with player ids
        - "minutes": np.ndarray of shape (n_players, m_matches) - no. minutes played by
        each player in each match
        """
        ...

    @abstractmethod
    def get_probs(self) -> dict[str, np.ndarray]:
        """Get probability of all players scoring, assisting or doing neither for a
        goal. Returns dict with followinig keys:
        - "player_id": np.ndarray of shape (n_players,) with player ids
        - "prob_score": np.ndarray of shape (n_players,) with goal probabilities
        - "prob_assist": np.ndarray of shape (n_players,) with assist probabilities
        - "prob_neither": np.ndarray of shape (n_players,) with neither probabilities
        """
        ...


class NumpyroPlayerModel(BasePlayerModel):
    """
    numpyro implementation of the AIrsenal player model.
    """

    def __init__(self, config: NumpyroPlayerConfig | None = None):
        self.config = config or NumpyroPlayerConfig()
        self.player_ids: np.ndarray | None = None
        self.samples: dict[str, Any] | None = None

    @staticmethod
    def _model(
        nplayer: int,
        nmatch: int,  # noqa: ARG004
        minutes: "jnp.ndarray",
        y: "jnp.ndarray",
        alpha: "jnp.ndarray",
    ) -> "jnp.ndarray":
        # jax and numpyro are imported here rather than at module scope: they cost
        # seconds to import, and only this model needs them.
        import jax.numpy as jnp  # noqa: PLC0415
        import numpyro  # noqa: PLC0415
        import numpyro.distributions as dist  # noqa: PLC0415

        theta = dist.Dirichlet(concentration=alpha)
        # one sample from the prior per player
        with numpyro.plate("nplayer", nplayer):
            dprobs = numpyro.sample("probs", theta)
            # now it's all about how to broadcast in the right dimensions.....
        if not isinstance(dprobs, jnp.ndarray):
            dprobs = jnp.array(dprobs)
        prob_score = numpyro.deterministic(
            "prob_score", dprobs[:, 0, None] * (minutes / 90.0)
        )
        prob_assist = numpyro.deterministic(
            "prob_assist", dprobs[:, 1, None] * (minutes / 90.0)
        )
        prob_neither = numpyro.deterministic(
            "prob_neither",
            dprobs[:, 2, None] * (minutes / 90.0) + (90.0 - minutes) / 90.0,
        )
        theta_mins = dist.Multinomial(
            probs=jnp.moveaxis(jnp.array([prob_score, prob_assist, prob_neither]), 0, 2)
        )
        return numpyro.sample("obs", theta_mins, obs=y)

    def fit(self, data: dict[str, Any]) -> "NumpyroPlayerModel":
        import jax.random as random  # noqa: PLC0415
        from numpyro.infer import MCMC, NUTS  # noqa: PLC0415

        self.player_ids = data["player_ids"]
        kernel = NUTS(self._model)
        mcmc = MCMC(
            kernel,
            num_warmup=self.config.num_warmup,
            num_samples=self.config.num_samples,
            num_chains=self.config.num_chains,
            progress_bar=True,
        )
        rng_key, _rng_key_predict = random.split(
            random.PRNGKey(self.config.random_state)
        )
        mcmc.run(
            rng_key,
            data["nplayer"],
            data["nmatch"],
            data["minutes"],
            data["y"],
            data["alpha"],
        )
        self.samples = mcmc.get_samples()
        return self

    def get_probs(self) -> dict[str, np.ndarray]:
        if self.samples is None or self.player_ids is None:
            msg = "Model samples or player_ids have not been set yet."
            raise RuntimeError(msg)
        prob_dict = {
            "player_id": np.zeros_like(self.player_ids, dtype=int),
            "prob_score": np.zeros_like(self.player_ids, dtype=float),
            "prob_assist": np.zeros_like(self.player_ids, dtype=float),
            "prob_neither": np.zeros_like(self.player_ids, dtype=float),
        }
        for i, pid in enumerate(self.player_ids):
            prob_dict["player_id"][i] = pid
            prob_dict["prob_score"][i] = float(self.samples["probs"][:, i, 0].mean())
            prob_dict["prob_assist"][i] = float(self.samples["probs"][:, i, 1].mean())
            prob_dict["prob_neither"][i] = float(self.samples["probs"][:, i, 2].mean())
        return prob_dict


class ConjugatePlayerModel(BasePlayerModel):
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
    def get_prior(scaled_goals: np.ndarray, n_goals_prior: int) -> np.ndarray:
        """Compute alpha parameters for Dirichlet prior. Calculated by summing
        up all player goal involvements, then normalise to sum to n_goals_prior.
        """
        alpha = scaled_goals.sum(axis=0)
        return n_goals_prior * alpha / alpha.sum()

    @staticmethod
    def get_posterior(prior_alpha: np.ndarray, scaled_goals: np.ndarray) -> np.ndarray:
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


class ConstantPlayerModel(BasePlayerModel):
    """
    Every player equally likely to score, assist, or do neither.

    A null baseline, and a fast path when debugging something downstream of
    prediction, since it does no fitting at all.
    """

    def __init__(self, config: ConstantPlayerConfig | None = None) -> None:
        self.config = config or ConstantPlayerConfig()
        self.player_ids: np.ndarray | None = None

    def fit(self, data: dict[str, Any]) -> "ConstantPlayerModel":
        self.player_ids = data["player_ids"]
        return self

    def _probabilities(self) -> np.ndarray:
        return np.array(
            [
                self.config.prob_score,
                self.config.prob_assist,
                1.0 - self.config.prob_score - self.config.prob_assist,
            ]
        )

    def get_probs(self) -> dict[str, np.ndarray]:
        if self.player_ids is None:
            msg = "Model has not been fitted yet."
            raise RuntimeError(msg)
        probs = self._probabilities()
        n = len(self.player_ids)
        return {
            "player_id": self.player_ids,
            "prob_score": np.full(n, probs[0]),
            "prob_assist": np.full(n, probs[1]),
            "prob_neither": np.full(n, probs[2]),
        }
