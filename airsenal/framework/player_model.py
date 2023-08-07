from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def get_empirical_bayes_estimates(df_emp, prior_goals=None):
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


def scale_goals_by_minutes(goals, minutes):
    """
    Scale player goal involvements by the proportion of minutes they played
    (specifically: reduce the number of "neither" goals where the player is said
    to have had no involvement.
    goals: np.array with shape (n_players, n_matches, 3) where last axis is no. goals,
    mo. assists and no. goals not involved in
    minutes: np.array with shape (n_players, m_matches)
    """
    select_matches = (goals.sum(axis=2) > 0) & (minutes > 0)
    n_players, _, _ = goals.shape
    scaled_goals = np.zeros((n_players, 3))
    for p in range(n_players):
        if select_matches[p, :].sum() == 0:
            # player not involved in any matches with goals
            scaled_goals[p, :] = [0, 0, 0]
            continue

        team_goals = goals[p, select_matches[p, :], :].sum()
        team_mins = 90 * select_matches[p, :].sum()
        player_mins = minutes[p, select_matches[p, :]].sum()
        player_goals = goals[p, select_matches[p, :], 0].sum()
        player_assists = goals[p, select_matches[p, :], 1].sum()
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
    def fit(self, data: Dict[str, Any], **kwargs) -> BasePlayerModel:
        """Fit model. Data must have the following keys (at minimum):
        - "y": np.ndarray of shape (n_players, n_matches, 3) with player goal
        involvements in each match. Last axis is (no. goals, no. assists, no. neither)
        - "player_ids": np.ndarray of shape (n_players,) with player ids
        - "minutes": np.ndarray of shape (n_players, m_matches) - no. minutes played by
        each player in each match
        """
        ...

    @abstractmethod
    def get_probs(self) -> Dict[str, np.ndarray]:
        """Get probability of all players scoring, assisting or doing neither for a
        goal. Returns dict with followinig keys:
        - "player_id": np.ndarray of shape (n_players,) with player ids
        - "prob_score": np.ndarray of shape (n_players,) with goal probabilities
        - "prob_assist": np.ndarray of shape (n_players,) with assist probabilities
        - "prob_neither": np.ndarray of shape (n_players,) with neither probabilities
        """
        ...

    @abstractmethod
    def get_probs_for_player(self, player_id: int) -> np.ndarray:
        """Get probability that a player scores, assists or does neither for a goal
        their team scores. Returns np.ndarray of shape (3, )."""
        ...


class NumpyroPlayerModel(BasePlayerModel):
    """
    numpyro implementation of the AIrsenal player model.
    """

    def __init__(self):
        self.player_ids = None
        self.samples = None

    @staticmethod
    def _model(
        nplayer: int, nmatch: int, minutes: jnp.array, y: jnp.array, alpha: jnp.array
    ):
        theta = dist.Dirichlet(concentration=alpha)
        # one sample from the prior per player
        with numpyro.plate("nplayer", nplayer):
            dprobs = numpyro.sample("probs", theta)
            # now it's all about how to broadcast in the right dimensions.....
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

    def fit(
        self,
        data,
        random_state: int = 42,
        num_warmup: int = 500,
        num_samples: int = 2000,
        mcmc_kwargs: Optional[Dict[str, Any]] = None,
        run_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.player_ids = data["player_ids"]
        kernel = NUTS(self._model)
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=1,
            progress_bar=True,
            **(mcmc_kwargs or {}),
        )
        rng_key, rng_key_predict = random.split(random.PRNGKey(random_state))
        mcmc.run(
            rng_key,
            data["nplayer"],
            data["nmatch"],
            data["minutes"],
            data["y"],
            data["alpha"],
            **(run_kwargs or {}),
        )
        self.samples = mcmc.get_samples()
        return self

    def get_probs(self):
        prob_dict = {
            "player_id": [],
            "prob_score": [],
            "prob_assist": [],
            "prob_neither": [],
        }
        for i, pid in enumerate(self.player_ids):
            prob_dict["player_id"].append(pid)
            prob_dict["prob_score"].append(float(self.samples["probs"][:, i, 0].mean()))
            prob_dict["prob_assist"].append(
                float(self.samples["probs"][:, i, 1].mean())
            )
            prob_dict["prob_neither"].append(
                float(self.samples["probs"][:, i, 2].mean())
            )
        return prob_dict

    def get_probs_for_player(self, player_id):
        try:
            index = list(self.player_ids).index(player_id)
        except ValueError:
            raise RuntimeError(f"Unknown player_id {player_id}")
        prob_score = float(self.samples["probs"][:, index, 0].mean())
        prob_assist = float(self.samples["probs"][:, index, 1].mean())
        prob_neither = float(self.samples["probs"][:, index, 2].mean())
        return (prob_score, prob_assist, prob_neither)


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

    def __init__(self):
        self.player_ids = None
        self.prior = None
        self.posterior = None
        self.mean_probabilities = None

    def fit(
        self, data: Dict[str, Any], n_goals_prior: int = 13
    ) -> ConjugatePlayerModel:
        goals = data["y"]
        minutes = data["minutes"]
        self.player_ids = data["player_ids"]

        scaled_goals = scale_goals_by_minutes(goals, minutes)
        self.prior = self.get_prior(scaled_goals, n_goals_prior=n_goals_prior)
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

    def get_probs(self) -> Dict[str, np.ndarray]:
        return {
            "player_id": self.player_ids,
            "prob_score": self.mean_probabilities[:, 0],
            "prob_assist": self.mean_probabilities[:, 1],
            "prob_neither": self.mean_probabilities[:, 2],
        }

    def get_probs_for_player(self, player_id: int) -> np.ndarray:
        try:
            index = list(self.player_ids).index(player_id)
        except ValueError:
            raise RuntimeError(f"Unknown player_id {player_id}")
        return self.mean_probabilities[index, :]
