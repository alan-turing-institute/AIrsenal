import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import pandas as pd

from typing import Any, Dict, Optional


def get_empirical_bayes_estimates(df_emp, prior_goals=None):
    """
    Get values to use either for Dirichlet prior alphas or for updating posterior
    in conjugate model. Returns number of goals, assists and neither scaled by the
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
    print("Alpha is {}".format(alpha))
    return alpha


class PlayerModel(object):
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
            "prob_neither", dprobs[:, 2, None] * (minutes / 90.0) + (90.0 - minutes)
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
        rng_key, rng_key_predict = random.split(random.PRNGKey(44))
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
        except (ValueError):
            raise RuntimeError(f"Unknown player_id {player_id}")
        prob_score = float(self.samples["probs"][:, index, 0].mean())
        prob_assist = float(self.samples["probs"][:, index, 1].mean())
        prob_neither = float(self.samples["probs"][:, index, 2].mean())
        return (prob_score, prob_assist, prob_neither)


class ConjugatePlayerModel(object):
    """Exact implementation of player model:
    Prior: Dirichlet(alpha)
    Posterior: Dirichlet(alpha + n)
    where n is the result of get_empirical_bayes_estimates for each player (i.e. total
    number of goal involvements for player weighted by amount of time on pitch).
    Strength of prior controlled by sum(alpha), by default 13 which is roughly the
    average no. of goals a team's expected to score in 10 matches. alpha values comes
    from average goal involvements for all players in that position.
    """
    def __init__(self):
        self.player_ids = None
        self.samples = None

    @staticmethod
    def _model(df, prior):
        neff = {
            idx: get_empirical_bayes_estimates(data)
            for idx, data in df.groupby("player_id")
        }
        neff = pd.DataFrame(neff).T
        neff = neff.fillna(0)
        neff.columns = ["prob_score", "prob_assist", "prob_neither"]
        return prior + neff

    def fit(self, data):

        dict(
            player_ids=player_ids,
            nplayer=nplayer,
            nmatch=nmatch,
            minutes=minutes.astype("int64"),
            y=y.astype("int64"),
            alpha=alpha,
        )
        alpha = data["alpha"]
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
        except (ValueError):
            raise RuntimeError(f"Unknown player_id {player_id}")
        prob_score = float(self.samples["probs"][:, index, 0].mean())
        prob_assist = float(self.samples["probs"][:, index, 1].mean())
        prob_neither = float(self.samples["probs"][:, index, 2].mean())
        return (prob_score, prob_assist, prob_neither)
