import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from typing import Any, Dict, Optional


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
