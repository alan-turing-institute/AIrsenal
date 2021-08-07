
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import reparam
from numpyro.infer import MCMC, NUTS

from typing import Any, Dict, Iterable, Optional, Union

class PlayerModel(object):
    """
    numpyro implementation of the AIrsenal player model.
    """
    def __init__(self):
        self.player_ids = None
        self.probs = None

    @staticmethod
    def _model(
            nplayer: int,
            nmatch: int,
            minutes: jnp.array,
            y: jnp.array,
            alpha: jnp.array
    ):
        theta = dist.Dirichlet(concentration=alpha)
        # one sample from the prior per player
        with numpyro.plate("nplayer", nplayer) as player_index:
            dprobs = numpyro.sample("probs", theta)
            # now it's all about how to broadcast in the right dimensions.....
        prob_score = numpyro.deterministic("prob_score",dprobs[:,0,None] * (minutes / 90.))
        prob_assist = numpyro.deterministic("prob_assist",dprobs[:,1,None] * (minutes / 90.))
        prob_neither = numpyro.deterministic("prob_neither",dprobs[:,2,None] * (minutes / 90.) + (90. - minutes))
        theta_mins = dist.Multinomial(
                probs=jnp.moveaxis(jnp.array([prob_score, prob_assist, prob_neither]),0,2)
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
            num_warmup=1500,
            num_samples=3000,
            num_chains=1,
            progress_bar=True
        )
        rng_key, rng_key_predict = random.split(random.PRNGKey(44))
        mcmc.run(
            rng_key,
            data["nplayer"],
            data["nmatch"],
            data["minutes"],
            data["y"],
            data["alpha"]
        )
        self.samples =  mcmc.get_samples()
        return self

    def get_probs(self):
        prob_dict = {"player_id": [],
                   "prob_score": [],
                   "prob_assist": [],
                   "prob_neither": []
                   }
        for i, pid in enumerate(self.player_ids):
            prob_dict["player_id"].append(pid)
            prob_dict["prob_score"].append(float(self.samples["probs"][:,i,0].mean()))
            prob_dict["prob_assist"].append(float(self.samples["probs"][:,i,1].mean()))
            prob_dict["prob_neither"].append(float(self.samples["probs"][:,i,2].mean()))
        return prob_dict
