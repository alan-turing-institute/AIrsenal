"""The MCMC player model, implemented with numpyro."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from airsenal.prediction.protocols import PlayerFitData

if TYPE_CHECKING:
    import jax.numpy as jnp


# How much of a match a player who did not appear is treated as having played.
# See `NumpyroPlayerModel._model` for why it is not zero.
MIN_PLAYED_FRACTION = 1e-6


@dataclass(frozen=True)
class NumpyroPlayerConfig:
    """
    Settings for the MCMC player model.

    Deliberately has no epsilon or n_goals_prior: this model implements neither
    time weighting nor a goals prior, so it rejects them rather than accepting
    and ignoring them.
    """

    num_warmup: int = 500
    num_samples: int = 2000
    num_chains: int = 1
    random_state: int = 42


class NumpyroPlayerModel:
    """NumPyro implementation of the AIrsenal player model, fitted by MCMC."""

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
        # Floored away from zero. A match the player did not appear in gives
        # probabilities of exactly (0, 0, 1) and observed counts of (0, 0, n), and
        # the multinomial's 0 * log(0) term is a finite 0 whose derivative is 0/0 -
        # so every unplayed match puts a NaN in the gradient and NUTS reports that
        # it cannot find valid initial parameters. Those matches say nothing about
        # the player either way, so moving them a hair off the boundary costs
        # nothing and makes the gradient finite.
        played = jnp.clip(minutes / 90.0, MIN_PLAYED_FRACTION, 1.0)
        prob_score = numpyro.deterministic("prob_score", dprobs[:, 0, None] * played)
        prob_assist = numpyro.deterministic("prob_assist", dprobs[:, 1, None] * played)
        prob_neither = numpyro.deterministic(
            "prob_neither", dprobs[:, 2, None] * played + (1.0 - played)
        )
        # total_count is the goals the player's team scored in that match, which
        # is what the three outcome counts in `y` sum to. It varies per player and
        # match, so it has to be passed: Multinomial's default of 1 puts every row
        # where the team scored anything other than once outside the support, and
        # numpyro rejects the initial parameters rather than fitting.
        theta_mins = dist.Multinomial(
            total_count=y.sum(-1),
            probs=jnp.moveaxis(
                jnp.array([prob_score, prob_assist, prob_neither]), 0, 2
            ),
        )
        return numpyro.sample("obs", theta_mins, obs=y)

    def fit(self, data: PlayerFitData) -> "NumpyroPlayerModel":
        import jax.random as random  # noqa: PLC0415
        from numpyro.infer import MCMC, NUTS  # noqa: PLC0415

        alpha = np.asarray(data["alpha"])
        if not (alpha > 0).all():
            # A Dirichlet needs strictly positive concentrations, and numpyro
            # reports a bare "invalid concentration parameter" that says nothing
            # about where it came from. A zero here means the fitting window
            # contains no goals, or no assists, for the whole position.
            msg = (
                f"Dirichlet prior {alpha} has a non-positive concentration, so "
                "this position recorded no goals or no assists in the fitting "
                "window. Fit over more gameweeks, or use a model that does not "
                "need a proper prior, such as --player-model conjugate."
            )
            raise ValueError(msg)

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
