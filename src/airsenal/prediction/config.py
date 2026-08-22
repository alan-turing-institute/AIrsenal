"""
Configuration for the prediction models.

Each model owns a frozen config dataclass rather than taking loose keyword
arguments. That is what makes the choice of model and its settings expressible as
data - and it removes the place where hyperparameters used to disappear: fit()
accepted **kwargs, so passing epsilon to a model that does not implement time
weighting silently did nothing.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ConjugatePlayerConfig:
    """Settings for the conjugate Bayesian player model."""

    # None disables time weighting entirely.
    epsilon: float | None = 0.2
    n_goals_prior: int = 35
    rescale_weights: bool = True


@dataclass(frozen=True)
class NumpyroPlayerConfig:
    """
    Settings for the MCMC player model.

    Deliberately has no epsilon or n_goals_prior: this model does not implement
    time weighting or a goals prior. Asking for either is now an error from the
    registry rather than a silent no-op.
    """

    num_warmup: int = 500
    num_samples: int = 2000
    num_chains: int = 1
    random_state: int = 42


@dataclass(frozen=True)
class DixonColesConfig:
    """Settings for the BPL Dixon-Coles team models."""

    epsilon: float = 0.9  # time-weighting decay rate
    rescale_weights: bool = True

    def fit_args(self) -> dict[str, object]:
        """bpl takes these when fitting rather than when constructing."""
        return {"epsilon": self.epsilon, "rescale_weights": self.rescale_weights}


@dataclass(frozen=True)
class RandomTeamModelConfig:
    """The null team model takes no settings; the class exists for the registry."""

    def fit_args(self) -> dict[str, object]:
        return {}
