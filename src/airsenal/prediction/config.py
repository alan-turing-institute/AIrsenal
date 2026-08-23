"""
Configuration for the prediction models.

Each model owns a frozen config dataclass rather than taking loose keyword
arguments. That is what makes the choice of model and its settings expressible as
data - and it removes the place where hyperparameters used to disappear: fit()
accepted **kwargs, so passing epsilon to a model that does not implement time
weighting silently did nothing.
"""

from dataclasses import dataclass

# Named rather than written inline below, so the values have one home. They used to
# live in player_models.py, which imports this module - so this is the direction the
# dependency has to run. The team-model ones came from team_models/dixon_coles.py for
# the same reason: three modules re-stated `{"epsilon": DEFAULT_TEAM_EPSILON}` as their
# own fallback, so the "default" existed in four places and could disagree.
DEFAULT_PLAYER_EPSILON = 0.2
DEFAULT_N_GOALS_PRIOR = 35

# Default time weighting for team model, calculated using best on average across 20/21
# to 24/25 season, assuming 3 seasons of history before the current season in the DB and
# predicting 5 weeks ahead.
DEFAULT_TEAM_EPSILON = 0.9
# Rescale weights to sum to number of matches in training data (what they would sum to
# if no time weighting would apply to the model). The optimal value of epsilon above is
# for the case where this is True.
DEFAULT_RESCALE_WEIGHTS = True


@dataclass(frozen=True)
class ConjugatePlayerConfig:
    """Settings for the conjugate Bayesian player model."""

    # None disables time weighting entirely.
    epsilon: float | None = DEFAULT_PLAYER_EPSILON
    n_goals_prior: int = DEFAULT_N_GOALS_PRIOR
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

    epsilon: float = DEFAULT_TEAM_EPSILON  # time-weighting decay rate
    rescale_weights: bool = DEFAULT_RESCALE_WEIGHTS

    def fit_args(self) -> dict[str, object]:
        """bpl takes these when fitting rather than when constructing."""
        return {"epsilon": self.epsilon, "rescale_weights": self.rescale_weights}


@dataclass(frozen=True)
class RandomTeamModelConfig:
    """The null team model takes no settings; the class exists for the registry."""

    def fit_args(self) -> dict[str, object]:
        return {}


@dataclass(frozen=True)
class ConstantPlayerConfig:
    """
    Settings for the null player model.

    The defaults are roughly the league-wide split of goal involvements, so the
    baseline is uninformative rather than obviously wrong.
    """

    prob_score: float = 0.25
    prob_assist: float = 0.2

    def __post_init__(self) -> None:
        if self.prob_score + self.prob_assist > 1:
            msg = (
                f"prob_score + prob_assist must not exceed 1, got "
                f"{self.prob_score} + {self.prob_assist}"
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class ConstantTeamModelConfig:
    """Settings for the null team model."""

    max_goals: int = 10

    def fit_args(self) -> dict[str, object]:
        """Nothing to pass at fit time."""
        return {}
