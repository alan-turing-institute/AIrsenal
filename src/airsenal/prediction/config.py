"""
Configuration for the player models.

Each model owns a frozen config dataclass rather than taking loose keyword
arguments, and defaults it, so the class needs no arguments to construct. The
team models take their settings as constructor arguments directly - see
`team_models/dixon_coles.py`.
"""

from dataclasses import dataclass

# Named here rather than inline so each value has one home; player_models.py imports
# this module, so this is the direction the dependency has to run.
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

    Deliberately has no epsilon or n_goals_prior: this model implements neither
    time weighting nor a goals prior, and used to swallow both silently.
    """

    num_warmup: int = 500
    num_samples: int = 2000
    num_chains: int = 1
    random_state: int = 42


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
