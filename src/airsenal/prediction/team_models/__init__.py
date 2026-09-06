"""
Team models: one module per way of predicting match scorelines.

`TEAM_MODELS` maps a `--team-model` name to a factory. Unlike the other
component tables its factories are not zero-argument: each takes an optional
keyword-only `epsilon`, the time-weighting decay rate.
"""

from collections.abc import Callable

from airsenal.core.lookup import ConfigError, lookup
from airsenal.prediction.protocols import ScorelineTeamModel

DEFAULT_TEAM_MODEL = "extended"


# The Dixon-Coles entries are functions rather than the class itself so that bpl,
# and therefore jax, is imported only when one is actually built.
def _extended(*, epsilon: float | None = None) -> ScorelineTeamModel:
    from airsenal.prediction.team_models.dixon_coles import (  # noqa: PLC0415
        DixonColesTeamModel,
    )

    return DixonColesTeamModel(epsilon=epsilon)


def _neutral(*, epsilon: float | None = None) -> ScorelineTeamModel:
    from airsenal.prediction.team_models.dixon_coles import (  # noqa: PLC0415
        DixonColesTeamModel,
    )

    return DixonColesTeamModel(neutral=True, epsilon=epsilon)


def _random(*, epsilon: float | None = None) -> ScorelineTeamModel:
    from airsenal.prediction.team_models.random_model import (  # noqa: PLC0415
        RandomTeamModel,
    )

    return RandomTeamModel(epsilon=epsilon)


def _xg(*, epsilon: float | None = None) -> ScorelineTeamModel:
    """
    Expected goals, read as a Poisson over goal counts.

    The one entry in the table that wraps: `XGTeamModel` predicts a mean and
    `PoissonScorelines` gives it the distribution the points calculation needs.
    """
    from airsenal.prediction.team_models.scorelines import (  # noqa: PLC0415
        PoissonScorelines,
    )
    from airsenal.prediction.team_models.xg import (  # noqa: PLC0415
        XGTeamConfig,
        XGTeamModel,
    )

    config = XGTeamConfig() if epsilon is None else XGTeamConfig(epsilon=epsilon)
    return PoissonScorelines(XGTeamModel(config))


def _constant(*, epsilon: float | None = None) -> ScorelineTeamModel:
    from airsenal.prediction.team_models.constant import (  # noqa: PLC0415
        ConstantTeamModel,
    )

    return ConstantTeamModel(epsilon=epsilon)


TEAM_MODELS: dict[str, Callable[..., ScorelineTeamModel]] = {
    "constant": _constant,
    "extended": _extended,
    "neutral": _neutral,
    "random": _random,
    "xg": _xg,
}


def build_team_model(
    name: str = DEFAULT_TEAM_MODEL, epsilon: float | None = None
) -> ScorelineTeamModel:
    """The named team model, with `--epsilon` applied if one was given."""
    return lookup(TEAM_MODELS, name, "team model")(epsilon=epsilon)


__all__ = [
    "DEFAULT_TEAM_MODEL",
    "TEAM_MODELS",
    "build_team_model",
]
