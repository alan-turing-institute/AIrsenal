"""
Team models: one module per way of predicting match scorelines.

`TEAM_MODELS` maps a `--team-model` name to a factory. Unlike the other
component tables its factories are not zero-argument: each takes an optional
keyword-only `epsilon`, the time-weighting decay rate.

The default is `xg`, which needs expected goals in the training data - the FPL
API has recorded them since `game.season.FIRST_SEASON_WITH_EXPECTED_GOALS`.
Fitting a season before that means asking for `extended`, which is fitted to
goals.
"""

from collections.abc import Callable

from airsenal.core.lookup import lookup
from airsenal.prediction.protocols import ScorelineTeamModel

# Measured better than `extended` on held-out scorelines in every season with
# expected goals, and on predicted points too. See docs/xg-models.md.
DEFAULT_TEAM_MODEL = "xg"


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
    Expected goals, read as a Conway-Maxwell-Poisson over goal counts.

    `XGTeamModel` predicts only a mean, so the wrapper supplies the distribution
    over counts the points calculation needs; see `DEFAULT_GOAL_DISPERSION`.
    """
    from airsenal.prediction.team_models.scorelines import (  # noqa: PLC0415
        ConwayMaxwellScorelines,
    )
    from airsenal.prediction.team_models.xg import (  # noqa: PLC0415
        XGTeamConfig,
        XGTeamModel,
    )

    config = XGTeamConfig() if epsilon is None else XGTeamConfig(epsilon=epsilon)
    return ConwayMaxwellScorelines(XGTeamModel(config))


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
