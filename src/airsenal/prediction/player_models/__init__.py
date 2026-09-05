"""
Player models: one module per way of predicting how a team's goals are shared out.

`PLAYER_MODELS` maps a `--player-model` name to a zero-argument factory.
"""

from collections.abc import Callable

from airsenal.core.lookup import lookup
from airsenal.prediction.player_models.conjugate import (
    ConjugatePlayerConfig,
    ConjugatePlayerModel,
)
from airsenal.prediction.player_models.constant import (
    ConstantPlayerConfig,
    ConstantPlayerModel,
)
from airsenal.prediction.player_models.mcmc import (
    NumpyroPlayerConfig,
    NumpyroPlayerModel,
)
from airsenal.prediction.protocols import PlayerModel

DEFAULT_PLAYER_MODEL = "conjugate"
PLAYER_MODELS: dict[str, Callable[[], PlayerModel]] = {
    "conjugate": ConjugatePlayerModel,
    "constant": ConstantPlayerModel,
    "numpyro": NumpyroPlayerModel,
}


def build_player_model(name: str = DEFAULT_PLAYER_MODEL) -> PlayerModel:
    """The named player model, with its own default settings."""
    return lookup(PLAYER_MODELS, name, "player model")()


__all__ = [
    "DEFAULT_PLAYER_MODEL",
    "PLAYER_MODELS",
    "ConjugatePlayerConfig",
    "ConjugatePlayerModel",
    "ConstantPlayerConfig",
    "ConstantPlayerModel",
    "NumpyroPlayerConfig",
    "NumpyroPlayerModel",
    "build_player_model",
]
