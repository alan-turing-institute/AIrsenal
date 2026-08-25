"""
Player models: one module per way of predicting how a team's goals are shared out.

`PLAYER_MODELS` maps a `--player-model` name to a zero-argument factory. Being in
it is only how a *name* reaches an implementation - `AIrsenalPipeline` takes
objects, so a model defined in a notebook can be dropped straight in.
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

# Which model a command uses when it is not told. Named here because the CLI, the
# pipeline and the replay driver all had to state it, and `airsenal replay` had
# already drifted to a different set of fit arguments as a result.
DEFAULT_PLAYER_MODEL = "conjugate"

# Each class defaults its own config, so the class is the factory.
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
