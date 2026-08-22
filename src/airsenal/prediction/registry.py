"""
The available prediction models, by name.

Replaces two copies of the same if/elif chain - one in
fill_predictedscore_table.run_prediction, one in parse_team_model_from_str - and the
--sampling boolean, which could only ever express two of the player models.
"""

from typing import Any

from airsenal.core.registry import Registry
from airsenal.prediction.config import (
    ConjugatePlayerConfig,
    DixonColesConfig,
    NumpyroPlayerConfig,
    RandomTeamModelConfig,
)

# The team-model factories ignore their config: bpl takes epsilon and
# rescale_weights when fitting, not when constructing, so the caller reads them
# off the config via fit_args(). Use TEAM_MODELS.build() to get both.
PLAYER_MODELS: Registry[Any] = Registry("player model")
TEAM_MODELS: Registry[Any] = Registry("team model")


@PLAYER_MODELS.register("conjugate", ConjugatePlayerConfig)
def _conjugate(config: ConjugatePlayerConfig) -> Any:
    from airsenal.prediction.player_models import ConjugatePlayerModel  # noqa: PLC0415

    return ConjugatePlayerModel(config)


@PLAYER_MODELS.register("numpyro", NumpyroPlayerConfig)
def _numpyro(config: NumpyroPlayerConfig) -> Any:
    from airsenal.prediction.player_models import NumpyroPlayerModel  # noqa: PLC0415

    return NumpyroPlayerModel(config)


@TEAM_MODELS.register("extended", DixonColesConfig)
def _extended(_config: DixonColesConfig) -> Any:
    # bpl is imported lazily: it pulls in jax, which is slow to import and not
    # needed by commands that never fit a team model.
    from bpl import ExtendedDixonColesMatchPredictor  # noqa: PLC0415

    return ExtendedDixonColesMatchPredictor()


@TEAM_MODELS.register("neutral", DixonColesConfig)
def _neutral(_config: DixonColesConfig) -> Any:
    from bpl import NeutralDixonColesMatchPredictor  # noqa: PLC0415

    return NeutralDixonColesMatchPredictor()


@TEAM_MODELS.register("random", RandomTeamModelConfig)
def _random(_config: RandomTeamModelConfig) -> Any:
    from airsenal.prediction.team_models.random_model import (  # noqa: PLC0415
        RandomMatchPredictor,
    )

    return RandomMatchPredictor()
