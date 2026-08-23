"""
The available prediction models, by name.

Replaces two copies of the same if/elif chain - one in
fill_predictedscore_table.run_prediction, one in parse_team_model_from_str - and the
--sampling boolean, which could only ever express two of the player models.
"""

from collections.abc import Mapping
from typing import Any

from airsenal.core.registry import Registry
from airsenal.prediction.config import (
    ConjugatePlayerConfig,
    ConstantPlayerConfig,
    ConstantTeamModelConfig,
    DixonColesConfig,
    NumpyroPlayerConfig,
    RandomTeamModelConfig,
)
from airsenal.prediction.protocols import ConfiguredTeamModel

# Which models a command uses when it is not told. Named here because the CLI, the
# pipeline and the replay driver all had to state them, and `airsenal replay` had
# already drifted to a different set of fit arguments as a result.
DEFAULT_TEAM_MODEL = "extended"
DEFAULT_PLAYER_MODEL = "conjugate"

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


@PLAYER_MODELS.register("constant", ConstantPlayerConfig)
def _constant_player(config: ConstantPlayerConfig) -> Any:
    from airsenal.prediction.player_models import ConstantPlayerModel  # noqa: PLC0415

    return ConstantPlayerModel(config)


@TEAM_MODELS.register("constant", ConstantTeamModelConfig)
def _constant_team(config: ConstantTeamModelConfig) -> Any:
    from airsenal.prediction.team_models.constant import (  # noqa: PLC0415
        ConstantTeamModel,
    )

    return ConstantTeamModel(config.max_goals)


def build_team_model(
    name: str = DEFAULT_TEAM_MODEL,
    options: Mapping[str, str] | None = None,
    epsilon: float | None = None,
) -> ConfiguredTeamModel:
    """
    The team model named on the command line, with the settings it fits with.

    The single place a team model is chosen. It used to be three: `airsenal run`,
    `airsenal predict` and `airsenal replay` each merged --epsilon into the option
    dict themselves, and the first two then had to remember `config.fit_args()`
    while the third did not.

    Parameters
    ----------
    name : str
        A name registered in `TEAM_MODELS`.
    options : Mapping[str, str], optional
        `key=value` overrides, as given by repeated `--set-team`.
    epsilon : float, optional
        The time-weighting decay rate. First-class because it is the knob people
        actually tune; everything else goes through `options`. Only forwarded when
        given, so a model without an epsilon is not an error.

    Returns
    -------
    ConfiguredTeamModel
        The model and the arguments to fit it with.
    """
    merged = dict(options or {})
    if epsilon is not None:
        # --set-team wins over --epsilon, which is the order these have always
        # resolved in. Note `optimize squad` resolves --set-ga the other way round;
        # they should agree, but changing either is a behaviour change of its own.
        merged = {"epsilon": str(epsilon), **merged}
    model, config = TEAM_MODELS.build(name, merged)
    return ConfiguredTeamModel(model, config.fit_args())
