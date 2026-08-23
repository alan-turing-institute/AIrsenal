"""
The available prediction models, by name.

Each table maps a name to something callable with no arguments, typed against
the protocol it has to satisfy - so adding a model is a class plus one line
here, and mypy checks the class fits at the point you add it.
"""

from collections.abc import Callable

from airsenal.core.registry import lookup
from airsenal.prediction.player_models import (
    ConjugatePlayerModel,
    ConstantPlayerModel,
    NumpyroPlayerModel,
)
from airsenal.prediction.protocols import PlayerModel, TeamModel

# Which models a command uses when it is not told. Named here because the CLI, the
# pipeline and the replay driver all had to state them, and `airsenal replay` had
# already drifted to a different set of fit arguments as a result.
DEFAULT_TEAM_MODEL = "extended"
DEFAULT_PLAYER_MODEL = "conjugate"

# Each class defaults its own config, so the class is the factory.
PLAYER_MODELS: dict[str, Callable[[], PlayerModel]] = {
    "conjugate": ConjugatePlayerModel,
    "constant": ConstantPlayerModel,
    "numpyro": NumpyroPlayerModel,
}


# The Dixon-Coles entries are functions rather than the class itself so that bpl,
# and therefore jax, is imported only when one is actually built.
def _extended(*, epsilon: float | None = None) -> TeamModel:
    from airsenal.prediction.team_models.dixon_coles import (  # noqa: PLC0415
        DixonColesTeamModel,
    )

    return DixonColesTeamModel(epsilon=epsilon)


def _neutral(*, epsilon: float | None = None) -> TeamModel:
    from airsenal.prediction.team_models.dixon_coles import (  # noqa: PLC0415
        DixonColesTeamModel,
    )

    return DixonColesTeamModel(neutral=True, epsilon=epsilon)


def _random(*, epsilon: float | None = None) -> TeamModel:
    from airsenal.prediction.team_models.random_model import (  # noqa: PLC0415
        RandomMatchPredictor,
    )

    return RandomMatchPredictor(epsilon=epsilon)


def _constant(*, epsilon: float | None = None) -> TeamModel:
    from airsenal.prediction.team_models.constant import (  # noqa: PLC0415
        ConstantTeamModel,
    )

    return ConstantTeamModel(epsilon=epsilon)


# Entries take an optional keyword-only `epsilon`, the time-weighting decay rate:
# it is the one knob the command line exposes, and a model that does no time
# weighting rejects it rather than ignoring it.
TEAM_MODELS: dict[str, Callable[..., TeamModel]] = {
    "constant": _constant,
    "extended": _extended,
    "neutral": _neutral,
    "random": _random,
}


def build_team_model(
    name: str = DEFAULT_TEAM_MODEL, epsilon: float | None = None
) -> TeamModel:
    """The named team model, with `--epsilon` applied if one was given."""
    return lookup(TEAM_MODELS, name, "team model")(epsilon=epsilon)


def build_player_model(name: str = DEFAULT_PLAYER_MODEL) -> PlayerModel:
    """The named player model, with its own default settings."""
    return lookup(PLAYER_MODELS, name, "player model")()
