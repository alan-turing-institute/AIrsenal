"""
Team models: one module per way of predicting match scorelines.

The table below is how a name on the command line reaches an implementation.
Adding a model is a class satisfying `TeamModel`, plus one line here - the table
is typed against the protocol, so mypy checks the class fits where you add it.

Entries take an optional keyword-only `epsilon`, the time-weighting decay rate:
it is the one hyperparameter the command line exposes, and a model that does no
time weighting rejects it rather than ignoring it. That makes this the one table
whose factories are not zero-argument.
"""

from collections.abc import Callable

from airsenal.core.lookup import lookup
from airsenal.prediction.protocols import TeamModel

# Which model a command uses when it is not told. Named here because the CLI, the
# pipeline and the replay driver all had to state it, and `airsenal replay` had
# already drifted to a different set of fit arguments as a result.
DEFAULT_TEAM_MODEL = "extended"


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
        RandomTeamModel,
    )

    return RandomTeamModel(epsilon=epsilon)


def _constant(*, epsilon: float | None = None) -> TeamModel:
    from airsenal.prediction.team_models.constant import (  # noqa: PLC0415
        ConstantTeamModel,
    )

    return ConstantTeamModel(epsilon=epsilon)


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


__all__ = [
    "DEFAULT_TEAM_MODEL",
    "TEAM_MODELS",
    "build_team_model",
]
