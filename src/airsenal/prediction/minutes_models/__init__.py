"""
Minutes models: one module per way of predicting how long a player will play.

`MINUTES_MODELS` maps a `--minutes-model` name to a zero-argument factory.
"""

from collections.abc import Callable

from airsenal.core.lookup import lookup
from airsenal.prediction.minutes_models.recent import (
    RecentMinutesConfig,
    RecentMinutesModel,
)
from airsenal.prediction.protocols import MinutesModel

DEFAULT_MINUTES_MODEL = "recent"
MINUTES_MODELS: dict[str, Callable[[], MinutesModel]] = {
    "recent": RecentMinutesModel,
}


def build_minutes_model(name: str = DEFAULT_MINUTES_MODEL) -> MinutesModel:
    """The named minutes model, with its own default settings."""
    return lookup(MINUTES_MODELS, name, "minutes model")()


__all__ = [
    "DEFAULT_MINUTES_MODEL",
    "MINUTES_MODELS",
    "RecentMinutesConfig",
    "RecentMinutesModel",
    "build_minutes_model",
]
