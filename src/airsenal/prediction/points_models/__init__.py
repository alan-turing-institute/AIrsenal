"""
Points models: one module per way of predicting what a player will score.

`POINTS_MODELS` maps a `--points-model` name to a zero-argument factory. This is
the seam the rest of prediction is behind: a team model, a player model, a
minutes model and the components are how `ComponentPointsModel` happens to
work, not something every points model has to have.
"""

from collections.abc import Callable

from airsenal.core.lookup import ConfigError, lookup
from airsenal.prediction.minutes_models import (
    DEFAULT_MINUTES_MODEL,
    build_minutes_model,
)
from airsenal.prediction.player_models import DEFAULT_PLAYER_MODEL, build_player_model
from airsenal.prediction.point_components import PointsConfig
from airsenal.prediction.points_models.component import ComponentPointsModel
from airsenal.prediction.protocols import PointsModel
from airsenal.prediction.team_models import DEFAULT_TEAM_MODEL, build_team_model

DEFAULT_POINTS_MODEL = "component"
POINTS_MODELS: dict[str, Callable[[], PointsModel]] = {
    "component": ComponentPointsModel,
}


def build_points_model(
    name: str = DEFAULT_POINTS_MODEL,
    team_model: str = DEFAULT_TEAM_MODEL,
    player_model: str = DEFAULT_PLAYER_MODEL,
    minutes_model: str = DEFAULT_MINUTES_MODEL,
    epsilon: float | None = None,
    points: PointsConfig | None = None,
) -> PointsModel:
    """
    The named points model, built from the flags that describe its parts.

    A team, player and minutes model are what `ComponentPointsModel` is made of,
    not what every points model has: naming a different one *and* a part of the
    component model is refused rather than half-honoured, the same rule as
    `--epsilon` on a team model that does no time weighting.
    """
    if name == DEFAULT_POINTS_MODEL:
        return ComponentPointsModel(
            team_model=build_team_model(team_model, epsilon),
            player_model=build_player_model(player_model),
            minutes_model=build_minutes_model(minutes_model),
            points=points,
        )

    given = [
        flag
        for flag, value, default in (
            ("--team-model", team_model, DEFAULT_TEAM_MODEL),
            ("--player-model", player_model, DEFAULT_PLAYER_MODEL),
            ("--minutes-model", minutes_model, DEFAULT_MINUTES_MODEL),
            ("--epsilon", epsilon, None),
            ("the points components", points, None),
        )
        if value != default
    ]
    if given:
        msg = (
            f"the {name} points model is not made of components, so it has no "
            f"{', '.join(given)}"
        )
        raise ConfigError(msg)
    return lookup(POINTS_MODELS, name, "points model")()


__all__ = [
    "DEFAULT_POINTS_MODEL",
    "POINTS_MODELS",
    "ComponentPointsModel",
    "build_points_model",
]
