"""
Point components: one module per part of an FPL score.

`POINT_COMPONENTS` maps a name to a zero-argument factory, like the other
component tables. Three of them are rules rather than models - appearance,
attacking and defending points fall out of the team and player models and the
scoring rules - and four are small empirical-Bayes averages over a player's own
history, shrunk towards the average for their position.

`PointsConfig` decides which of them a run uses. The four that can be turned off
are the fitted ones: turning one off skips fitting it and leaves that part out
of the total.
"""

from collections.abc import Callable
from dataclasses import dataclass

from airsenal.core.lookup import lookup
from airsenal.prediction.point_components.actual import (
    RESIDUAL,
    actual_component_points,
)
from airsenal.prediction.point_components.appearance import AppearanceComponent
from airsenal.prediction.point_components.attacking import (
    AttackingComponent,
    get_attacking_points,
)
from airsenal.prediction.point_components.bonus import BonusComponent, fit_bonus_points
from airsenal.prediction.point_components.cards import CardComponent, fit_card_points
from airsenal.prediction.point_components.def_con import DefConComponent, fit_def_con
from airsenal.prediction.point_components.defending import (
    DefendingComponent,
    get_defending_points,
)
from airsenal.prediction.point_components.empirical_bayes import mean_group_prior
from airsenal.prediction.point_components.saves import SaveComponent, fit_save_points
from airsenal.prediction.protocols import PointComponent

POINT_COMPONENTS: dict[str, Callable[[], PointComponent]] = {
    "appearance": AppearanceComponent,
    "attacking": AttackingComponent,
    "defending": DefendingComponent,
    "bonus": BonusComponent,
    "cards": CardComponent,
    "saves": SaveComponent,
    "def_con": DefConComponent,
}

# Every score has these, whatever a run is configured to predict: they are the
# scoring rules applied to the team and player models, with nothing fitted.
ALWAYS_PREDICTED = ("appearance", "attacking", "defending")


def build_point_component(name: str) -> PointComponent:
    """The named component, with its own default settings."""
    return lookup(POINT_COMPONENTS, name, "point component")()


@dataclass(frozen=True)
class PointsConfig:
    """
    Which components of an FPL score to predict.

    Each flag turns off one of the fitted components, which skips fitting it and
    leaves that component out of the total. The three that are always predicted
    have no flag: without them there is no score to speak of.
    """

    bonus: bool = True
    cards: bool = True
    saves: bool = True
    def_con: bool = True

    def component_names(self) -> list[str]:
        """The components a run with this configuration predicts, in table order."""
        optional = {
            "bonus": self.bonus,
            "cards": self.cards,
            "saves": self.saves,
            "def_con": self.def_con,
        }
        return [
            name
            for name in POINT_COMPONENTS
            if name in ALWAYS_PREDICTED or optional[name]
        ]

    def components(self) -> list[PointComponent]:
        """Those components, built but not yet fitted."""
        return [build_point_component(name) for name in self.component_names()]


__all__ = [
    "ALWAYS_PREDICTED",
    "POINT_COMPONENTS",
    "RESIDUAL",
    "AppearanceComponent",
    "AttackingComponent",
    "BonusComponent",
    "CardComponent",
    "DefConComponent",
    "DefendingComponent",
    "PointsConfig",
    "SaveComponent",
    "actual_component_points",
    "build_point_component",
    "fit_bonus_points",
    "fit_card_points",
    "fit_def_con",
    "fit_save_points",
    "get_attacking_points",
    "get_defending_points",
    "mean_group_prior",
]
