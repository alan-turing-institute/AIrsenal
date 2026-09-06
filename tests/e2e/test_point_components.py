"""
Every point component, against a real database, and one nobody registered.

Parametrized over `POINT_COMPONENTS`, so adding a component to the table gets it
fitted here for free. The last two tests are the point of the phase: a component
written outside the package reaches a real prediction run, and turning one off
changes the answer.
"""

import math

import pytest
from sqlalchemy import select

from airsenal.db.models import PlayerPrediction
from airsenal.game.enums import Position
from airsenal.prediction.player_models import build_player_model
from airsenal.prediction.point_components import (
    POINT_COMPONENTS,
    PointsConfig,
    build_point_component,
)
from airsenal.prediction.points_models import ComponentPointsModel
from airsenal.prediction.protocols import ComponentRequest
from airsenal.prediction.run import make_predictedscore_table
from airsenal.prediction.team_models import build_team_model
from tests.e2e.conftest import FUTURE_GAMEWEEKS, PAST_SEASONS, SEASON

FIT_GAMEWEEK = 8
FIT_SEASON = PAST_SEASONS[-1]


def a_request(position=Position.MID, minutes=90.0):
    return ComponentRequest(
        player_id=1,
        position=position,
        minutes=minutes,
        team_score_probability={0: 0.4, 1: 0.4, 2: 0.2},
        team_concede_probability={0: 0.5, 1: 0.3, 2: 0.2},
        prob_score=0.1,
        prob_assist=0.1,
    )


@pytest.mark.parametrize("name", sorted(POINT_COMPONENTS))
def test_every_component_fits_and_answers(pipeline_db, name):
    component = build_point_component(name)
    fitted = component.fit(FIT_GAMEWEEK, FIT_SEASON, pipeline_db)
    assert fitted is component, "fit returns self, like the other models"
    assert component.name == name
    for position in Position:
        points = component.expected_points(a_request(position=position))
        assert math.isfinite(points)


@pytest.mark.parametrize("name", sorted(POINT_COMPONENTS))
def test_no_component_pays_a_player_who_did_not_appear(pipeline_db, name):
    """
    Zero minutes is zero points, whatever the component.

    The points calculation relies on this: a player with no recent minutes is
    predicted zero by short-circuiting, and that has to agree with what the
    components would have said.
    """
    component = build_point_component(name).fit(FIT_GAMEWEEK, FIT_SEASON, pipeline_db)
    for position in Position:
        assert (
            component.expected_points(a_request(position=position, minutes=0.0)) == 0.0
        )


class DoubleAppearance:
    """A component no table knows about: two points for turning up at all."""

    name = "double_appearance"

    def __init__(self):
        self.fitted_at = None

    def fit(self, gameweek, season, dbsession):
        del dbsession
        self.fitted_at = (gameweek, season)
        return self

    def expected_points(self, request):
        return 2.0 if request.minutes > 0 else 0.0


def test_a_component_no_table_knows_about_can_be_predicted_with(pipeline_db):
    """
    What makes this a pluggable kind rather than a fixed list.

    The same promise `tests/e2e/test_pipeline_composition.py` makes for the
    optimizers: a class written in a notebook drops straight in.
    """
    component = DoubleAppearance()
    tag = make_predictedscore_table(
        gameweeks=FUTURE_GAMEWEEKS[:1],
        season=SEASON,
        points_model=ComponentPointsModel(
            team_model=build_team_model("constant"),
            player_model=build_player_model("constant"),
            components=[component],
        ),
        dbsession=pipeline_db,
    )
    assert component.fitted_at == (FUTURE_GAMEWEEKS[0], SEASON)
    points = pipeline_db.scalars(
        select(PlayerPrediction.predicted_points).where(PlayerPrediction.tag == tag)
    ).all()
    assert points
    # the only component in the run, so every player who plays is worth two
    assert set(points) <= {0.0, 2.0}
    assert 2.0 in set(points)


def test_turning_a_component_off_leaves_it_out_of_the_total(pipeline_db):
    """`PointsConfig` still decides what a run predicts, now by naming components."""
    assert PointsConfig().component_names() == list(POINT_COMPONENTS)
    without = PointsConfig(bonus=False).component_names()
    assert "bonus" not in without
    assert "attacking" in without

    tags = {}
    for label, config in (
        ("all", PointsConfig()),
        ("no_bonus", PointsConfig(bonus=False)),
    ):
        tags[label] = make_predictedscore_table(
            gameweeks=FUTURE_GAMEWEEKS[:1],
            season=SEASON,
            points_model=ComponentPointsModel(
                team_model=build_team_model("constant"),
                player_model=build_player_model("constant"),
                points=config,
            ),
            dbsession=pipeline_db,
        )
    totals = {
        label: sum(
            pipeline_db.scalars(
                select(PlayerPrediction.predicted_points).where(
                    PlayerPrediction.tag == tag
                )
            ).all()
        )
        for label, tag in tags.items()
    }
    # bonus points are never negative, so dropping them can only lose points
    assert totals["no_bonus"] < totals["all"]
