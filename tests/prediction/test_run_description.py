"""
What a run records about the components that produced it.

`airsenal replay` writes this into its JSON, and it is the only thing telling
two replays apart: the points, the hits and the error mean nothing without it.
A component that wraps another has to say so, or the record names a wrapper
every candidate shares and loses the model being compared.
"""

from airsenal.prediction.player_models import ConstantPlayerModel
from airsenal.prediction.points_models import ComponentPointsModel
from airsenal.prediction.points_models.component import describe_component
from airsenal.prediction.team_models import build_team_model
from airsenal.prediction.team_models.constant import ConstantTeamModel


def test_a_component_is_named_by_its_class():
    """The plain case, and what a component with nothing to add still gets."""
    assert describe_component(ConstantTeamModel()) == "ConstantTeamModel"
    assert describe_component(ConstantPlayerModel()) == "ConstantPlayerModel"


def test_a_wrapped_model_names_both():
    """
    The default team model is a wrapper around `XGTeamModel`.

    Recording only `ConwayMaxwellScorelines` would name the distribution put
    over the goal counts and not the model that produced the mean, which is the
    part a comparison is usually about.
    """
    assert (
        describe_component(build_team_model("xg"))
        == "ConwayMaxwellScorelines(XGTeamModel)"
    )


def test_the_points_model_records_the_model_inside_its_wrapper():
    """
    End to end: what a replay's `config` block holds for a default run.

    Asserted on the value rather than on `xg` being the default, so this keeps
    saying something if the default moves.
    """
    described = ComponentPointsModel().describe()
    assert described["points_model"] == "ComponentPointsModel"
    assert described["team_model"] == "ConwayMaxwellScorelines(XGTeamModel)"
    assert described["player_model"] == "XGPlayerModel"


def test_a_named_team_model_that_does_not_wrap_is_named_alone():
    """A wrapper is not added to a model that already has scorelines."""
    assert describe_component(build_team_model("constant")) == "ConstantTeamModel"
