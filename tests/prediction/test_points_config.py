"""
Which components of a score to predict, and that the answer reaches the code.

A points flag passes through two layers before it reaches the function that
branches on it, and a flag that is offered, accepted and then dropped fails
silently. tests/test_cli.py checks the flags are still offered; these check they
still arrive.
"""

import pytest

from airsenal.pipeline import AIrsenalPipeline
from airsenal.prediction.points import PointsConfig
from airsenal.prediction.run import make_predictedscore_table


def test_everything_is_predicted_by_default():
    """A run that says nothing must keep predicting every component."""
    config = PointsConfig()
    assert (config.bonus, config.cards, config.saves, config.def_con) == (
        True,
        True,
        True,
        True,
    )


def test_the_config_is_immutable():
    with pytest.raises(AttributeError):
        PointsConfig().bonus = False


@pytest.mark.parametrize("component", ["bonus", "cards", "saves", "def_con"])
def test_each_component_reaches_the_prediction(monkeypatch, component):
    """
    Every flag has to survive the hop from the pipeline to the table filler.

    Patched at the seam so nothing is actually fitted: what is under test is the
    plumbing, which is what broke last time.
    """
    seen = {}

    def record(**kwargs):
        seen.update(kwargs)
        return "tag"

    monkeypatch.setattr("airsenal.pipeline.run.make_predictedscore_table", record)

    pipeline = AIrsenalPipeline(points=PointsConfig(**{component: False}))
    pipeline.predict([1, 2], dbsession=None)

    assert getattr(seen["points"], component) is False
    # and the others are untouched
    assert all(
        getattr(seen["points"], other) is True
        for other in ("bonus", "cards", "saves", "def_con")
        if other != component
    )


def test_a_window_must_be_given():
    """
    `make_predictedscore_table` has no default window: resolving one is the
    pipeline's job, and a second hardcoded default here could disagree with it.
    """
    with pytest.raises(TypeError):
        make_predictedscore_table()  # type: ignore[call-arg]
