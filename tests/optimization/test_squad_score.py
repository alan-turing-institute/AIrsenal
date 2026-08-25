"""How a squad is scored, and the settings that say how to weigh it."""

import pytest

from airsenal.optimization.squad_score import (
    DEFAULT_DISCOUNT,
    SquadScoringConfig,
    get_discount_factor,
)
from airsenal.squad.squad import SubWeights


def test_sub_weights_defaults():
    assert SubWeights().gk == 0.03
    assert SubWeights().outfield == (0.65, 0.3, 0.1)


def test_no_subs_ignores_the_bench_entirely():
    assert SubWeights.none().gk == 0.0
    assert SubWeights.none().outfield == (0.0, 0.0, 0.0)


def test_a_bench_boost_counts_every_substitute_in_full():
    """What `Squad.total_points_for_subs` falls back to when given nothing."""
    assert SubWeights.full().gk == 1.0
    assert SubWeights.full().outfield == (1.0, 1.0, 1.0)


def test_sub_weights_are_immutable():
    with pytest.raises(AttributeError):
        SubWeights().gk = 0.5


def test_squad_scoring_config_defaults():
    config = SquadScoringConfig()
    assert config.sub_weights == SubWeights()
    assert config.dummy_sub_cost == 45
    assert config.budget == 1000


def test_each_squad_scoring_config_gets_its_own_sub_weights():
    """A shared mutable default would let one config's edits leak into another."""
    assert SquadScoringConfig().sub_weights is not SquadScoringConfig().sub_weights


def test_a_gameweek_further_out_counts_for_less():
    assert get_discount_factor(1, 1) == 1
    assert get_discount_factor(1, 2) == DEFAULT_DISCOUNT
    assert get_discount_factor(1, 4) == DEFAULT_DISCOUNT**3
