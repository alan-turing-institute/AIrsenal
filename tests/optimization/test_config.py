"""Optimisation configuration."""

from dataclasses import replace

import pytest

from airsenal.optimization.config import (
    DEFAULT_SUB_WEIGHTS,
    GeneticAlgorithmConfig,
    SquadScoringConfig,
    SubWeights,
)


def test_default_sub_weights_come_from_one_definition():
    """
    The squad builder used to hard-code a different set from DEFAULT_SUB_WEIGHTS, so
    `optimize squad` and `optimize transfers` scored benches differently.
    """
    assert SubWeights().as_dict() == DEFAULT_SUB_WEIGHTS


def test_sub_weights_shape_matches_what_the_scoring_code_expects():
    assert SubWeights().as_dict() == {"GK": 0.03, "Outfield": (0.65, 0.3, 0.1)}


def test_no_subs_ignores_the_bench_entirely():
    assert SubWeights.none().as_dict() == {"GK": 0.0, "Outfield": (0.0, 0.0, 0.0)}


def test_sub_weights_are_immutable():
    with pytest.raises(AttributeError):
        SubWeights().gk = 0.5


def test_genetic_algorithm_defaults():
    config = GeneticAlgorithmConfig()
    assert config.population_size == 100
    assert config.generations == 100
    assert config.crossover_prob == 0.7
    assert config.mutation_prob == 0.3
    assert config.tournament_size == 3
    assert config.random_state is None


def test_scaled_sets_population_and_generations_together():
    """
    The wildcard and free-hit paths have a single num_iterations knob driving both.
    Questionable, but explicit here rather than buried in make_best_transfers.
    """
    scaled = GeneticAlgorithmConfig().scaled(7)
    assert scaled.population_size == 7
    assert scaled.generations == 7


def test_scaled_leaves_everything_else_alone():
    base = GeneticAlgorithmConfig(tournament_size=5, random_state=1)
    scaled = base.scaled(9)
    assert scaled == replace(base, population_size=9, generations=9)


def test_squad_scoring_config_defaults():
    config = SquadScoringConfig()
    assert config.sub_weights == SubWeights()
    assert config.dummy_sub_cost == 45
    assert config.budget == 1000


def test_each_squad_scoring_config_gets_its_own_sub_weights():
    """A shared mutable default would let one config's edits leak into another."""
    assert SquadScoringConfig().sub_weights is not SquadScoringConfig().sub_weights
