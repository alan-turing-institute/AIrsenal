"""Every swappable kind is built the same way: `build_<kind>(name, **overrides)`."""

import pytest

from airsenal.core.lookup import ConfigError
from airsenal.optimization.squad_optimizers import (
    DEFAULT_SQUAD_OPTIMIZER,
    GeneticAlgorithmConfig,
    build_squad_optimizer,
)
from airsenal.optimization.transfer_optimizers import (
    DEFAULT_TRANSFER_OPTIMIZER,
    TreeSearchConfig,
    build_transfer_optimizer,
)
from airsenal.prediction.player_models import build_player_model
from airsenal.prediction.team_models import build_team_model

BUILDERS = {
    "player model": build_player_model,
    "team model": build_team_model,
    "squad optimizer": build_squad_optimizer,
    "transfer optimizer": build_transfer_optimizer,
}


@pytest.mark.parametrize("kind", BUILDERS)
def test_every_kind_builds_from_no_arguments_at_all(kind):
    """Each builder defaults to its kind's default name."""
    assert BUILDERS[kind]() is not None


@pytest.mark.parametrize("kind", BUILDERS)
def test_an_unknown_name_lists_the_known_ones(kind):
    with pytest.raises(ConfigError) as excinfo:
        BUILDERS[kind]("nope")
    assert "nope" in str(excinfo.value)
    assert "Choose from" in str(excinfo.value)


def test_the_tree_search_flags_reach_the_tree_search():
    optimizer = build_transfer_optimizer(
        DEFAULT_TRANSFER_OPTIMIZER, num_thread=3, num_iterations=7, profile=True
    )
    assert optimizer.config.num_thread == 3
    assert optimizer.config.num_iterations == 7
    assert optimizer.config.profile is True


def test_an_unset_tree_search_flag_leaves_the_default_alone():
    optimizer = build_transfer_optimizer(DEFAULT_TRANSFER_OPTIMIZER, num_thread=3)
    assert optimizer.config.num_iterations == TreeSearchConfig().num_iterations


def test_the_genetic_algorithm_flags_reach_the_genetic_algorithm():
    optimizer = build_squad_optimizer(
        DEFAULT_SQUAD_OPTIMIZER, num_generations=11, population_size=13
    )
    assert optimizer.config.generations == 11
    assert optimizer.config.population_size == 13


def test_an_unset_genetic_algorithm_flag_leaves_the_default_alone():
    optimizer = build_squad_optimizer(DEFAULT_SQUAD_OPTIMIZER, num_generations=11)
    assert optimizer.config.population_size == GeneticAlgorithmConfig().population_size
