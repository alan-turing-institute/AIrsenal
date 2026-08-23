"""
The seam between "pick a whole squad" and "pick it with a genetic algorithm".

The point of the protocol is that both callers - the standalone squad build and
the wildcard/free-hit strategy - go through it, so a different optimizer can be
substituted without editing either. These tests pin that, using a stub optimizer
that no production code knows about.
"""

import pytest

from airsenal.core.registry import ConfigError
from airsenal.optimization.config import GeneticAlgorithmConfig, SquadScoringConfig
from airsenal.optimization.moves import GameweekMove
from airsenal.optimization.protocols import SquadRequest
from airsenal.optimization.squad_optimizers import (
    SQUAD_OPTIMIZERS,
    GeneticSquadOptimizer,
)
from airsenal.optimization.strategies import TRANSFER_STRATEGIES
from airsenal.optimization.strategies.full_squad import FullSquadStrategy


class StubSquadOptimizer:
    """Records the requests it is given and returns a squad it was handed."""

    def __init__(self, squad=None, increments=7):
        self.squad = squad
        self.increments = increments
        self.requests = []
        self.scaled_to = []

    def num_increments(self):
        return self.increments

    def scaled(self, num_iterations):
        self.scaled_to.append(num_iterations)
        return self

    def optimize(self, request):
        self.requests.append(request)
        return self.squad


def test_the_genetic_optimizer_is_registered():
    assert "genetic" in SQUAD_OPTIMIZERS.names()


def test_an_unknown_optimizer_names_the_valid_ones():
    with pytest.raises(ConfigError, match=r"Unknown squad optimizer 'nope'.*genetic"):
        SQUAD_OPTIMIZERS.create("nope")


def test_the_registry_applies_the_config():
    optimizer = SQUAD_OPTIMIZERS.create_with("genetic", {"generations": "12"})
    assert optimizer.num_increments() == 12


def test_increments_are_the_generations_the_search_will_run():
    config = GeneticAlgorithmConfig(generations=30)
    assert GeneticSquadOptimizer(config).num_increments() == 30


def test_scaling_resizes_the_search_without_mutating_the_original():
    original = GeneticSquadOptimizer(GeneticAlgorithmConfig())
    scaled = original.scaled(20)

    assert scaled.num_increments() == 20
    assert scaled.config.population_size == 20
    assert original.num_increments() == 100


def test_scaling_keeps_the_settings_it_does_not_size():
    scaled = GeneticSquadOptimizer(
        GeneticAlgorithmConfig(tournament_size=5, random_state=1)
    ).scaled(20)

    assert scaled.config.tournament_size == 5
    assert scaled.config.random_state == 1


def test_a_request_defaults_to_todays_scoring():
    """The seam must not quietly change how a squad is scored."""
    request = SquadRequest(gameweeks=[1], tag="t", season="2526")

    assert request.budget == 1000
    assert request.scoring.dummy_sub_cost == 45
    assert request.scoring.sub_weights.as_dict() == {
        "GK": 0.03,
        "Outfield": (0.65, 0.3, 0.1),
    }


def test_progress_is_only_reported_when_something_is_watching():
    # SquadOpt.optimize branches on whether it was given a reporter, so a request
    # with no progress must not manufacture one.
    SquadRequest(gameweeks=[1], tag="t", season="2526").advance_progress(1.0)

    seen = []
    watched = SquadRequest(gameweeks=[1], tag="t", season="2526", progress=seen.append)
    watched.advance_progress(3.5)
    assert seen == [3.5]


def test_the_registered_full_squad_strategy_is_genetic():
    strategy = TRANSFER_STRATEGIES.create("full_squad")
    assert isinstance(strategy.optimizer, GeneticSquadOptimizer)


def test_full_squad_options_reach_the_optimizer():
    """
    The asymmetry this seam exists to remove.

    `optimize squad` could tune the genetic algorithm, but the wildcard and free
    hit path built its strategy with defaults and no caller could reach it.
    """
    strategy = TRANSFER_STRATEGIES.create_with("full_squad", {"generations": "9"})
    assert strategy.optimizer.num_increments() == 9


def test_full_squad_sizes_its_cost_from_the_optimizer_it_was_given():
    optimizer = StubSquadOptimizer(increments=42)
    strategy = FullSquadStrategy(optimizer)

    assert strategy.num_increments(GameweekMove(), num_iterations=15) == 42
    assert optimizer.scaled_to == [15]


def test_scoring_config_carries_the_budget_the_optimizer_must_respect():
    scoring = SquadScoringConfig(budget=825)
    request = SquadRequest(gameweeks=[1], tag="t", season="2526", scoring=scoring)
    assert request.budget == 825
