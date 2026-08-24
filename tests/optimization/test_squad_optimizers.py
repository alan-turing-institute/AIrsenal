"""
The seam between "pick a whole squad" and "pick it with a genetic algorithm".

The point of the protocol is that both callers - the standalone squad build and
the wildcard/free-hit strategy - go through it, so a different optimizer can be
substituted without editing either. These tests pin that, using a stub optimizer
that no production code knows about.
"""

import pytest

from airsenal.core.registry import ConfigError, lookup
from airsenal.optimization.config import GeneticAlgorithmConfig, SquadScoringConfig
from airsenal.optimization.moves import GameweekMove
from airsenal.optimization.protocols import SquadRequest
from airsenal.optimization.squad_optimizers import (
    SQUAD_OPTIMIZERS,
    GeneticSquadOptimizer,
    genetic_optimizer,
)
from airsenal.optimization.strategies import TRANSFER_STRATEGIES
from airsenal.optimization.strategies.full_squad import FullSquadStrategy


class StubSquadOptimizer:
    """Records the requests it is given and returns a squad it was handed."""

    def __init__(self, squad=None, increments=7):
        self.squad = squad
        self.increments = increments
        self.requests = []

    def num_increments(self):
        return self.increments

    def optimize(self, request):
        self.requests.append(request)
        return self.squad


def test_the_genetic_optimizer_is_registered():
    assert "genetic" in SQUAD_OPTIMIZERS


def test_an_unknown_optimizer_names_the_valid_ones():
    with pytest.raises(ConfigError, match=r"Unknown squad optimizer 'nope'.*genetic"):
        lookup(SQUAD_OPTIMIZERS, "nope", "squad optimizer")


def test_the_genetic_optimizer_defaults_its_own_config():
    """Every table entry has to be constructible with no arguments."""
    assert SQUAD_OPTIMIZERS["genetic"]().config == GeneticAlgorithmConfig()


def test_increments_are_the_generations_the_search_will_run():
    config = GeneticAlgorithmConfig(generations=30)
    assert GeneticSquadOptimizer(config).num_increments() == 30


def test_scaling_resizes_the_search_without_mutating_the_original():
    """
    Sizing is a property of the config, not of the optimizer protocol: the
    wildcard path has one --num-iterations knob and builds a sized optimizer
    from it, while the standalone squad build keeps its full config.
    """
    base = GeneticAlgorithmConfig()
    scaled = genetic_optimizer(20)

    assert scaled.num_increments() == 20
    assert scaled.config.population_size == 20
    assert base.generations == 100


def test_scaling_keeps_the_settings_it_does_not_size():
    scaled = GeneticSquadOptimizer(
        GeneticAlgorithmConfig(tournament_size=5, random_state=1).scaled(20)
    )

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
    strategy = TRANSFER_STRATEGIES["full_squad"]()
    assert isinstance(strategy.make_optimizer(10), GeneticSquadOptimizer)


def test_full_squad_sizes_its_optimizer_from_the_iteration_count():
    """
    The asymmetry this seam exists to remove.

    `optimize squad` could tune the genetic algorithm, but the wildcard and free
    hit path built its strategy with defaults and no caller could reach it.
    """
    strategy = TRANSFER_STRATEGIES["full_squad"]()
    assert strategy.num_increments(GameweekMove(), num_iterations=9) == 9


def test_full_squad_sizes_its_cost_from_the_optimizer_it_was_given():
    sized_to = []

    def make(num_iterations):
        sized_to.append(num_iterations)
        return StubSquadOptimizer(increments=42)

    strategy = FullSquadStrategy(make)

    assert strategy.num_increments(GameweekMove(), num_iterations=15) == 42
    assert sized_to == [15]


def test_scoring_config_carries_the_budget_the_optimizer_must_respect():
    scoring = SquadScoringConfig(budget=825)
    request = SquadRequest(gameweeks=[1], tag="t", season="2526", scoring=scoring)
    assert request.budget == 825
