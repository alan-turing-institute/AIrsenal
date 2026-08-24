"""
The seam between "pick a whole squad" and "pick it with a genetic algorithm".

The point of the protocol is that both callers - the standalone squad build and
the wildcard/free-hit strategy - go through it, so a different optimizer can be
substituted without editing either. These tests pin that, using a stub optimizer
that no production code knows about.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from airsenal.core.enums import Chip
from airsenal.core.lookup import ConfigError, lookup
from airsenal.optimization.moves import GameweekMove
from airsenal.optimization.protocols import SquadRequest, TransferRequest
from airsenal.optimization.squad_optimizers import (
    SQUAD_OPTIMIZERS,
    GeneticAlgorithmConfig,
    GeneticSquadOptimizer,
)
from airsenal.optimization.squad_score import SquadScoringConfig
from airsenal.optimization.strategies import TRANSFER_STRATEGIES
from airsenal.optimization.strategies.full_squad import FullSquadStrategy


class StubSquad:
    """Just enough of a Squad for the rebuild strategy to describe it."""

    def __init__(self, player_ids=()):
        self.players = [SimpleNamespace(player_id=i) for i in player_ids]

    def sale_value(self, gameweek, use_api=False):  # noqa: ARG002
        return 1000


class StubSquadOptimizer:
    """Records the requests it is given and returns a squad it was handed."""

    def __init__(self, squad=None, increments=7):
        self.squad = squad if squad is not None else StubSquad()
        self.increments = increments
        self.requests = []

    def num_increments(self, effort=None):
        return self.increments if effort is None else effort + self.increments

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


def test_genetic_algorithm_defaults():
    config = GeneticAlgorithmConfig()
    assert config.population_size == 100
    assert config.generations == 100
    assert config.crossover_prob == 0.7
    assert config.mutation_prob == 0.3
    assert config.tournament_size == 3
    assert config.random_state is None


def test_scaled_sets_population_and_generations_together():
    """The one knob the transfer search has drives both, explicitly, here."""
    scaled = GeneticAlgorithmConfig().scaled(7)
    assert scaled.population_size == 7
    assert scaled.generations == 7


def test_scaled_leaves_everything_else_alone():
    base = GeneticAlgorithmConfig(tournament_size=5, random_state=1)
    assert base.scaled(9) == replace(base, population_size=9, generations=9)


def test_increments_are_the_generations_the_search_will_run():
    config = GeneticAlgorithmConfig(generations=30)
    assert GeneticSquadOptimizer(config).num_increments() == 30


def test_an_effort_budget_resizes_the_search_without_mutating_the_optimizer():
    """
    How hard to search comes from the request, not from a second optimizer.

    The wildcard and free-hit path has one --num-iterations knob and passes it as
    `effort`; a standalone squad build passes none and keeps its full config.
    """
    optimizer = GeneticSquadOptimizer()

    assert optimizer.num_increments(20) == 20
    assert optimizer.num_increments() == 100
    # the optimizer is reusable: sizing one request did not change it
    assert optimizer.config.generations == 100
    assert optimizer.config.population_size == 100


def test_an_effort_budget_keeps_the_settings_it_does_not_size():
    optimizer = GeneticSquadOptimizer(
        GeneticAlgorithmConfig(tournament_size=5, random_state=1)
    )
    sized = optimizer._config_for(20)

    assert sized.population_size == 20
    assert sized.generations == 20
    assert sized.tournament_size == 5
    assert sized.random_state == 1


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


def _rebuild_request(squad_optimizer=None, num_iterations=100):
    return TransferRequest(
        move=GameweekMove(chip=Chip.WILDCARD),
        squad=StubSquad([1, 2, 3]),  # type: ignore[arg-type]
        tag="t",
        gameweeks=[1, 2],
        root_gw=1,
        season="2526",
        num_iterations=num_iterations,
        squad_optimizer=squad_optimizer,
    )


def test_the_registered_full_squad_strategy_defaults_to_genetic():
    """A request that names no optimizer still gets one, and it is the shipped one."""
    strategy = TRANSFER_STRATEGIES["full_squad"]()
    assert isinstance(strategy._optimizer(_rebuild_request()), GeneticSquadOptimizer)


def test_full_squad_sizes_the_default_optimizer_from_the_iteration_count():
    """
    The asymmetry this seam exists to remove.

    `optimize squad` could tune the genetic algorithm, but the wildcard and free
    hit path built its strategy with defaults and no caller could reach it.
    """
    strategy = TRANSFER_STRATEGIES["full_squad"]()
    assert strategy.num_increments(_rebuild_request(num_iterations=9)) == 9


def test_full_squad_sizes_its_cost_from_the_optimizer_on_the_request():
    """The number is the optimizer's, and it is told the effort budget to size to."""
    strategy = FullSquadStrategy()
    stub = StubSquadOptimizer(increments=42)

    assert strategy.num_increments(_rebuild_request(stub, num_iterations=15)) == 57


def test_the_optimizer_on_the_request_is_the_one_that_rebuilds_the_squad():
    """
    What §4 of the refactor is for: `TRANSFER_STRATEGIES` builds strategies by
    name with no arguments, so a constructor argument could never reach here.
    """
    stub = StubSquadOptimizer(squad=StubSquad([1, 4, 5]))
    request = _rebuild_request(stub)

    proposal = FullSquadStrategy().propose(request)

    assert proposal.squad is stub.squad
    # 1 was kept, so it is neither in nor out
    assert proposal.players_in == [4, 5]
    assert proposal.players_out == [2, 3]
    assert len(stub.requests) == 1
    assert stub.requests[0].effort == request.num_iterations


def test_scoring_config_carries_the_budget_the_optimizer_must_respect():
    scoring = SquadScoringConfig(budget=825)
    request = SquadRequest(gameweeks=[1], tag="t", season="2526", scoring=scoring)
    assert request.budget == 825
