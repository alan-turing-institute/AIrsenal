"""
The strategy-tree expansion, run for real on the small database.

Nothing else exercised it: the worker-failure tests drive the queue with
synthetic tasks, and the e2e transfer tests call `make_best_transfers`, which is
one node of the tree rather than the tree.

The worker is run in a thread rather than a forked process. The forking is not
what this is checking, and it cannot be checked here anyway: the search can only
fork before jax has been initialised, and predicting the points these strategies
are scored against initialises it. What is checked is everything the worker
does - pick a strategy for each move, score the resulting squad, extend the
strategy, and put the children back on the queue for the next gameweek.
"""

import math
import queue as queue_module
import threading

import pytest

from airsenal.core.concurrency import CustomQueue
from airsenal.db.queries.gameweeks import reset_gameweek_cache, set_next_gameweek
from airsenal.game.enums import Chip
from airsenal.optimization.moves import ChipSchedule, GameweekMove
from airsenal.optimization.plan import Plan, TransferSearchResult
from airsenal.optimization.protocols import (
    SquadRequest,
    TransferConstraints,
    TransferSearchRequest,
)
from airsenal.optimization.squad_optimizers import (
    GeneticAlgorithmConfig,
    GeneticSquadOptimizer,
)
from airsenal.optimization.squad_score import SquadScoringConfig
from airsenal.optimization.transfer_optimizers import TreeSearchConfig
from airsenal.optimization.transfer_optimizers.tree_search import optimize
from airsenal.prediction.player_models import (
    build_player_model,
)
from airsenal.prediction.run import make_predictedscore_table
from airsenal.prediction.team_models import (
    build_team_model,
)
from airsenal.squad.squad import SubWeights
from tests.e2e.conftest import FUTURE_GAMEWEEKS, SEASON

SEARCH_GAMEWEEKS = FUTURE_GAMEWEEKS[:2]


@pytest.fixture(scope="module")
def seeded(pipeline_db):
    set_next_gameweek(FUTURE_GAMEWEEKS[0])
    yield pipeline_db
    reset_gameweek_cache()
    set_next_gameweek(1)


@pytest.fixture(scope="module")
def tag(seeded):
    return make_predictedscore_table(
        gameweeks=FUTURE_GAMEWEEKS,
        season=SEASON,
        player_model=build_player_model("constant"),
        team_model=build_team_model("constant"),
        dbsession=seeded,
    )


@pytest.fixture(scope="module")
def starting_squad(seeded, tag):
    optimizer = GeneticSquadOptimizer(
        GeneticAlgorithmConfig(population_size=20, generations=5, random_state=0)
    )
    return optimizer.optimize(
        SquadRequest(
            gameweeks=SEARCH_GAMEWEEKS, tag=tag, season=SEASON, dbsession=seeded
        )
    )


@pytest.fixture(scope="module")
def result(seeded, tag, starting_squad):
    """Grow the whole tree with one in-thread worker, then read the answer off it."""
    request = TransferSearchRequest(
        starting_squad=starting_squad,
        gameweeks=SEARCH_GAMEWEEKS,
        tag=tag,
        season=SEASON,
        chip_schedule=ChipSchedule.from_weeks(SEARCH_GAMEWEEKS, {}),
        num_free_transfers=1,
        constraints=TransferConstraints(max_opt_transfers=1),
    )
    config = TreeSearchConfig(num_thread=1, num_iterations=5)

    work: CustomQueue = CustomQueue()
    finished: queue_module.Queue = queue_module.Queue()
    worker = threading.Thread(
        target=optimize, args=(work, 0, finished, request, config), daemon=True
    )
    worker.start()
    # the root node, which exists only to put this gameweek's moves on the queue
    work.put((GameweekMove(), request.num_free_transfers, 0, 0, starting_squad, None))
    work.join()
    work.put(None)
    worker.join(timeout=120)
    assert not worker.is_alive(), "the worker did not shut down"

    strategies = []
    while not finished.empty():
        strategies.append(finished.get())
    return TransferSearchResult.from_plans(strategies)


def test_the_tree_produces_finished_strategies(result):
    assert isinstance(result.best, Plan)
    assert len(result.best) == len(SEARCH_GAMEWEEKS)


def test_the_tree_branches(result):
    # one node per legal move per gameweek; a single result means it did not expand
    assert len(result.considered) > 1


def test_every_strategy_covers_every_gameweek(result):
    for strategy in result.considered:
        assert [o.gameweek for o in strategy.outcomes] == SEARCH_GAMEWEEKS


def test_the_baseline_is_among_them(result):
    assert result.baseline is not None
    assert result.baseline.is_baseline


def test_the_best_is_the_best_considered(result):
    assert result.best.total_score == max(s.total_score for s in result.considered)


def test_the_best_is_at_least_as_good_as_doing_nothing(result):
    assert result.best.total_score >= result.baseline_score


def test_scores_are_finite(result):
    assert all(math.isfinite(s.total_score) for s in result.considered)


def test_the_constraint_on_transfers_per_gameweek_is_respected(result):
    for strategy in result.considered:
        for outcome in strategy.outcomes:
            assert outcome.move.n_transfers <= 1


def test_no_strategy_goes_into_the_red(result):
    for strategy in result.considered:
        assert all(outcome.bank >= 0 for outcome in strategy.outcomes)


class RecordingSquadOptimizer:
    """Satisfies SquadOptimizer, records what it was asked, builds a real squad."""

    def __init__(self):
        self.requests = []
        self._real = GeneticSquadOptimizer(
            GeneticAlgorithmConfig(population_size=20, generations=5, random_state=0)
        )

    def num_increments(self, effort=None):
        return self._real.num_increments(effort)

    def optimize(self, request):
        self.requests.append(request)
        return self._real.optimize(request)


def test_the_squad_optimizer_on_the_request_rebuilds_a_wildcard_squad(
    seeded, tag, starting_squad
):
    """
    §4 of the refactor, checked end to end.

    `StrategySet` carries strategy *names*, so `FullSquadStrategy` is built with
    no arguments and a constructor argument could never reach it. Before this,
    a wildcard always rebuilt with the genetic algorithm no matter what the
    caller passed. The optimizer now travels on the request instead.
    """
    optimizer = RecordingSquadOptimizer()
    gameweeks = SEARCH_GAMEWEEKS[:1]
    request = TransferSearchRequest(
        starting_squad=starting_squad,
        gameweeks=gameweeks,
        tag=tag,
        season=SEASON,
        # forced, so every node of this one-gameweek tree plays it
        chip_schedule=ChipSchedule.from_weeks(gameweeks, {Chip.WILDCARD: gameweeks[0]}),
        num_free_transfers=1,
        constraints=TransferConstraints(max_opt_transfers=1),
        squad_optimizer=optimizer,
    )
    config = TreeSearchConfig(num_thread=1, num_iterations=5)

    work: CustomQueue = CustomQueue()
    finished: queue_module.Queue = queue_module.Queue()
    worker = threading.Thread(
        target=optimize, args=(work, 0, finished, request, config), daemon=True
    )
    worker.start()
    work.put((GameweekMove(), request.num_free_transfers, 0, 0, starting_squad, None))
    work.join()
    work.put(None)
    worker.join(timeout=120)
    assert not worker.is_alive(), "the worker did not shut down"

    assert optimizer.requests, "the wildcard rebuild did not reach the given optimizer"
    # the search's --num-iterations arrives as the effort budget to size to
    assert all(r.effort == config.num_iterations for r in optimizer.requests)


@pytest.mark.parametrize("chip", [None, Chip.WILDCARD])
def test_the_bench_weighting_on_the_request_reaches_the_search(
    seeded, tag, starting_squad, chip
):
    """
    §5 of the refactor: `--no-subs` used to apply to `optimize squad` alone.

    Every `get_discounted_squad_score` call on the transfer path omitted
    `sub_weights`, so the squad builder and the transfer search scored benches
    differently - the exact divergence `SquadScoringConfig` was created to end.
    Scoring the same window with and without the bench must not agree.
    """
    gameweeks = SEARCH_GAMEWEEKS[:1]
    chips = {chip: gameweeks[0]} if chip is not None else {}

    def score(scoring):
        request = TransferSearchRequest(
            starting_squad=starting_squad,
            gameweeks=gameweeks,
            tag=tag,
            season=SEASON,
            chip_schedule=ChipSchedule.from_weeks(gameweeks, chips),
            num_free_transfers=1,
            constraints=TransferConstraints(max_opt_transfers=1),
            scoring=scoring,
        )
        config = TreeSearchConfig(num_thread=1, num_iterations=5)
        work: CustomQueue = CustomQueue()
        finished: queue_module.Queue = queue_module.Queue()
        worker = threading.Thread(
            target=optimize, args=(work, 0, finished, request, config), daemon=True
        )
        worker.start()
        work.put((GameweekMove(), 1, 0, 0, starting_squad, None))
        work.join()
        work.put(None)
        worker.join(timeout=120)
        assert not worker.is_alive(), "the worker did not shut down"
        plans = []
        while not finished.empty():
            plans.append(finished.get())
        return TransferSearchResult.from_plans(plans).best.total_score

    with_bench = score(SquadScoringConfig())
    without_bench = score(SquadScoringConfig(sub_weights=SubWeights.none()))
    assert with_bench != without_bench
