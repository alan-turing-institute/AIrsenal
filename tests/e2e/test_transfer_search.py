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
from airsenal.optimization.config import GeneticAlgorithmConfig
from airsenal.optimization.moves import (
    ChipSchedule,
    GameweekMove,
    TransferConstraints,
)
from airsenal.optimization.protocols import SquadRequest, TransferSearchRequest
from airsenal.optimization.squad_optimizers import SQUAD_OPTIMIZERS
from airsenal.optimization.strategy import Strategy, TransferSearchResult
from airsenal.optimization.transfer_optimizers import TreeSearchConfig
from airsenal.optimization.transfer_optimizers.tree_search import optimize
from airsenal.prediction.registry import PLAYER_MODELS, build_team_model
from airsenal.prediction.run import make_predictedscore_table
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
        player_model=PLAYER_MODELS.create("constant"),
        team_model=build_team_model("constant"),
        dbsession=seeded,
    )


@pytest.fixture(scope="module")
def starting_squad(seeded, tag):
    optimizer = SQUAD_OPTIMIZERS.create(
        "genetic",
        GeneticAlgorithmConfig(population_size=20, generations=5, random_state=0),
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
    return TransferSearchResult.from_strategies(strategies)


def test_the_tree_produces_finished_strategies(result):
    assert isinstance(result.best, Strategy)
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
