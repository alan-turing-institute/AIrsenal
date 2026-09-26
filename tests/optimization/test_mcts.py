"""
The Monte Carlo transfer search, on a tree whose scores are made up.

Each move scores a fixed number of points that depends only on the move and its
gameweek, so the best plan can be found by brute force and compared. Workers are
threads rather than forked processes, which pytest on macOS cannot fork safely.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from airsenal.game.enums import Chip
from airsenal.optimization.moves import ChipGameweeks, ChipSchedule, GameweekMove
from airsenal.optimization.plan import GameweekOutcome, Plan
from airsenal.optimization.protocols import (
    Proposal,
    TransferConstraints,
    TransferRequest,
    TransferSearchRequest,
)
from airsenal.optimization.transfer_optimizers import (
    TRANSFER_OPTIMIZERS,
    build_transfer_optimizer,
    mcts,
)
from airsenal.optimization.transfer_optimizers.branches import (
    count_tree_nodes,
    next_gameweek_transfers,
)
from airsenal.optimization.transfer_optimizers.mcts import MCTSConfig, MCTSOptimizer

GAMEWEEKS = [5, 6, 7]


def _points(move: GameweekMove, gameweek: int) -> float:
    """An arbitrary, deterministic score with a best plan that is not all one move."""
    if move.chip is Chip.WILDCARD:
        return 9.0 if gameweek == 6 else 1.0
    return float((move.n_transfers * gameweek) % 5) + 0.5 * move.n_transfers


class _Squad:
    budget = 0


@pytest.fixture(autouse=True)
def made_up_scores(monkeypatch):
    """Score every move with `_points`, and a held squad as worth nothing more."""
    calls = []

    def make_best_transfers(request: TransferRequest, strategy):
        calls.append(request.move)
        squad = request.squad
        points = _points(request.move, request.transfer_gameweek)
        return squad, Proposal(squad, [], []), points

    def baseline_plan(squad, gameweeks, tag, root_gameweek=None, *, sub_weights):
        outcomes = tuple(
            GameweekOutcome(
                gameweek=gameweek,
                move=GameweekMove(),
                points=_points(GameweekMove(), gameweek),
                discount_factor=1.0,
                points_hit=0,
                free_transfers=0,
            )
            for gameweek in gameweeks
        )
        return Plan(root_gameweek=gameweeks[0], outcomes=outcomes)

    monkeypatch.setattr(mcts, "make_best_transfers", make_best_transfers)
    monkeypatch.setattr(mcts, "get_discounted_squad_score", lambda *a, **k: 0.0)
    monkeypatch.setattr(mcts, "get_discount_factor", lambda *a, **k: 1.0)
    monkeypatch.setattr(mcts, "baseline_plan", baseline_plan)
    return calls


@pytest.fixture(autouse=True)
def thread_workers(monkeypatch):
    """Make nodes on threads, started the way the forked workers are."""

    def executor(request, config):
        return ThreadPoolExecutor(
            max_workers=config.num_thread,
            initializer=mcts._start_worker,
            initargs=(request, config),
        )

    monkeypatch.setattr(mcts, "_executor", executor)


def _request(
    constraints: TransferConstraints | None = None,
    chips: ChipGameweeks | None = None,
    num_free_transfers: int = 1,
    gameweeks: list[int] = GAMEWEEKS,
) -> TransferSearchRequest:
    return TransferSearchRequest(
        starting_squad=_Squad(),  # type: ignore[arg-type]
        gameweeks=gameweeks,
        tag="tag",
        season="2425",
        chip_schedule=ChipSchedule.from_gameweeks(gameweeks, chips or ChipGameweeks()),
        num_free_transfers=num_free_transfers,
        constraints=constraints or TransferConstraints(),
    )


def _brute_force(request: TransferSearchRequest) -> tuple[float, int]:
    """The best total score in the whole tree, and how many nodes the tree has."""
    constraints = request.constraints

    def walk(depth, free_transfers, hit_so_far, chips_played):
        if depth == len(request.gameweeks):
            return 0.0, 0
        gameweek = request.gameweeks[depth]
        branches = next_gameweek_transfers(
            free_transfers,
            hit_so_far,
            chips_played,
            max_total_hit=constraints.max_total_hit,
            allow_unused_transfers=constraints.allow_unused_transfers,
            max_opt_transfers=constraints.max_opt_transfers,
            chips=request.chip_schedule.for_gameweek(gameweek),
            max_free_transfers=constraints.max_free_transfers,
        )
        best, nodes = float("-inf"), 0
        for move, new_free_transfers, new_hit, hit in branches:
            rest, rest_nodes = walk(
                depth + 1, new_free_transfers, new_hit, [*chips_played, move.chip]
            )
            best = max(best, _points(move, gameweek) - hit + rest)
            nodes += 1 + rest_nodes
        return best, nodes

    return walk(0, request.num_free_transfers, 0, [])


@pytest.mark.parametrize(
    "request_",
    [
        _request(),
        _request(TransferConstraints(max_total_hit=None, max_opt_transfers=3)),
        _request(chips=ChipGameweeks(wildcard=0)),
    ],
    ids=["defaults", "no hit limit", "wildcard any gameweek"],
)
@pytest.mark.parametrize("num_thread", [1, 4])
def test_a_budget_as_big_as_the_tree_finds_the_best_plan(request_, num_thread):
    """Once every node is made, the search is exhaustive, and must agree with it."""
    best, n_nodes = _brute_force(request_)
    optimizer = MCTSOptimizer(
        MCTSConfig(max_expansions=10 * n_nodes, num_thread=num_thread, seed=0)
    )

    result = optimizer.search(request_)

    assert result.best.total_score == pytest.approx(best)
    assert len(result.best) == len(GAMEWEEKS)


@pytest.mark.parametrize(
    "request_",
    [
        _request(),
        _request(TransferConstraints(max_total_hit=None, max_opt_transfers=3)),
        _request(chips=ChipGameweeks(wildcard=0)),
    ],
    ids=["defaults", "no hit limit", "wildcard any gameweek"],
)
def test_the_tree_is_sized_in_the_nodes_a_search_makes(request_):
    """What the automatic optimizer compares with the MCTS budget."""
    _, n_nodes = _brute_force(request_)

    assert count_tree_nodes(request_, every_transfer_count=True) == n_nodes


@pytest.mark.parametrize("num_thread", [1, 4])
def test_the_search_stops_once_every_node_is_made(made_up_scores, num_thread):
    """
    The rest of the budget would only revisit plans already finished.

    With workers, a node must not be closed off while a move below it is still
    out, or the subtree that move starts would never be searched.
    """
    request = _request(TransferConstraints(max_total_hit=None, max_opt_transfers=3))
    _, n_nodes = _brute_force(request)

    MCTSOptimizer(
        MCTSConfig(max_expansions=10 * n_nodes, num_thread=num_thread, seed=0)
    ).search(request)

    assert len(made_up_scores) == n_nodes


def test_no_node_is_made_twice(made_up_scores):
    request = _request(TransferConstraints(max_total_hit=None, max_opt_transfers=3))

    result = MCTSOptimizer(MCTSConfig(max_expansions=1000, seed=0)).search(request)

    labels = [plan.label() for plan in result.considered]
    assert len(labels) == len(set(labels))


@pytest.mark.parametrize("num_thread", [1, 4])
@pytest.mark.parametrize("max_expansions", [1, 5, 12])
def test_the_budget_is_counted_in_nodes_made(
    made_up_scores, max_expansions, num_thread
):
    MCTSOptimizer(
        MCTSConfig(max_expansions=max_expansions, num_thread=num_thread, seed=0)
    ).search(_request(TransferConstraints(max_total_hit=None, max_opt_transfers=3)))

    assert len(made_up_scores) == max_expansions


def test_the_result_is_the_best_plan_it_finished():
    """Not the most-visited path, which can be a worse plan."""
    result = MCTSOptimizer(MCTSConfig(max_expansions=15, seed=1)).search(
        _request(TransferConstraints(max_total_hit=None, max_opt_transfers=3))
    )

    assert result.best.total_score == max(p.total_score for p in result.considered)


def test_the_baseline_is_there_even_when_the_search_never_reached_it():
    """Too small a budget to finish any plan leaves doing nothing as the answer."""
    result = MCTSOptimizer(MCTSConfig(max_expansions=1, seed=0)).search(_request())

    assert result.baseline is not None
    assert result.baseline.is_baseline
    assert result.best is result.baseline


def test_no_legal_move_leaves_the_baseline_alone():
    """--max-transfers 0 with a full bank of free transfers admits nothing."""
    request = _request(
        TransferConstraints(max_opt_transfers=0),
        num_free_transfers=5,
    )

    result = MCTSOptimizer(MCTSConfig(seed=0)).search(request)

    assert result.best.is_baseline


def test_a_seed_makes_a_single_worker_search_repeatable():
    """With workers, the order nodes come back in is up to the workers."""
    request = _request(TransferConstraints(max_total_hit=None, max_opt_transfers=3))
    config = MCTSConfig(max_expansions=10, num_thread=1, seed=3)

    first = MCTSOptimizer(config).search(request)
    second = MCTSOptimizer(config).search(request)

    assert [p.label() for p in first.considered] == [
        p.label() for p in second.considered
    ]


def test_workers_make_nodes_at_the_same_time(monkeypatch, made_up_scores):
    """Up to `num_thread` moves are out at once, not one after another."""
    num_thread = 3
    # every worker must be inside at once for any of them to get past it
    barrier = threading.Barrier(num_thread, timeout=10)
    make = mcts.make_best_transfers

    def make_together(request, strategy):
        if len(made_up_scores) < num_thread:
            barrier.wait()
        return make(request, strategy)

    monkeypatch.setattr(mcts, "make_best_transfers", make_together)
    request = _request(TransferConstraints(max_total_hit=None, max_opt_transfers=3))

    MCTSOptimizer(
        MCTSConfig(max_expansions=num_thread, num_thread=num_thread, seed=0)
    ).search(request)

    assert len(made_up_scores) == num_thread


def test_a_node_is_not_closed_off_while_a_move_below_it_is_out():
    """
    Otherwise the search stops early and the moves still out are never used.

    A one-gameweek window, so every move made is a finished plan: once the first
    comes back, every node the root has is done, but the others are still out.
    """
    request = _request(
        TransferConstraints(max_total_hit=None, max_opt_transfers=3), gameweeks=[5]
    )
    search = mcts._Search(request, MCTSConfig(num_thread=4, seed=0))
    claims = []
    while (claim := search.claim()) is not None:
        claims.append(claim)
    assert len(claims) > 1

    for node, branch in claims[:-1]:
        made = mcts._make_node(
            request, search.config, node.squad, branch[0], 0, search.candidates
        )
        search.complete(node, branch, made)
        assert not search.root.exhausted

    node, branch = claims[-1]
    made = mcts._make_node(
        request, search.config, node.squad, branch[0], 0, search.candidates
    )
    search.complete(node, branch, made)
    assert search.root.exhausted


def test_a_move_being_made_steers_the_next_choice_away():
    """A subtree with a move out counts it as a visit, so UCB1 favours the other."""
    search = mcts._Search(_request(), MCTSConfig(num_thread=2, seed=0))
    parent = search.root
    parent.untried = []
    first, second = (
        mcts._Node(parent.squad, parent.plan, 1, 0, parent=parent) for _ in range(2)
    )
    parent.children = [first, second]
    for child in (first, second):
        child.visits, child.total_value = 1, 10.0
    parent.visits = 2
    search.mark_pending(first, 1)

    assert search.ucb(second, parent) > search.ucb(first, parent)


def test_the_mcts_is_a_named_transfer_optimizer():
    """Adding it is a table entry: `--transfer-optimizer mcts` builds it."""
    assert "mcts" in TRANSFER_OPTIMIZERS
    assert isinstance(build_transfer_optimizer("mcts"), MCTSOptimizer)
