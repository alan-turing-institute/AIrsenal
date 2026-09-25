"""
The Monte Carlo transfer search, on a tree whose scores are made up.

Each move scores a fixed number of points that depends only on the move and its
gameweek, so the best plan can be found by brute force and compared.
"""

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


def _request(
    constraints: TransferConstraints | None = None,
    chips: ChipGameweeks | None = None,
    num_free_transfers: int = 1,
) -> TransferSearchRequest:
    return TransferSearchRequest(
        starting_squad=_Squad(),  # type: ignore[arg-type]
        gameweeks=GAMEWEEKS,
        tag="tag",
        season="2425",
        chip_schedule=ChipSchedule.from_gameweeks(GAMEWEEKS, chips or ChipGameweeks()),
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
def test_a_budget_as_big_as_the_tree_finds_the_best_plan(request_):
    """Once every node is made, the search is exhaustive, and must agree with it."""
    best, n_nodes = _brute_force(request_)
    optimizer = MCTSOptimizer(MCTSConfig(max_expansions=10 * n_nodes, seed=0))

    result = optimizer.search(request_)

    assert result.best.total_score == pytest.approx(best)
    assert len(result.best) == len(GAMEWEEKS)


def test_the_search_stops_once_every_node_is_made(made_up_scores):
    """The rest of the budget would only revisit plans already finished."""
    request = _request()
    _, n_nodes = _brute_force(request)

    MCTSOptimizer(MCTSConfig(max_expansions=10 * n_nodes, seed=0)).search(request)

    assert len(made_up_scores) == n_nodes


def test_no_node_is_made_twice(made_up_scores):
    request = _request(TransferConstraints(max_total_hit=None, max_opt_transfers=3))

    result = MCTSOptimizer(MCTSConfig(max_expansions=1000, seed=0)).search(request)

    labels = [plan.label() for plan in result.considered]
    assert len(labels) == len(set(labels))


@pytest.mark.parametrize("max_expansions", [1, 5, 12])
def test_the_budget_is_counted_in_nodes_made(made_up_scores, max_expansions):
    MCTSOptimizer(MCTSConfig(max_expansions=max_expansions, seed=0)).search(
        _request(TransferConstraints(max_total_hit=None, max_opt_transfers=3))
    )

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


def test_a_seed_makes_the_search_repeatable():
    request = _request(TransferConstraints(max_total_hit=None, max_opt_transfers=3))
    config = MCTSConfig(max_expansions=10, seed=3)

    first = MCTSOptimizer(config).search(request)
    second = MCTSOptimizer(config).search(request)

    assert [p.label() for p in first.considered] == [
        p.label() for p in second.considered
    ]


def test_the_mcts_is_a_named_transfer_optimizer():
    """Adding it is a table entry: `--transfer-optimizer mcts` builds it."""
    assert TRANSFER_OPTIMIZERS["mcts"] is MCTSOptimizer
    assert isinstance(build_transfer_optimizer("mcts"), MCTSOptimizer)
