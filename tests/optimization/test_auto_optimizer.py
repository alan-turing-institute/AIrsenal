"""The automatic transfer optimizer: the tree search on a small tree, MCTS on a big."""

import pytest

from airsenal.optimization.transfer_optimizers import auto
from airsenal.optimization.transfer_optimizers.auto import (
    TREE_TO_BUDGET_RATIO,
    AutoOptimizer,
)
from airsenal.optimization.transfer_optimizers.mcts import MCTSConfig, MCTSOptimizer
from airsenal.optimization.transfer_optimizers.tree_search import (
    TreeSearchConfig,
    TreeSearchOptimizer,
)

BUDGET = 50


class _Recording:
    """Stands in for one of the two searches, noting that it was asked."""

    def __init__(self, name: str, searched: list[str]) -> None:
        self.name = name
        self.searched = searched

    def search(self, request):  # noqa: ARG002 - a search's signature
        self.searched.append(self.name)
        return self.name


@pytest.fixture
def searched(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        TreeSearchOptimizer, "search", _Recording("tree search", calls).search
    )
    monkeypatch.setattr(MCTSOptimizer, "search", _Recording("mcts", calls).search)
    return calls


def _optimizer() -> AutoOptimizer:
    return AutoOptimizer(
        tree_search=TreeSearchOptimizer(TreeSearchConfig()),
        mcts=MCTSOptimizer(MCTSConfig(max_expansions=BUDGET)),
    )


@pytest.mark.parametrize(
    ("n_nodes", "expected"),
    [
        (1, "tree search"),
        (TREE_TO_BUDGET_RATIO * BUDGET, "tree search"),
        (TREE_TO_BUDGET_RATIO * BUDGET + 1, "mcts"),
        (100 * BUDGET, "mcts"),
    ],
    ids=["tiny", "at the threshold", "just past it", "huge"],
)
def test_the_tree_size_against_the_budget_chooses_the_search(
    monkeypatch, searched, n_nodes, expected
):
    monkeypatch.setattr(auto, "count_tree_nodes", lambda *a, **k: n_nodes)

    _optimizer().search(request=None)  # type: ignore[arg-type]

    assert searched == [expected]


def test_the_tree_is_counted_as_the_tree_search_would_walk_it(monkeypatch, searched):
    """With coarse transfer counts if that is how the tree search is set up."""
    counted_with = []

    def count(request, *, every_transfer_count):
        counted_with.append(every_transfer_count)
        return 1

    monkeypatch.setattr(auto, "count_tree_nodes", count)
    optimizer = AutoOptimizer(
        tree_search=TreeSearchOptimizer(TreeSearchConfig(every_transfer_count=True))
    )

    optimizer.search(request=None)  # type: ignore[arg-type]

    assert counted_with == [True]
