"""
The exhaustive tree search on a small plan tree, the Monte Carlo search on a big one.

Both make the same nodes at the same cost each, so the choice is only how many
to make. The exhaustive search makes every node and is certain of the best plan;
the Monte Carlo search makes at most its budget. Once the tree is several times
the budget, the Monte Carlo search is the cheaper one, and on six- and
seven-gameweek windows it found the exhaustive search's best plan well within
its default budget.
"""

from airsenal.core.logging import get_logger
from airsenal.optimization.plan import TransferSearchResult
from airsenal.optimization.protocols import TransferSearchRequest
from airsenal.optimization.transfer_optimizers.branches import count_tree_nodes
from airsenal.optimization.transfer_optimizers.mcts import MCTSOptimizer
from airsenal.optimization.transfer_optimizers.tree_search import TreeSearchOptimizer

logger = get_logger(__name__)

# The Monte Carlo search is used once the tree has more than this many times as
# many nodes as its budget. Below that it would make most of the tree anyway,
# for no certainty that it found the best plan.
TREE_TO_BUDGET_RATIO = 2


class AutoOptimizer:
    """Chooses the tree search or the Monte Carlo search by the size of the tree."""

    def __init__(
        self,
        tree_search: TreeSearchOptimizer | None = None,
        mcts: MCTSOptimizer | None = None,
    ) -> None:
        self.tree_search = (
            tree_search if tree_search is not None else TreeSearchOptimizer()
        )
        self.mcts = mcts if mcts is not None else MCTSOptimizer()

    def search(self, request: TransferSearchRequest) -> TransferSearchResult:
        n_nodes = count_tree_nodes(
            request, every_transfer_count=self.tree_search.config.every_transfer_count
        )
        budget = self.mcts.config.max_expansions
        if n_nodes > TREE_TO_BUDGET_RATIO * budget:
            logger.info(
                "The plan tree has %s nodes: searching at most %s of them by MCTS",
                n_nodes,
                budget,
            )
            return self.mcts.search(request)
        logger.info("The plan tree has %s nodes: searching all of them", n_nodes)
        return self.tree_search.search(request)
