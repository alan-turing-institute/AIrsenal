"""
Monte Carlo tree search over a gameweek window's moves.

The same tree as the exhaustive search, walked selectively: each step descends
by UCB1 to a node with a move not yet tried, makes that move, and backs up an
estimate of what the node is worth. A node is only made once, so the cost is the
number of nodes made, which is what `max_expansions` bounds.

There is no random rollout. A node's value is the points its plan has banked
plus what its squad scores if held, unchanged, to the end of the window.

The dynamics are deterministic: making the same move from the same node gives
the same squad and score. So every finished plan the search reaches is exactly
scored, and the result is the best of them, not the most-visited path.
"""

import math
import random
from dataclasses import dataclass, field

from airsenal.core.console import progress_bar
from airsenal.optimization.moves import GameweekMove
from airsenal.optimization.plan import (
    GameweekOutcome,
    Plan,
    TransferSearchResult,
    baseline_plan,
)
from airsenal.optimization.protocols import (
    DEFAULT_NUM_ITERATIONS,
    TransferRequest,
    TransferSearchRequest,
)
from airsenal.optimization.squad_score import (
    get_discount_factor,
    get_discounted_squad_score,
)
from airsenal.optimization.strategies import DEFAULT_STRATEGIES, StrategySet
from airsenal.optimization.transfer_optimizers.branches import (
    make_best_transfers,
    next_gameweek_transfers,
)
from airsenal.squad.squad import Squad

# (move, free transfers it leaves, total hit including it, hit this gameweek), as
# `next_gameweek_transfers` returns them.
type Branch = tuple[GameweekMove, int, int, int]


@dataclass(frozen=True)
class MCTSConfig:
    """Settings for the Monte Carlo search, as opposed to the problem it is solving."""

    # Nodes to make before stopping, each one a strategy run and scored. On a
    # six-gameweek window the exhaustive tree makes about a thousand.
    max_expansions: int = 200
    # In points, since that is what a node's value is in: how far a rarely
    # visited move may trail the best one and still be tried.
    exploration_constant: float = 5.0
    num_iterations: int = DEFAULT_NUM_ITERATIONS
    seed: int | None = None
    strategies: StrategySet = field(default_factory=lambda: DEFAULT_STRATEGIES)


class _Node:
    """One node of the search tree: a plan so far and the squad it leaves."""

    def __init__(
        self,
        squad: Squad,
        plan: Plan,
        free_transfers: int,
        hit_so_far: int,
        parent: "_Node | None" = None,
    ) -> None:
        self.squad = squad
        self.plan = plan
        self.free_transfers = free_transfers
        self.hit_so_far = hit_so_far
        self.parent = parent
        self.children: list[_Node] = []
        # None until the node is first reached, then the moves not yet made
        self.untried: list[Branch] | None = None
        self.visits = 0
        self.total_value = 0.0
        # every node below this one has been made
        self.exhausted = False

    @property
    def depth(self) -> int:
        return len(self.plan)

    @property
    def mean_value(self) -> float:
        return self.total_value / self.visits


class _Search:
    """One run of the search, holding the tree and the plans it finished."""

    def __init__(self, request: TransferSearchRequest, config: MCTSConfig) -> None:
        self.request = request
        self.config = config
        self.rng = random.Random(config.seed)
        self.n_gameweeks = len(request.gameweeks)
        self.finished: list[Plan] = []
        self.expansions = 0
        self.root = _Node(
            request.starting_squad,
            Plan(root_gameweek=request.gameweeks[0]),
            request.num_free_transfers,
            0,
        )

    def branches(self, node: _Node) -> list[Branch]:
        constraints = self.request.constraints
        gameweek = self.request.gameweeks[node.depth]
        return next_gameweek_transfers(
            node.free_transfers,
            node.hit_so_far,
            node.plan.chips_played,
            max_total_hit=constraints.max_total_hit,
            allow_unused_transfers=constraints.allow_unused_transfers,
            max_opt_transfers=constraints.max_opt_transfers,
            chips=self.request.chip_schedule.for_gameweek(gameweek),
            max_free_transfers=constraints.max_free_transfers,
        )

    def expand(self, node: _Node, branch: Branch) -> _Node:
        """Make one move from `node`, as a worker of the exhaustive search would."""
        move, free_transfers, hit_so_far, hit_this_gameweek = branch
        request = self.request
        root_gameweek = node.plan.root_gameweek
        gameweek = request.gameweeks[node.depth]
        transfer_request = TransferRequest(
            move=move,
            squad=node.squad,
            tag=request.tag,
            gameweeks=request.gameweeks[node.depth :],
            root_gameweek=root_gameweek,
            season=request.season,
            num_iterations=self.config.num_iterations,
            scoring=request.scoring,
            squad_optimizer=request.squad_optimizer,
        )
        new_squad, proposal, points = make_best_transfers(
            transfer_request, self.config.strategies.create(move)
        )
        self.expansions += 1

        discount_factor = get_discount_factor(root_gameweek, gameweek)
        plan = node.plan.extend(
            GameweekOutcome(
                gameweek=gameweek,
                move=move,
                points=points - hit_this_gameweek * discount_factor,
                discount_factor=discount_factor,
                points_hit=hit_this_gameweek,
                free_transfers=free_transfers,
                players_in=tuple(proposal.players_in),
                players_out=tuple(proposal.players_out),
                bank=new_squad.budget,
            )
        )
        child = _Node(new_squad, plan, free_transfers, hit_so_far, parent=node)
        node.children.append(child)
        if child.depth == self.n_gameweeks:
            self.finished.append(plan)
        return child

    def value(self, node: _Node) -> float:
        """Points banked so far, plus the squad's score if held to the end."""
        remaining = self.request.gameweeks[node.depth :]
        if not remaining:
            return node.plan.total_score
        return node.plan.total_score + get_discounted_squad_score(
            node.squad,
            remaining,
            self.request.tag,
            root_gameweek=node.plan.root_gameweek,
            sub_weights=self.request.scoring.sub_weights,
        )

    def select(self) -> _Node:
        """Descend by UCB1 to the first node that still has a move to make."""
        node = self.root
        while node.untried == [] and node.depth < self.n_gameweeks:
            parent = node
            candidates = [child for child in node.children if not child.exhausted]
            node = max(candidates, key=lambda child: self.ucb(child, parent.visits))
        return node

    def ucb(self, child: _Node, parent_visits: int) -> float:
        exploration = math.sqrt(math.log(parent_visits) / child.visits)
        return child.mean_value + self.config.exploration_constant * exploration

    def mark_exhausted(self, node: _Node | None) -> None:
        """Mark `node`, and each ancestor it completes, as having nothing left."""
        while node is not None:
            if node.depth < self.n_gameweeks and not (
                node.untried == [] and all(child.exhausted for child in node.children)
            ):
                return
            node.exhausted = True
            node = node.parent

    def step(self) -> None:
        """Make one node, or find that the one selected has no legal move."""
        node = self.select()
        if node.untried is None:
            node.untried = self.branches(node)
        if node.untried:
            branch = node.untried.pop(self.rng.randrange(len(node.untried)))
            node = self.expand(node, branch)
        self.mark_exhausted(node)

        value = self.value(node)
        current: _Node | None = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent

    def run(self) -> TransferSearchResult:
        with progress_bar(transient=True) as progress:
            task = progress.add_task("Nodes searched", total=self.config.max_expansions)
            while (
                self.expansions < self.config.max_expansions and not self.root.exhausted
            ):
                before = self.expansions
                self.step()
                progress.advance(task, self.expansions - before)

        plans = list(self.finished)
        if not any(plan.is_baseline for plan in plans):
            # the search may not have reached it, and it is what a plan is judged
            # against - as well as the answer when nothing else finished
            plans.append(
                baseline_plan(
                    self.request.starting_squad,
                    self.request.gameweeks,
                    self.request.tag,
                    sub_weights=self.request.scoring.sub_weights,
                )
            )
        return TransferSearchResult.from_plans(plans)


class MCTSOptimizer:
    """Chooses transfers by searching the most promising parts of the plan tree."""

    def __init__(self, config: MCTSConfig | None = None) -> None:
        self.config = config if config is not None else MCTSConfig()

    def search(self, request: TransferSearchRequest) -> TransferSearchResult:
        return _Search(request, self.config).run()
