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

Making a node is nearly all of the cost, so with `num_thread` above one this
process keeps the tree and forked workers make the nodes. Up to `num_thread`
moves are out at once; a subtree with a move out counts that move as a visit
when UCB1 chooses between subtrees, so the next choice goes elsewhere.
"""

import math
import multiprocessing
import os
import random
from collections.abc import Callable
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ProcessPoolExecutor,
    wait,
)
from dataclasses import dataclass, field

from airsenal.core.console import progress_bar
from airsenal.core.logging import relay_child_logs
from airsenal.optimization.moves import GameweekMove
from airsenal.optimization.plan import (
    GameweekOutcome,
    Plan,
    TransferSearchResult,
    baseline_plan,
)
from airsenal.optimization.protocols import (
    DEFAULT_NUM_ITERATIONS,
    Proposal,
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

# What making a move returns, as `make_best_transfers` does.
type MadeNode = tuple[Squad, Proposal, float]


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
    # Worker processes making nodes. One makes them in this process, which is
    # the only setting a seed makes repeatable.
    num_thread: int = 4
    seed: int | None = None
    strategies: StrategySet = field(default_factory=lambda: DEFAULT_STRATEGIES)


def _make_node(
    request: TransferSearchRequest,
    config: MCTSConfig,
    squad: Squad,
    move: GameweekMove,
    depth: int,
) -> MadeNode:
    """Make one move from a node's squad, as a worker of the exhaustive search would."""
    transfer_request = TransferRequest(
        move=move,
        squad=squad,
        tag=request.tag,
        gameweeks=request.gameweeks[depth:],
        root_gameweek=request.gameweeks[0],
        season=request.season,
        num_iterations=config.num_iterations,
        scoring=request.scoring,
        squad_optimizer=request.squad_optimizer,
    )
    return make_best_transfers(transfer_request, config.strategies.create(move))


# The search a worker process makes nodes for, set once when the worker starts
# rather than sent with every node.
_worker_search: tuple[TransferSearchRequest, MCTSConfig] | None = None


def _start_worker(request: TransferSearchRequest, config: MCTSConfig) -> None:
    global _worker_search  # noqa: PLW0603
    _worker_search = (request, config)


def _make_node_in_worker(squad: Squad, move: GameweekMove, depth: int) -> MadeNode:
    if _worker_search is None:
        msg = "An MCTS worker was asked for a node before it was started"
        raise RuntimeError(msg)
    request, config = _worker_search
    return _make_node(request, config, squad, move, depth)


def _executor(request: TransferSearchRequest, config: MCTSConfig) -> Executor:
    """
    The worker pool that makes nodes.

    Forked, like the tree search's workers: the request and config are inherited
    rather than pickled, and a worker must not start jax afresh.
    """
    context = multiprocessing.get_context("fork") if os.name == "posix" else None
    return ProcessPoolExecutor(
        max_workers=config.num_thread,
        mp_context=context,
        initializer=_start_worker,
        initargs=(request, config),
    )


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
        # moves being made at or below this node, not yet back
        self.pending = 0
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
        # moves handed out to be made, and moves made
        self.claimed = 0
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
            self.request.chips_used_up(node.plan, gameweek),
            max_total_hit=constraints.max_total_hit,
            allow_unused_transfers=constraints.allow_unused_transfers,
            max_opt_transfers=constraints.max_opt_transfers,
            chips=self.request.chip_schedule.for_gameweek(gameweek),
            max_free_transfers=constraints.max_free_transfers,
        )

    def attach(self, node: _Node, branch: Branch, made: MadeNode) -> _Node:
        """Add the node that making `branch` from `node` produced."""
        move, free_transfers, hit_so_far, hit_this_gameweek = branch
        new_squad, proposal, points = made
        self.expansions += 1
        gameweek = self.request.gameweeks[node.depth]
        discount_factor = get_discount_factor(node.plan.root_gameweek, gameweek)
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

    def claimable(self, node: _Node) -> bool:
        """Whether a move can be handed out at or below `node` now."""
        if node.exhausted or node.depth == self.n_gameweeks:
            return False
        if node.untried is None or node.untried:
            return True
        return any(self.claimable(child) for child in node.children)

    def select(self) -> _Node:
        """Descend by UCB1 to the first node that still has a move to make."""
        node = self.root
        while node.untried == [] and node.depth < self.n_gameweeks:
            parent = node
            candidates = [child for child in node.children if self.claimable(child)]
            node = max(candidates, key=lambda child: self.ucb(child, parent))
        return node

    def ucb(self, child: _Node, parent: _Node) -> float:
        # a move still out counts as a visit, so it steers the next choice away
        exploration = math.sqrt(
            math.log(parent.visits + parent.pending) / (child.visits + child.pending)
        )
        return child.mean_value + self.config.exploration_constant * exploration

    def mark_exhausted(self, node: _Node | None) -> None:
        """Mark `node`, and each ancestor it completes, as having nothing left."""
        while node is not None:
            if node.depth < self.n_gameweeks and not (
                node.untried == []
                and node.pending == 0
                and all(child.exhausted for child in node.children)
            ):
                return
            node.exhausted = True
            node = node.parent

    def back_up(self, node: _Node, value: float) -> None:
        current: _Node | None = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent

    def mark_pending(self, node: _Node, change: int) -> None:
        current: _Node | None = node
        while current is not None:
            current.pending += change
            current = current.parent

    def claim(self) -> tuple[_Node, Branch] | None:
        """
        The next move to make, marked as out, or None if none can be handed out.

        A node found to have no legal move is closed off on the way.
        """
        while self.claimed < self.config.max_expansions and self.claimable(self.root):
            node = self.select()
            if node.untried is None:
                node.untried = self.branches(node)
            if node.untried:
                branch = node.untried.pop(self.rng.randrange(len(node.untried)))
                self.claimed += 1
                self.mark_pending(node, 1)
                return node, branch
            self.mark_exhausted(node)
            self.back_up(node, self.value(node))
        return None

    def complete(self, node: _Node, branch: Branch, made: MadeNode) -> None:
        """Take back a move that has been made, and back up what it is worth."""
        self.mark_pending(node, -1)
        child = self.attach(node, branch, made)
        self.mark_exhausted(child)
        self.back_up(child, self.value(child))

    def run(self, executor: Executor | None = None) -> TransferSearchResult:
        """Search, making nodes here, or in `executor`'s workers if given."""
        with progress_bar(transient=True) as progress:
            task = progress.add_task("Nodes searched", total=self.config.max_expansions)
            if executor is None:
                while (claim := self.claim()) is not None:
                    node, branch = claim
                    made = _make_node(
                        self.request, self.config, node.squad, branch[0], node.depth
                    )
                    self.complete(node, branch, made)
                    progress.advance(task)
            else:
                self.run_in(executor, lambda: progress.advance(task))

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

    def run_in(self, executor: Executor, advance: Callable[[], None]) -> None:
        """Keep up to `num_thread` moves out at once until nothing is left to make."""
        out: dict[Future[MadeNode], tuple[_Node, Branch]] = {}
        while True:
            while len(out) < self.config.num_thread:
                claim = self.claim()
                if claim is None:
                    break
                node, branch = claim
                future = executor.submit(
                    _make_node_in_worker, node.squad, branch[0], node.depth
                )
                out[future] = claim
            if not out:
                return
            done, _ = wait(out, return_when=FIRST_COMPLETED)
            for future in done:
                node, branch = out.pop(future)
                self.complete(node, branch, future.result())
                advance()


class MCTSOptimizer:
    """Chooses transfers by searching the most promising parts of the plan tree."""

    def __init__(self, config: MCTSConfig | None = None) -> None:
        self.config = config if config is not None else MCTSConfig()

    def search(self, request: TransferSearchRequest) -> TransferSearchResult:
        search = _Search(request, self.config)
        if self.config.num_thread <= 1:
            return search.run()
        # Workers are forked when the first node is handed out, inside this
        # block: anything they logged themselves would land in the progress bar.
        with relay_child_logs(), _executor(request, self.config) as executor:
            return search.run(executor)
