"""
Monte Carlo Tree Search over the multi-week transfer-count/chip decision tree.

This is an alternative to the exhaustive tree search in
airsenal.scripts.fill_transfersuggestion_table: rather than enumerating and fully
evaluating every transfer-count/chip branch at every gameweek (intractable once
chips and higher transfer counts are both in play), MCTS selectively expands
whichever branches look promising, using next_week_transfers() to generate
candidate actions and make_best_transfers() to evaluate them - both reused
unchanged from the exhaustive tree.

Unlike classical MCTS, there's no need for a random rollout to a terminal state to
estimate a node's value: get_discounted_squad_score() already gives an analytic
estimate of "hold this squad for the rest of the window", used as the leaf value in
place of a rollout. That estimate gets refined over the search as a node is visited
and expanded further (i.e. as more of "the rest of the window" gets replaced by
actually-searched-for transfers instead of the static hold-still assumption).
"""

import json
import math
import random
from collections.abc import Callable
from copy import deepcopy

from airsenal.framework.optimization_transfers import make_best_transfers
from airsenal.framework.optimization_utils import (
    MAX_FREE_TRANSFERS,
    get_discount_factor,
    get_discounted_squad_score,
    next_week_transfers,
)
from airsenal.framework.squad import Squad

DEFAULT_EXPLORATION_CONSTANT = 5.0


class MCTSNode:
    """One node in the search tree: a squad reachable at a particular depth (gameweek
    index into gameweek_range) via a particular sequence of transfer/chip decisions.

    `strat_dict` mirrors the dict built up by `optimize()` in
    fill_transfersuggestion_table.py - same keys, same int-typed gameweek keys
    internally (only converted to str, matching what the exhaustive tree produces via
    its JSON round-trip, when a final trajectory is extracted for the caller).
    """

    def __init__(
        self,
        squad: Squad,
        depth: int,
        free_transfers: int,
        hit_so_far: int,
        strat_dict: dict,
        parent: "MCTSNode | None" = None,
        action: tuple | None = None,
    ) -> None:
        self.squad = squad
        self.depth = depth
        self.free_transfers = free_transfers
        self.hit_so_far = hit_so_far
        self.strat_dict = strat_dict
        self.parent = parent
        self.action = action
        self.children: dict[tuple, MCTSNode] = {}
        self.untried_actions: list[tuple] | None = None
        self.visits = 0
        self.total_value = 0.0

    @property
    def is_terminal(self) -> bool:
        return self.depth >= len(self.strat_dict["_gameweek_range"])

    @property
    def average_value(self) -> float:
        return self.total_value / self.visits if self.visits else 0.0


def _make_root(
    starting_squad: Squad,
    gameweek_range: list[int],
    num_free_transfers: int,
) -> MCTSNode:
    strat_dict = {
        "total_score": 0,
        "points_per_gw": {},
        "free_transfers": {},
        "num_transfers": {},
        "points_hit": {},
        "discount_factor": {},
        "players_in": {},
        "players_out": {},
        "chips_played": {},
        "bank": {},
        "root_gw": gameweek_range[0],
        # not part of the public strat_dict contract - used internally to know the
        # search horizon without threading it through every function call. Stripped
        # out before a trajectory is returned to the caller.
        "_gameweek_range": gameweek_range,
    }
    return MCTSNode(
        squad=starting_squad,
        depth=0,
        free_transfers=num_free_transfers,
        hit_so_far=0,
        strat_dict=strat_dict,
    )


def _get_untried_actions(
    node: MCTSNode,
    chips_gw_dict: dict,
    max_total_hit: int | None,
    allow_unused_transfers: bool,
    max_free_transfers: int,
) -> list[tuple]:
    gw = node.strat_dict["_gameweek_range"][node.depth]
    return next_week_transfers(
        (node.free_transfers, node.hit_so_far, node.strat_dict),
        gw,
        max_total_hit=max_total_hit,
        allow_unused_transfers=allow_unused_transfers,
        # MCTS only tries a sample of actions per node rather than every one of
        # them, so there's no need to coarsen the action set the way the exhaustive
        # tree does - go back to full granularity (0..max_free_transfers).
        max_opt_transfers=max_free_transfers,
        chips=chips_gw_dict.get(gw, {}),
        max_free_transfers=max_free_transfers,
    )


def _expand_child(
    node: MCTSNode,
    action: tuple,
    tag: str,
    season: str,
    num_iterations: int,
) -> MCTSNode:
    """Build the child of `node` reached by `action`, mirroring the per-node update
    logic in fill_transfersuggestion_table.optimize() exactly (same strat_dict
    fields, same discount/hit accounting) so the two searches produce
    interchangeable output.
    """
    num_transfers, new_free_transfers, total_points_hit, hit_this_gw = action
    gameweek_range = node.strat_dict["_gameweek_range"]
    gw = gameweek_range[node.depth]
    root_gw = node.strat_dict["root_gw"]

    strat_dict = deepcopy(node.strat_dict)

    if isinstance(num_transfers, str):
        if num_transfers.startswith("T"):
            strat_dict["chips_played"][gw] = "triple_captain"
        elif num_transfers.startswith("B"):
            strat_dict["chips_played"][gw] = "bench_boost"
        elif num_transfers == "W":
            strat_dict["chips_played"][gw] = "wildcard"
        elif num_transfers == "F":
            strat_dict["chips_played"][gw] = "free_hit"
    else:
        strat_dict["chips_played"][gw] = None

    new_squad, transfers, points = make_best_transfers(
        num_transfers,
        node.squad,
        tag,
        gameweek_range[node.depth :],
        root_gw,
        season,
        num_iterations,
        None,
    )

    discount_factor = get_discount_factor(root_gw, gw)
    points -= hit_this_gw * discount_factor
    strat_dict["total_score"] += points
    strat_dict["points_per_gw"][gw] = points
    strat_dict["free_transfers"][gw] = new_free_transfers
    strat_dict["num_transfers"][gw] = num_transfers
    strat_dict["points_hit"][gw] = hit_this_gw
    strat_dict["discount_factor"][gw] = discount_factor
    strat_dict["players_in"][gw] = transfers["in"]
    strat_dict["players_out"][gw] = transfers["out"]
    strat_dict["bank"][gw] = new_squad.budget

    child = MCTSNode(
        squad=new_squad,
        depth=node.depth + 1,
        free_transfers=new_free_transfers,
        hit_so_far=total_points_hit,
        strat_dict=strat_dict,
        parent=node,
        action=action,
    )
    node.children[action] = child
    return child


def _value_estimate(node: MCTSNode, tag: str) -> float:
    """Realized score so far (node.strat_dict["total_score"], exactly what the
    exhaustive tree accumulates) plus a static estimate for the unexplored remainder
    of the window - get_discounted_squad_score assuming node.squad is held
    unchanged - standing in for a rollout. 0 remainder once terminal.
    """
    gameweek_range = node.strat_dict["_gameweek_range"]
    realized = node.strat_dict["total_score"]
    if node.is_terminal:
        return realized
    remaining = get_discounted_squad_score(
        node.squad,
        gameweek_range[node.depth :],
        tag,
        root_gw=node.strat_dict["root_gw"],
    )
    return realized + remaining


def _ucb1(child: MCTSNode, parent_visits: int, exploration_constant: float) -> float:
    exploitation = child.average_value
    exploration = exploration_constant * math.sqrt(
        math.log(parent_visits) / child.visits
    )
    return exploitation + exploration


def _select(node: MCTSNode, exploration_constant: float) -> MCTSNode:
    while (
        not node.is_terminal
        and node.untried_actions is not None
        and not node.untried_actions
        and node.children
    ):
        node = max(
            node.children.values(),
            key=lambda c: _ucb1(c, node.visits, exploration_constant),
        )
    return node


def _backpropagate(node: MCTSNode, value: float) -> None:
    current: MCTSNode | None = node
    while current is not None:
        current.visits += 1
        current.total_value += value
        current = current.parent


def run_mcts_tree(
    starting_squad: Squad,
    gameweek_range: list[int],
    tag: str,
    season: str,
    chips_gw_dict: dict,
    num_free_transfers: int,
    n_iterations: int,
    max_total_hit: int | None = None,
    allow_unused_transfers: bool = False,
    max_free_transfers: int = MAX_FREE_TRANSFERS,
    num_iterations: int = 100,
    exploration_constant: float = DEFAULT_EXPLORATION_CONSTANT,
    random_state: int | None = None,
    progress_callback: Callable[[], None] | None = None,
) -> MCTSNode:
    """Run a single MCTS search from `starting_squad`/gameweek_range[0], for
    `n_iterations` select/expand/backpropagate iterations, and return the searched
    root. `num_iterations` (distinct from `n_iterations`, the MCTS budget) is the
    population_size/generations passed through to make_best_transfers' GA search for
    >2-transfer actions - same meaning as elsewhere in the codebase. If given,
    `progress_callback` is called once per iteration (e.g. to drive a progress bar).
    """
    if random_state is not None:
        random.seed(random_state)

    root = _make_root(starting_squad, gameweek_range, num_free_transfers)

    for _ in range(n_iterations):
        node = _select(root, exploration_constant)

        if not node.is_terminal:
            if node.untried_actions is None:
                node.untried_actions = _get_untried_actions(
                    node,
                    chips_gw_dict,
                    max_total_hit,
                    allow_unused_transfers,
                    max_free_transfers,
                )
            if node.untried_actions:
                action = node.untried_actions.pop(
                    random.randrange(len(node.untried_actions))
                )
                node = _expand_child(node, action, tag, season, num_iterations)

        value = _value_estimate(node, tag)
        _backpropagate(node, value)

        if progress_callback is not None:
            progress_callback()

    return root


def extract_best_trajectory(
    root: MCTSNode,
    tag: str,
    season: str,
    chips_gw_dict: dict,
    max_total_hit: int | None,
    allow_unused_transfers: bool,
    max_free_transfers: int,
    num_iterations: int,
) -> dict:
    """Walk the most-visited child at each level to reconstruct one complete
    strategy. If the search never expanded a full path this deeply (possible for a
    branch that looked good near the root but wasn't explored further), complete
    the remainder deterministically by always taking the first available action,
    rather than returning a partial/unusable strategy - the search should always be
    able to return *something* valid, even if it isn't confident about the tail end
    of it.
    """
    node = root
    while not node.is_terminal:
        if node.children:
            node = max(node.children.values(), key=lambda c: c.visits)
            continue
        if node.untried_actions is None:
            node.untried_actions = _get_untried_actions(
                node,
                chips_gw_dict,
                max_total_hit,
                allow_unused_transfers,
                max_free_transfers,
            )
        if not node.untried_actions:
            # no valid action at all from this state (e.g. constraints filtered
            # out even a 0-transfer option) - shouldn't normally happen, but fail
            # clearly rather than crash on an empty pop().
            msg = (
                f"No valid transfer action available at gameweek "
                f"{node.strat_dict['_gameweek_range'][node.depth]} - can't complete "
                "a full strategy."
            )
            raise RuntimeError(msg)
        action = node.untried_actions.pop(0)
        node = _expand_child(node, action, tag, season, num_iterations)

    strat_dict = {k: v for k, v in node.strat_dict.items() if k != "_gameweek_range"}
    # match the string-keyed-by-gameweek shape the exhaustive tree produces via its
    # JSON round-trip (json.dump/json.load in find_best_strat_from_json), which
    # fill_suggestion_table/fill_transaction_table/print_strat etc. all assume.
    return json.loads(json.dumps(strat_dict))
