"""
The multiprocess plan-tree search.

The algorithm behind `airsenal optimize transfers`: enumerate every legal move
for every gameweek in the window, score each resulting squad, and keep the best
whole-window plan.

Only the search itself lives here. Fetching the starting squad, persisting the
suggestions and printing the summary stay in `run_transfers.py`, so substituting
a different search does not mean reimplementing any of that. The progress display
is the exception: a bar per worker sized by `count_expected_outputs` only makes
sense for a forked tree walk.
"""

import cProfile
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Process, Queue
from typing import Literal

from airsenal.core.concurrency import CustomQueue, StallWatchdog
from airsenal.core.console import progress_bar
from airsenal.core.logging import get_logger, relay_child_logs
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.game.enums import Chip
from airsenal.game.scoring import MAX_FREE_TRANSFERS
from airsenal.optimization.moves import (
    NO_CHIPS,
    ChipSchedule,
    GameweekChips,
    GameweekMove,
    calc_free_transfers,
    calc_points_hit,
)
from airsenal.optimization.plan import (
    GameweekOutcome,
    Plan,
    TransferSearchResult,
    baseline_plan,
)
from airsenal.optimization.protocols import (
    ProgressResetter,
    ProgressUpdater,
    TransferRequest,
    TransferSearchRequest,
    strategy_total,
)
from airsenal.optimization.squad_score import (
    get_discount_factor,
    get_discounted_squad_score,
)
from airsenal.optimization.strategies import (
    DEFAULT_STRATEGIES,
    StrategySet,
    TransferStrategy,
)
from airsenal.squad.squad import Squad

logger = get_logger(__name__)

# What the plan-tree queue carries: either a node still to expand, or the
# shutdown sentinel. A node is (move, free_transfers, hit_so_far, hit_this_gw,
# squad, plan), where `plan` is None only for the root.
PlanNode = tuple[GameweekMove, int, int, int, Squad, "Plan | None"]
QueueItem = PlanNode | None

# What a worker sends back to the progress display: count off one step of a bar
# (a candidate squad on a worker's bar, or a finished plan on the total), or
# restart one worker's bar for a new plan, which sizes it to the number of
# candidate squads that plan will consider.
ProgressMessage = (
    tuple[Literal["advance"], int | None]
    | tuple[Literal["reset"], int, str, int | None]
)


# ----------------------- enumerating the tree's branches -------------------
#
# Which moves are legal in the next gameweek, and how many whole-window plans
# that adds up to. Only a tree walk asks either question, and the second is
# only asked to size a progress bar, so both live with the search rather than
# with `GameweekMove` - which is then a pure description of a move, with no
# reason to reach for the database.


def next_week_transfers(
    free_transfers: int,
    hit_so_far: int,
    chips_played: Iterable[Chip | None] = (),
    max_total_hit: int | None = None,
    allow_unused_transfers: bool = True,
    max_opt_transfers: int = 2,
    chips: GameweekChips | None = None,
    max_free_transfers: int = MAX_FREE_TRANSFERS,
) -> list[tuple[GameweekMove, int, int, int]]:
    """
    The moves - transfers, and any chip played - a strategy may make next gameweek.

    One node of the tree expanded into its children.

    Args:
        free_transfers: Available going into the gameweek.
        hit_so_far: Points hit this strategy has taken up to but not including
            this gameweek.
        chips_played: Chips this strategy has already used, so they are not
            offered again.
        allow_unused_transfers: If False and a free transfer would otherwise be
            lost, making none is not offered - which can exclude the baseline
            strategy, so a caller that needs it re-adds it.
        max_free_transfers: The most free transfers the game rules let a manager
            bank.

    Returns:
        Per move: the move, the free transfers it leaves for the gameweek after,
        the total hit including this gameweek, and the hit this gameweek alone.
    """
    chips = chips if chips is not None else NO_CHIPS
    chips_played = list(chips_played)

    if not allow_unused_transfers and free_transfers == max_free_transfers:
        # Force at least 1 free transfer if a free transfer will be lost otherwise.
        # NOTE: This can cause the baseline strategy to be excluded. Re-add it outside
        # this function in that case.
        ft_choices = list(range(1, max_opt_transfers + 1))
    else:
        ft_choices = list(range(max_opt_transfers + 1))

    if max_total_hit is not None:
        ft_choices = [
            nt
            for nt in ft_choices
            if hit_so_far + calc_points_hit(GameweekMove(nt), free_transfers)
            <= max_total_hit
        ]

    # if we are definitely going to play a wildcard or free_hit deal with that first
    if chips.chip_to_play is not None and chips.chip_to_play.rebuilds_squad:
        moves = [GameweekMove(chip=chips.chip_to_play)]
    elif chips.chip_to_play is not None:
        # triple captain or bench boost - we can still do ft_choices transfers
        moves = [GameweekMove(nt, chips.chip_to_play) for nt in ft_choices]
    else:
        # no chip definitely played, but some might be allowed
        moves = [GameweekMove(nt) for nt in ft_choices]
        for chip in (Chip.WILDCARD, Chip.FREE_HIT):
            if chips.allows(chip, chips_played):
                moves.append(GameweekMove(chip=chip))
        for chip in (Chip.BENCH_BOOST, Chip.TRIPLE_CAPTAIN):
            if chips.allows(chip, chips_played):
                moves += [GameweekMove(nt, chip) for nt in ft_choices]

    hit_this_gw = [calc_points_hit(move, free_transfers) for move in moves]
    total_points_hit = [hit_so_far + hit for hit in hit_this_gw]
    new_ft_available = [
        calc_free_transfers(move, free_transfers, max_free_transfers) for move in moves
    ]

    return list(
        zip(moves, new_ft_available, total_points_hit, hit_this_gw, strict=True)
    )


def count_expected_outputs(
    gw_ahead: int,
    next_gw: int | None = None,
    free_transfers: int = 1,
    max_total_hit: int | None = None,
    allow_unused_transfers: bool = True,
    max_opt_transfers: int = 2,
    chip_schedule: ChipSchedule | None = None,
    max_free_transfers: int = MAX_FREE_TRANSFERS,
) -> tuple[int, bool]:
    """
    Count the strategies a search over `gw_ahead` gameweeks will visit.

    Counted rather than enumerated, because this is what sizes the progress bar
    before the tree is built. Each chip may be played at most once.

    Args:
        max_total_hit: Points that may be spent on transfers across the whole
            window; None for no limit.
        allow_unused_transfers: If False, strategies that leave a free transfer
            unused - making none while two are available - are not counted.

    Returns:
        How many strategies will be computed, and whether the baseline strategy
        falls outside the main tree and so has to be computed separately, which
        `allow_unused_transfers=False` can cause. The count includes the
        baseline either way.
    """
    next_gw = next_gameweek() if next_gw is None else next_gw
    chip_schedule = chip_schedule if chip_schedule is not None else ChipSchedule()

    # (free transfers, points hit so far, moves made) - the moves are all that is
    # needed to count branches and to spot the do-nothing baseline among them
    branches: list[tuple[int, int, tuple[GameweekMove, ...]]] = [
        (free_transfers, 0, ())
    ]

    for gw in range(next_gw, next_gw + gw_ahead):
        new_branches = []
        for ft, hit, moves in branches:
            possibilities = next_week_transfers(
                ft,
                hit,
                [move.chip for move in moves],
                max_total_hit=max_total_hit,
                max_opt_transfers=max_opt_transfers,
                allow_unused_transfers=allow_unused_transfers,
                chips=chip_schedule.for_gameweek(gw),
                max_free_transfers=max_free_transfers,
            )
            new_branches += [
                (new_ft, new_hit, (*moves, move))
                for move, new_ft, new_hit, _ in possibilities
            ]
        branches = new_branches

    # if allow_unused_transfers is False the baseline of no transfers can be removed
    # above. Check whether the 1st strategy is the baseline and if not add it back in.
    #
    # `not branches` is the case where the constraints admit no move at all -
    # --max-transfers 0 with unused transfers disallowed and a full bank of free
    # transfers, which the search reaches by forcing at least one transfer and then
    # being allowed none. Doing nothing is still a plan, so the answer is the
    # baseline on its own rather than an IndexError from inside the progress sizing.
    baseline_moves = (GameweekMove(),) * gw_ahead
    baseline_excluded = not branches or branches[0][2] != baseline_moves
    if baseline_excluded:
        branches.insert(0, (max_free_transfers, 0, baseline_moves))

    return len(branches), baseline_excluded


@dataclass(frozen=True)
class TreeSearchConfig:
    """
    Settings for the tree search itself, as opposed to the problem it is solving.

    How the algorithm works rather than what it is asked to do, which is why
    these are here and not on `TransferSearchRequest`.
    """

    num_thread: int = 4
    num_iterations: int = 100
    profile: bool = False
    strategies: StrategySet = field(default_factory=lambda: DEFAULT_STRATEGIES)


def _make_best_transfers(
    request: TransferRequest, strategy: TransferStrategy
) -> tuple[Squad, dict[str, list[int]], float]:
    """
    Make this gameweek's move and score the squad it leaves.

    Returns the squad, the transfers as {"in": [player_ids], "out":
    [player_ids]}, and the points it is expected to score next gameweek.

    One node of the tree, which is why it is here: the strategy decides, and this
    scores what it came back with the same way every other node is scored.
    """
    proposal = strategy.propose(request)

    points = get_discounted_squad_score(
        proposal.squad,
        [request.transfer_gameweek],
        request.tag,
        root_gw=request.root_gw,
        bench_boost_gw=request.bench_boost_gw,
        triple_captain_gw=request.triple_captain_gw,
        sub_weights=request.scoring.sub_weights,
    )

    # A free hit is reverted after the gameweek it is played in, so the squad
    # that carries on to the next gameweek is the one we started with.
    resulting_squad = proposal.squad if request.move.carry_forward else request.squad
    return resulting_squad, proposal.as_transfer_dict(), points


def optimize(
    queue: CustomQueue[QueueItem],
    pid: int,
    results: "Queue[Plan]",
    request: TransferSearchRequest,
    config: "TreeSearchConfig",
    updater: ProgressUpdater | None = None,
    resetter: ProgressResetter | None = None,
) -> None:
    """
    Expand nodes of the plan tree until the queue is drained.

    `queue` is the multiprocessing queue and `pid` identifies the Process running
    this. The problem and the settings arrive as two frozen dataclasses.

    Things on the queue will either be None (shutdown sentinel, sent once all
    plans have been processed), or a tuple:
    (move, free_transfers, hit_so_far, hit_this_gw, squad, plan).

    `plan` is None for the root node, which exists only to add children to
    the queue. Finished plans are put on the `results` queue.
    """
    # A worker that wedges - on a lock inherited across fork, say - stays alive,
    # so the parent cannot distinguish it from one doing slow work and the run
    # just stops. Have the worker say where it stopped, from the inside.
    watchdog = StallWatchdog(f"worker-{pid}")
    watchdog.start()

    gameweeks = request.gameweeks
    season = request.season
    prediction_tag = request.tag
    chip_schedule = request.chip_schedule
    constraints = request.constraints
    num_iterations = config.num_iterations
    profile = config.profile
    strategy_set = config.strategies
    squad_optimizer = request.squad_optimizer

    while True:
        watchdog.idle()
        status = queue.get()
        watchdog.busy()
        if status is None:
            break

        # now assume we have set of parameters to do an optimization
        # from the queue.

        # turn on the profiler if requested
        if profile:
            profiler = cProfile.Profile()
            profiler.enable()
        else:
            profiler = None

        (
            move,
            free_transfers,
            hit_so_far,
            hit_this_gw,
            squad,
            plan,
        ) = status

        if plan is None:
            # the root node, which exists only to add children to the queue
            new_squad = squad
            plan = Plan(root_gameweek=gameweeks[0])
        else:
            # how far down the tree we are, and so which gameweeks are left
            remaining_gameweeks = gameweeks[len(plan) :]
            gw = remaining_gameweeks[0]
            root_gw = plan.root_gameweek

            # One request, used both to size the worker's bar and to do the work,
            # so the two cannot disagree about what is being asked for.
            transfer_request = TransferRequest(
                move=move,
                squad=squad,
                tag=prediction_tag,
                gameweeks=remaining_gameweeks,
                root_gw=root_gw,
                season=season,
                num_iterations=num_iterations,
                scoring=request.scoring,
                squad_optimizer=squad_optimizer,
                progress=partial(updater, pid) if updater is not None else None,
            )
            strategy = strategy_set.create(move)

            if resetter is not None:
                resetter(
                    pid,
                    f"{plan.label()}-{move.label()}".lstrip("-"),
                    strategy_total(strategy, transfer_request),
                )

            # calculate best transfers to make this gameweek (to maximise points across
            # remaining gameweeks)
            new_squad, transfers, points = _make_best_transfers(
                transfer_request, strategy
            )

            discount_factor = get_discount_factor(root_gw, gw)
            plan = plan.extend(
                GameweekOutcome(
                    gameweek=gw,
                    move=move,
                    points=points - hit_this_gw * discount_factor,
                    discount_factor=discount_factor,
                    points_hit=hit_this_gw,
                    free_transfers=free_transfers,
                    players_in=tuple(transfers["in"]),
                    players_out=tuple(transfers["out"]),
                    bank=new_squad.budget,
                )
            )

        if len(plan) >= len(gameweeks):
            results.put(plan)
            # call function to update the main progress bar
            if updater is not None:
                updater()

            if profile and profiler is not None:
                profiler.dump_stats(
                    f"process_plan_{prediction_tag}_{plan.label()}.pstat"
                )

        else:
            # add children to the queue
            branches = next_week_transfers(
                free_transfers,
                hit_so_far,
                plan.chips_played,
                max_total_hit=constraints.max_total_hit,
                allow_unused_transfers=constraints.allow_unused_transfers,
                max_opt_transfers=constraints.max_opt_transfers,
                chips=chip_schedule.for_gameweek(gameweeks[len(plan)]),
                max_free_transfers=constraints.max_free_transfers,
            )
            for branch in branches:
                move, free_transfers, hit_so_far, hit_this_gw = branch

                queue.put(
                    (
                        move,
                        free_transfers,
                        hit_so_far,
                        hit_this_gw,
                        new_squad,
                        plan,
                    )
                )

        # mark this task as done only now that any children have been queued,
        # so queue.join() can't return before the whole tree is processed.
        queue.task_done()


def _wait_for_queue(queue: CustomQueue[QueueItem], procs: list[Process]) -> None:
    """
    Wait for every queued task to be marked done, failing if a worker dies first.

    JoinableQueue.join() has no timeout and no interest in whether the workers are
    still alive, so a worker that raises leaves the parent blocked indefinitely.
    Workers are only asked to exit after this returns, so any worker that has
    already exited here has died prematurely.
    """
    joiner = threading.Thread(target=queue.join, daemon=True)
    joiner.start()
    while joiner.is_alive():
        joiner.join(timeout=2)
        if not joiner.is_alive():
            return
        dead = [(i, p.exitcode) for i, p in enumerate(procs) if p.exitcode is not None]
        if dead:
            detail = ", ".join(f"worker {i} exited with {code}" for i, code in dead)
            msg = (
                f"Transfer optimisation stopped: {detail}. The remaining plans "
                "cannot be evaluated. Re-run with --num-thread 1 to see the error."
            )
            raise RuntimeError(msg)


def search_transfer_tree(
    request: TransferSearchRequest, config: "TreeSearchConfig"
) -> list[Plan]:
    """
    Walk the tree of possible transfer plans and return every finished one.

    The tree is grown dynamically: a worker that has not reached the end of the
    gameweek window puts its children back on the same queue. That is why the
    queue is joinable and why the workers outlive any single plan.
    """
    starting_squad = request.starting_squad
    gameweeks = request.gameweeks
    tag = request.tag
    num_free_transfers = request.num_free_transfers
    constraints = request.constraints
    num_thread = config.num_thread
    # create a queue that we will add nodes to, and some processes to take
    # things off it
    squeue: CustomQueue[QueueItem] = CustomQueue()
    # workers put finished plans here for the parent to compare
    result_queue: Queue[Plan | None] = Queue()
    procs = []
    # number of nodes in tree will be something like 3^num_weeks unless we allow
    # a "chip" such as wildcard or free hit, in which case it gets complicated
    num_weeks = len(gameweeks)
    num_expected_outputs, baseline_excluded = count_expected_outputs(
        num_weeks,
        next_gw=gameweeks[0],
        free_transfers=num_free_transfers,
        max_total_hit=constraints.max_total_hit,
        allow_unused_transfers=constraints.allow_unused_transfers,
        max_opt_transfers=constraints.max_opt_transfers,
        chip_schedule=request.chip_schedule,
        max_free_transfers=constraints.max_free_transfers,
    )

    # The workers are forked below, while the progress bars own the terminal:
    # anything they logged themselves would land inside the display.
    with relay_child_logs(), progress_bar(transient=True) as progress:
        # one progress bar per worker process, plus one for overall progress. A
        # worker's total is only known once it starts a plan: different
        # plans consider very different numbers of candidate squads.
        worker_tasks = [
            progress.add_task(f"Worker {i}: idle", total=None)
            for i in range(num_thread)
        ]
        total_task = progress.add_task("Total plans", total=num_expected_outputs)

        # workers report progress back to this process (which owns the Rich
        # display) via a queue, rather than updating the progress bars directly
        # - the worker processes only ever see a fork-time copy of them.
        progress_queue: Queue[ProgressMessage | None] = Queue()

        def update_progress(index: int | None = None) -> None:
            progress_queue.put(("advance", index))

        def reset_progress(index: int, plan_label: str, num_steps: int | None) -> None:
            progress_queue.put(("reset", index, plan_label, num_steps))

        def consume_progress_updates() -> None:
            while True:
                message = progress_queue.get()
                if message is None:
                    break
                if message[0] == "reset":
                    _, worker, label, num_steps = message
                    progress.reset(
                        worker_tasks[worker],
                        total=num_steps,
                        description=f"Worker {worker}: {label}",
                    )
                else:
                    _, index = message
                    task = total_task if index is None else worker_tasks[index]
                    progress.advance(task)

        progress_thread = threading.Thread(target=consume_progress_updates, daemon=True)
        progress_thread.start()

        # Drain the results queue as plans finish. A worker blocks once
        # the pipe fills, so this cannot wait until the search is over.
        finished: list[Plan] = []

        def collect_results() -> None:
            while True:
                plan = result_queue.get()
                if plan is None:
                    break
                finished.append(plan)

        result_thread = threading.Thread(target=collect_results, daemon=True)
        result_thread.start()

        if baseline_excluded:
            # if we are excluding unused transfers the tree may not include the
            # baseline plan, so compute it here instead.
            baseline = baseline_plan(
                starting_squad,
                gameweeks,
                tag,
                root_gw=gameweeks[0],
                sub_weights=request.scoring.sub_weights,
            )
            progress.advance(total_task, 1)
        else:
            baseline = None

        # One worker per thread, each draining `squeue` until it sees the
        # shutdown sentinel. Everything a worker needs beyond the node it pops
        # is in `request` and `config`, both frozen.
        for i in range(num_thread):
            processor = Process(
                target=optimize,
                args=(
                    squeue,
                    i,
                    result_queue,
                    request,
                    config,
                    update_progress,
                    reset_progress,
                ),
            )
            processor.daemon = True
            processor.start()
            procs.append(processor)
        # add starting node to the queue
        squeue.put(
            (
                GameweekMove(),
                num_free_transfers,
                0,
                0,
                starting_squad,
                None,
            )
        )

        # Block until every node in the (dynamically-grown) plan tree has
        # been processed - i.e. the queue is empty and no worker is still
        # processing an item that could enqueue further children.
        #
        # A bare squeue.join() waits forever if a worker dies mid-task, because
        # the task it had taken is never marked done. That turns any worker
        # crash into a silent hang with the progress bar stopped part-way, and
        # no indication of what went wrong. Watch the workers while waiting.
        _wait_for_queue(squeue, procs)

        # Shut the workers down before the progress consumer, not after. A
        # worker cannot exit until its queue feeder threads have flushed, and
        # stopping the consumer first leaves nothing draining progress_queue
        # while p.join() waits: a worker with a full pipe can then never finish
        # writing, and the join waits for ever. Joining first costs nothing and
        # removes the window.
        for _ in procs:
            squeue.put(None)
        for p in procs:
            p.join()

        # No worker is left to write to either queue, and everything they wrote
        # has been flushed, so these sentinels cannot overtake real messages.
        progress_queue.put(None)
        progress_thread.join()
        progress.update(total_task, description="Transfer optimization complete")

    result_queue.put(None)
    result_thread.join()

    # `is not None`, not truthiness: Plan defines __len__, so a plan covering no
    # gameweeks is falsy and would be dropped here rather than reported.
    return finished + ([baseline] if baseline is not None else [])


class TreeSearchOptimizer:
    """Chooses transfers by walking the whole tree of legal plans."""

    def __init__(self, config: TreeSearchConfig | None = None) -> None:
        self.config = config if config is not None else TreeSearchConfig()

    def search(self, request: TransferSearchRequest) -> TransferSearchResult:
        return TransferSearchResult.from_plans(
            search_transfer_tree(request, self.config)
        )
