"""
The multiprocess plan-tree search.

The algorithm behind `airsenal optimize transfers`: enumerate every legal move
for every gameweek in the window, score each resulting squad, and keep the best
whole-window plan. It is deliberately the *only* thing that lives behind the
`TransferOptimizer` interface - fetching the starting squad, persisting the
suggestions and printing the summary all stay in `run_transfers.py`, so
substituting a different search does not mean reimplementing any of that.

The progress display does belong here, though: only the algorithm knows what its
steps are, and a bar per worker sized by `count_expected_outputs` means nothing
to a search that is not a forked tree walk.
"""

import cProfile
import threading
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Process, Queue
from typing import Literal

from airsenal.core.concurrency import CustomQueue, StallWatchdog
from airsenal.core.console import progress_bar
from airsenal.core.logging import get_logger, relay_child_logs
from airsenal.optimization.moves import (
    GameweekMove,
    count_expected_outputs,
    next_week_transfers,
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
    Make this gameweek's move, returning the resulting squad, the transfers made
    as {"in": [player_ids], "out": [player_ids]}, and the points it is expected
    to score next gameweek.

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
        sub_weights=request.sub_weights,
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

    Queue is the multiprocessing queue and pid is the Process that will execute
    this func. The problem and the settings arrive as two frozen dataclasses:
    they used to be fourteen positional elements of a tuple handed to Process,
    one short of this signature, which is how `max_free_transfers` came to be
    silently dropped on the way to every worker.

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
                sub_weights=request.scoring.sub_weights.as_dict(),
            )
            progress.advance(total_task, 1)
        else:
            baseline = None

        # Add Processes to run the target 'optimize' function.
        # This target function needs to know:
        #  the move (transfers and chip) to make
        #  current_team (list of player_ids)
        #  transfer_dict {"gw":<gw>,"in":[],"out":[]}
        #  total_score
        #  num_free_transfers
        #  budget
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

    return finished + ([baseline] if baseline else [])


class TreeSearchOptimizer:
    """Chooses transfers by walking the whole tree of legal plans."""

    def __init__(self, config: TreeSearchConfig | None = None) -> None:
        self.config = config if config is not None else TreeSearchConfig()

    def search(self, request: TransferSearchRequest) -> TransferSearchResult:
        return TransferSearchResult.from_plans(
            search_transfer_tree(request, self.config)
        )
