"""
usage:
python fill_transfersuggestions_table.py --n_gameweeks <num_weeks_ahead>
                                          --num_iterations <num_iterations>
output for each strategy tried is going to be a dict
{ "total_points": <float>,
"points_per_gw": {<gw>: <float>, ...},
"players_sold" : {<gw>: [], ...},
"players_bought" : {<gw>: [], ...}
}
This is done via a recursive tree search, where nodes on the tree do an optimization
for a given number of transfers, then adds some children to the multiprocessing queue
representing 0, 1, 2 transfers for the next gameweek.

"""

import cProfile
import json
import sys
import threading
from collections.abc import Callable
from multiprocessing import Process, Queue
from pathlib import Path

from rich.panel import Panel
from rich.text import Text
from sqlalchemy.orm import Session

from airsenal.core.concurrency import (
    CustomQueue,
    set_multiprocessing_start_method,
)
from airsenal.core.console import console, price_str, progress_bar, table
from airsenal.core.enums import Chip, Position
from airsenal.core.logging import get_logger
from airsenal.db.queries.gameweeks import get_gameweeks_array
from airsenal.db.queries.players import get_player, get_player_name
from airsenal.db.queries.tags import get_latest_prediction_tag
from airsenal.db.session import get_session
from airsenal.domain.season import CURRENT_SEASON
from airsenal.fetch.fpl_api import get_fetcher
from airsenal.optimization.config import GeneticAlgorithmConfig
from airsenal.optimization.moves import ChipSchedule, GameweekMove
from airsenal.optimization.run_squad import fill_initial_squad
from airsenal.optimization.strategy import GameweekOutcome, Strategy
from airsenal.optimization.transfers import (
    get_num_increments,
    make_best_transfers,
)
from airsenal.optimization.utils import (
    MAX_FREE_TRANSFERS,
    check_tag_valid,
    count_expected_outputs,
    fill_suggestion_table,
    fill_transaction_table,
    get_baseline_strat,
    get_discount_factor,
    get_starting_squad,
    next_week_transfers,
)
from airsenal.reporting.discord import post_webhook
from airsenal.reporting.squad_view import formation_table
from airsenal.squad.squad import Squad
from airsenal.squad.state import get_entry_start_gameweek, get_free_transfers

logger = get_logger(__name__)


def optimize(
    queue: CustomQueue,
    pid: Process,
    results: "Queue[Strategy]",
    gameweeks: list[int],
    season: str,
    prediction_tag: str,
    chip_schedule: ChipSchedule,
    max_total_hit: int | None = None,
    allow_unused_transfers: bool = False,
    max_transfers: int = 2,
    num_iterations: int = 100,
    updater: Callable | None = None,
    resetter: Callable | None = None,
    profile: bool = False,
    max_free_transfers: int = MAX_FREE_TRANSFERS,
) -> None:
    """
    Queue is the multiprocessing queue,
    pid is the Process that will execute this func,
    gameweeks will be a list of gameweeks to consider,
    season and prediction_tag are hopefully self-explanatory.

    The rest of the parameters needed for prediction are from the queue.

    Things on the queue will either be None (shutdown sentinel, sent once all
    strategies have been processed), or a tuple:
    (move, free_transfers, hit_so_far, hit_this_gw, squad, strategy).

    `strategy` is None for the root node, which exists only to add children to
    the queue. Finished strategies are put on the `results` queue.
    """
    while True:
        status = queue.get()
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
            strategy,
        ) = status

        if strategy is None:
            # the root node, which exists only to add children to the queue
            new_squad = squad
            strategy = Strategy(root_gameweek=gameweeks[0])
        else:
            if resetter is not None:
                resetter(pid, f"{strategy.label()}-{move.label()}".lstrip("-"))

            # how far down the tree we are, and so which gameweeks are left
            remaining_gameweeks = gameweeks[len(strategy) :]
            gw = remaining_gameweeks[0]
            root_gw = strategy.root_gameweek

            # calculate best transfers to make this gameweek (to maximise points across
            # remaining gameweeks)
            increment = 100 / get_num_increments(move, num_iterations)
            new_squad, transfers, points = make_best_transfers(
                move,
                squad,
                prediction_tag,
                remaining_gameweeks,
                root_gw,
                season,
                num_iterations,
                (updater, increment, pid) if updater is not None else None,
            )

            discount_factor = get_discount_factor(root_gw, gw)
            strategy = strategy.extend(
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

        if len(strategy) >= len(gameweeks):
            results.put(strategy)
            # call function to update the main progress bar
            if updater is not None:
                updater()

            if profile and profiler is not None:
                profiler.dump_stats(
                    f"process_strat_{prediction_tag}_{strategy.label()}.pstat"
                )

        else:
            # add children to the queue
            strategies = next_week_transfers(
                free_transfers,
                hit_so_far,
                strategy.chips_played,
                max_total_hit=max_total_hit,
                allow_unused_transfers=allow_unused_transfers,
                max_opt_transfers=max_transfers,
                chips=chip_schedule.for_gameweek(gameweeks[len(strategy)]),
                max_free_transfers=max_free_transfers,
            )
            for strat in strategies:
                move, free_transfers, hit_so_far, hit_this_gw = strat

                queue.put(
                    (
                        move,
                        free_transfers,
                        hit_so_far,
                        hit_this_gw,
                        new_squad,
                        strategy,
                    )
                )

        # mark this task as done only now that any children have been queued,
        # so queue.join() can't return before the whole tree is processed.
        queue.task_done()


def _wait_for_queue(queue: CustomQueue, procs: list[Process]) -> None:
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
                f"Transfer optimisation stopped: {detail}. The remaining strategies "
                "cannot be evaluated. Re-run with --num-thread 1 to see the error."
            )
            raise RuntimeError(msg)


def is_baseline(strategy: Strategy) -> bool:
    """Whether a strategy makes no transfers and plays no chips."""
    return all(outcome.move == GameweekMove() for outcome in strategy.outcomes)


def save_strategy_dump(strategies: list[Strategy], directory: Path, tag: str) -> None:
    """
    Write every strategy considered to one JSON file, for debugging.

    The search itself keeps strategies in memory; this exists only because
    inspecting the whole tree is occasionally the fastest way to understand a
    surprising suggestion.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"strategies_{tag}.json"
    with path.open("w") as f:
        json.dump([s.to_dict() for s in strategies], f, indent=2)
    logger.info("Wrote %s strategies to %s", len(strategies), path)


def print_optimization_summary(
    strat: Strategy,
    baseline_score: float,
    season: str = CURRENT_SEASON,
    fpl_team_id: int | None = None,
    use_api: bool = False,
    dbsession: Session | None = None,
) -> None:
    """
    Rich-formatted summary of an optimisation result: total score, the
    chosen strategy (transfers/chips/points hits per gameweek), a table of
    the transfers in/out (with purchase/sale prices), and the resulting
    bank balance.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    first_gw, last_gw = strat.gameweeks[0], strat.gameweeks[-1]
    total_score = strat.total_score
    total_hits = strat.total_points_hit

    summary = Text()
    summary.append(
        f"Gameweeks: {first_gw}-{last_gw}\n"
        if first_gw != last_gw
        else f"Gameweek: {first_gw}\n",
        style="bold",
    )
    summary.append(f"Team ID: {fpl_team_id}\n")
    summary.append(f"Baseline Score: {baseline_score:.1f}pts\n")
    summary.append(f"Optimised Score: {total_score:.1f}pts\n", style="bold green")
    summary.append(f"Points Gained: {total_score - baseline_score:+.1f}pts\n")
    summary.append(f"Total Points Hits: -{total_hits}pts", style="red")
    console.print(Panel(summary, title="Optimisation Result", expand=False))

    strategy_table = table(
        "Gameweek",
        "Transfers",
        "Chip",
        "Points Hit",
        "Predicted Score",
        title="Strategy",
    )
    for outcome in strat.outcomes:
        strategy_table.add_row(
            str(outcome.gameweek),
            outcome.move.label(),
            str(outcome.chip) if outcome.chip else "-",
            f"-{outcome.points_hit}pts" if outcome.points_hit else "0pts",
            f"{outcome.undiscounted_points:.1f}pts",
        )
    console.print(strategy_table)

    transfer_table = table(
        "GW",
        "Player Out",
        "Pos",
        "Team",
        "Sale Price",
        "Player In",
        "Pos",
        "Team",
        "Purchase Price",
        title="Transfers",
    )
    any_transfers = False
    squad = get_starting_squad(
        next_gw=first_gw,
        season=season,
        fpl_team_id=fpl_team_id,
        use_api=use_api,
    )
    for outcome in strat.outcomes:
        gw = outcome.gameweek
        for pid_out, pid_in in zip(
            outcome.players_out, outcome.players_in, strict=True
        ):
            any_transfers = True
            out_player = squad.get_player_from_id(pid_out)
            sale_price = squad.get_sell_price_for_player(
                pid_out, use_api=use_api, gameweek=gw, dbsession=dbsession
            )
            squad.remove_player(pid_out, price=sale_price, gameweek=gw)

            in_player_db = get_player(pid_in, dbsession=dbsession)
            purchase_price = in_player_db.price(season, gw) if in_player_db else None
            squad.add_player(
                pid_in,
                price=purchase_price,
                gameweek=gw,
                check_budget=False,
                check_team=False,
                dbsession=dbsession,
            )
            in_name = str(in_player_db) if in_player_db else get_player_name(pid_in)
            transfer_table.add_row(
                str(gw),
                str(out_player),
                out_player.position,
                out_player.team,
                price_str(sale_price),
                in_name,
                in_player_db.position(season) if in_player_db else "-",
                in_player_db.team(season, gw) if in_player_db else "-",
                price_str(purchase_price),
            )
    if any_transfers:
        console.print(transfer_table)
    else:
        console.print(f"{transfer_table.title}: no transfers made.")


def discord_payload(strat: Strategy, lineup: list[str]) -> dict:
    """
    json formated discord webhook content.
    """
    discord_embed = {
        "title": "AIrsenal webhook",
        "description": "Optimum strategy for gameweek(S)"
        f" {','.join(str(gw) for gw in strat.gameweeks)}:",
        "color": 0x35A800,
        "fields": [],
    }
    fields: list[dict] = []
    for outcome in strat.outcomes:
        gw = outcome.gameweek
        fields.append(
            {
                "name": f"GW{gw} chips:",
                "value": f"Chips played:  {outcome.chip}\n",
                "inline": False,
            }
        )
        pin = [str(get_player_name(p)) for p in outcome.players_in]
        pout = [str(get_player_name(p)) for p in outcome.players_out]
        fields.extend(
            [
                {
                    "name": f"GW{gw} transfers out:",
                    "value": "\n".join(pout),
                    "inline": True,
                },
                {
                    "name": f"GW{gw} transfers in:",
                    "value": "\n".join(pin),
                    "inline": True,
                },
            ]
        )
    discord_embed["fields"] = fields
    return {
        "content": "\n".join(lineup),
        "username": "AIrsenal",
        "embeds": [discord_embed],
    }


def print_team_for_next_gw(
    strat: Strategy,
    season: str = CURRENT_SEASON,
    fpl_team_id: int | None = None,
    use_api: bool = False,
) -> Squad:
    """
    Display the team (inc. subs and captain) for the next gameweek
    """
    outcome = strat.outcomes[0]
    next_gw = outcome.gameweek
    t = get_starting_squad(
        next_gw=next_gw, season=season, fpl_team_id=fpl_team_id, use_api=use_api
    )
    for pidout in outcome.players_out:
        t.remove_player(pidout)
    for pidin in outcome.players_in:
        t.add_player(pidin)
    tag = get_latest_prediction_tag(season=season)
    console.print(
        formation_table(
            t,
            tag,
            next_gw,
            bench_boost=outcome.chip is Chip.BENCH_BOOST,
            triple_captain=outcome.chip is Chip.TRIPLE_CAPTAIN,
        )
    )
    return t


def lineup_strings(
    squad: Squad, strategy: Strategy, baseline_score: float, fpl_team_id: int
) -> list[str]:
    """The squad, formatted as Discord markdown."""
    lines = [
        f"__Strategy for Team ID: **{fpl_team_id}**__",
        f"Baseline score: *{int(baseline_score)}*",
        f"Best score: *{int(strategy.total_score)}*",
        "\n__starting 11__",
    ]
    for position in list(Position.back_to_front()):
        lines.append(f"== **{position}** ==\n```")
        for p in squad.players:
            if p.position == position and p.is_starting:
                player_line = f"{p} ({p.team})"
                if p.is_captain:
                    player_line += "(C)"
                elif p.is_vice_captain:
                    player_line += "(VC)"
                lines.append(player_line)
        lines.append("```\n")
    lines += ["__subs__", "```"]
    subs = sorted(
        (p for p in squad.players if not p.is_starting), key=lambda p: p.sub_position
    )
    lines += [f"{p} ({p.team})" for p in subs]
    lines.append("```\n")
    return lines


def new_squad_from_scratch(
    gameweeks: list[int],
    tag: str,
    season: str,
    fpl_team_id: int,
    num_iterations: int,
    chip_gameweeks: dict,
) -> Squad:
    """
    Build a squad from nothing, for the start of a season or a brand new team.

    There is nothing to transfer from, so the transfer search has nothing to do.
    """
    return fill_initial_squad(
        tag=tag,
        gameweeks=gameweeks,
        season=season,
        fpl_team_id=fpl_team_id,
        ga_config=GeneticAlgorithmConfig().scaled(num_iterations),
        chip_gameweeks=chip_gameweeks,
    )


def search_transfer_tree(
    starting_squad: Squad,
    gameweeks: list[int],
    tag: str,
    season: str,
    chip_schedule: ChipSchedule,
    num_free_transfers: int,
    max_total_hit: int | None,
    allow_unused_transfers: bool,
    max_opt_transfers: int,
    num_iterations: int,
    num_thread: int,
    profile: bool,
    max_free_transfers: int = MAX_FREE_TRANSFERS,
) -> list[Strategy]:
    """
    Walk the tree of possible transfer strategies and return every finished one.

    The tree is grown dynamically: a worker that has not reached the end of the
    gameweek window puts its children back on the same queue. That is why the
    queue is joinable and why the workers outlive any single strategy.
    """
    # create a queue that we will add nodes to, and some processes to take
    # things off it
    squeue = CustomQueue()
    # workers put finished strategies here for the parent to compare
    result_queue: Queue[Strategy | None] = Queue()
    procs = []
    # number of nodes in tree will be something like 3^num_weeks unless we allow
    # a "chip" such as wildcard or free hit, in which case it gets complicated
    num_weeks = len(gameweeks)
    num_expected_outputs, baseline_excluded = count_expected_outputs(
        num_weeks,
        next_gw=gameweeks[0],
        free_transfers=num_free_transfers,
        max_total_hit=max_total_hit,
        allow_unused_transfers=allow_unused_transfers,
        max_opt_transfers=max_opt_transfers,
        chip_schedule=chip_schedule,
        max_free_transfers=max_free_transfers,
    )

    with progress_bar(transient=True) as progress:
        # one progress bar per worker process, plus one for overall progress
        worker_tasks = [
            progress.add_task(f"Worker {i}: idle", total=100) for i in range(num_thread)
        ]
        total_task = progress.add_task("Total strategies", total=num_expected_outputs)

        # workers report progress back to this process (which owns the Rich
        # display) via a queue, rather than updating the progress bars directly
        # - the worker processes only ever see a fork-time copy of them.
        progress_queue: Queue = Queue()

        def update_progress(increment: float = 1, index: int | None = None) -> None:
            progress_queue.put(("increment", index, increment))

        def reset_progress(index: int, strategy_string: str) -> None:
            progress_queue.put(("reset", index, strategy_string))

        def consume_progress_updates() -> None:
            while True:
                message = progress_queue.get()
                if message is None:
                    break
                kind, index, value = message
                if kind == "reset":
                    progress.reset(
                        worker_tasks[index], description=f"Worker {index}: {value}"
                    )
                elif index is None:
                    progress.advance(total_task, value)
                else:
                    progress.advance(worker_tasks[index], value)

        progress_thread = threading.Thread(target=consume_progress_updates, daemon=True)
        progress_thread.start()

        # Drain the results queue as strategies finish. A worker blocks once
        # the pipe fills, so this cannot wait until the search is over.
        finished: list[Strategy] = []

        def collect_results() -> None:
            while True:
                strategy = result_queue.get()
                if strategy is None:
                    break
                finished.append(strategy)

        result_thread = threading.Thread(target=collect_results, daemon=True)
        result_thread.start()

        if baseline_excluded:
            # if we are excluding unused transfers the tree may not include the
            # baseline strategy, so compute it here instead.
            baseline_strategy = get_baseline_strat(
                starting_squad, gameweeks, tag, root_gw=gameweeks[0]
            )
            progress.advance(total_task, 1)
        else:
            baseline_strategy = None

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
                    gameweeks,
                    season,
                    tag,
                    chip_schedule,
                    max_total_hit,
                    allow_unused_transfers,
                    max_opt_transfers,
                    num_iterations,
                    update_progress,
                    reset_progress,
                    profile,
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

        # Block until every node in the (dynamically-grown) strategy tree has
        # been processed - i.e. the queue is empty and no worker is still
        # processing an item that could enqueue further children.
        #
        # A bare squeue.join() waits forever if a worker dies mid-task, because
        # the task it had taken is never marked done. That turns any worker
        # crash into a silent hang with the progress bar stopped part-way, and
        # no indication of what went wrong. Watch the workers while waiting.
        _wait_for_queue(squeue, procs)

        progress_queue.put(None)
        progress_thread.join()
        progress.update(total_task, description="Transfer optimization complete")

    # tell each worker to shut down, then wait for them to exit
    for _ in procs:
        squeue.put(None)
    for p in procs:
        p.join()
    result_queue.put(None)
    result_thread.join()

    return finished + ([baseline_strategy] if baseline_strategy else [])


def run_optimization(
    gameweeks: list[int],
    tag: str,
    season: str = CURRENT_SEASON,
    fpl_team_id: int | None = None,
    chip_gameweeks: dict | None = None,
    num_free_transfers: int | None = None,
    max_total_hit: int | None = None,
    allow_unused_transfers: bool = False,
    max_opt_transfers: int = 2,
    num_iterations: int = 100,
    num_thread: int = 4,
    save_strategies: Path | None = None,
    profile: bool = False,
    is_replay: bool = False,  # for replaying seasons
    max_free_transfers: int = MAX_FREE_TRANSFERS,
) -> tuple[Squad, Strategy | None]:
    """
    This is the actual main function that sets up the multiprocessing
    and calls the optimize function for every move/gameweek
    combination, to find the best strategy.
    The chip-related variables e.g. wildcard_week are -1 if that chip
    is not to be played, 0 for 'play it any week', or the gw in which
    it should be played.
    """
    if chip_gameweeks is None:
        chip_gameweeks = {}
    if fpl_team_id is None:
        fpl_team_id = get_fetcher().FPL_TEAM_ID
    if fpl_team_id is None:  # still None after trying env vars
        msg = (
            "fpl_team_id must be set as argument, environment variables or config file."
        )
        raise ValueError(msg)

    # see if we are at the start of a season, or
    if gameweeks[0] == 1 or gameweeks[0] == get_entry_start_gameweek(
        fpl_team_id, fetcher=get_fetcher()
    ):
        logger.info(
            "This is the start of the season or a new team - will make a squad "
            "from scratch"
        )
        return new_squad_from_scratch(
            gameweeks, tag, season, fpl_team_id, num_iterations, chip_gameweeks
        ), None

    with console.status("Optimising transfers..."):
        logger.info("Running optimization with fpl_team_id %s", fpl_team_id)
        use_api = season == CURRENT_SEASON and not is_replay
        try:
            starting_squad = get_starting_squad(
                next_gw=gameweeks[0],
                season=season,
                fpl_team_id=fpl_team_id,
                use_api=use_api,
                fetcher=get_fetcher(),
            )
        except (ValueError, TypeError):
            # first week for this squad?
            logger.warning(
                "No existing squad or transfers found for team_id %s", fpl_team_id
            )
            logger.info("Will suggest a new starting squad:")
            return new_squad_from_scratch(
                gameweeks, tag, season, fpl_team_id, num_iterations, chip_gameweeks
            ), None
        # if we got to here, we can assume we are optimizing an existing squad.

        # How many free transfers are we starting with?
        if num_free_transfers is None:
            num_free_transfers = get_free_transfers(
                fpl_team_id,
                gameweeks[0],
                season=season,
                fetcher=get_fetcher(),
                is_replay=is_replay,
            )
        logger.info("Starting with %s free transfers", num_free_transfers)

        # Work out what chips we definitely or possibly will play in each gw
        chip_schedule = ChipSchedule.from_weeks(gameweeks, chip_gameweeks)

        finished = search_transfer_tree(
            starting_squad,
            gameweeks,
            tag,
            season,
            chip_schedule,
            num_free_transfers,
            max_total_hit,
            allow_unused_transfers,
            max_opt_transfers,
            num_iterations,
            num_thread,
            profile,
            max_free_transfers,
        )

        if save_strategies is not None:
            save_strategy_dump(finished, save_strategies, tag)
        if not finished:
            msg = "Failed to find a strategy!"
            raise ValueError(msg)
        best_strategy = max(finished, key=lambda s: s.total_score)

        # the baseline is the strategy that makes no transfers in any gameweek
        baseline = next((s for s in finished if is_baseline(s)), None)
        if baseline is None:
            logger.warning("No baseline strategy was evaluated")
        baseline_score = baseline.total_score if baseline is not None else 0.0
        fill_suggestion_table(baseline_score, best_strategy, season, fpl_team_id)
        if is_replay:
            # simulating a previous season, so imitate applying transfers by adding
            # the suggestions to the Transaction table
            fill_transaction_table(
                starting_squad, best_strategy, season, fpl_team_id, tag
            )

    console.print()

    print_optimization_summary(
        best_strategy,
        baseline_score,
        season=season,
        fpl_team_id=fpl_team_id,
        use_api=use_api,
    )
    best_squad = print_team_for_next_gw(
        best_strategy, season=season, fpl_team_id=fpl_team_id, use_api=use_api
    )

    post_webhook(
        discord_payload(
            best_strategy,
            lineup_strings(best_squad, best_strategy, baseline_score, fpl_team_id),
        )
    )

    return best_squad, best_strategy


def sanity_check_args(
    n_gameweeks: int | None,
    gameweek_start: int | None,
    gameweek_end: int | None,
    num_free_transfers: int | None,
) -> bool:
    """
    Check that command-line arguments are self-consistent.
    """
    if n_gameweeks and (gameweek_start or gameweek_end):
        msg = "Please only specify n_gameweeks OR gameweek_start/end"
        raise RuntimeError(msg)
    if (gameweek_start and not gameweek_end) or (gameweek_end and not gameweek_start):
        msg = "Need to specify both gameweek_start and gameweek_end"
        raise RuntimeError(msg)
    if num_free_transfers and num_free_transfers not in range(6):
        msg = "Number of free transfers must be 0 to 5"
        raise RuntimeError(msg)
    return True


def run_transfer_optimization(
    n_gameweeks: int | None,
    gameweek_start: int | None,
    gameweek_end: int | None,
    tag: str | None,
    wildcard_week: int,
    free_hit_week: int,
    triple_captain_week: int,
    bench_boost_week: int,
    num_free_transfers: int | None,
    max_hit: int,
    allow_unused: bool,
    max_transfers: int,
    num_iterations: int,
    num_thread: int,
    season: str,
    profile: bool,
    fpl_team_id: int | None,
    is_replay: bool,
    save_strategies: Path | None = None,
) -> None:
    """Run transfer optimization for a gameweek range."""
    sanity_check_args(
        n_gameweeks,
        gameweek_start,
        gameweek_end,
        num_free_transfers,
    )
    gameweeks = get_gameweeks_array(
        n_gameweeks=n_gameweeks,
        gameweek_start=gameweek_start,
        gameweek_end=gameweek_end,
        season=season,
    )
    tag = tag or get_latest_prediction_tag(season=season)
    chip_gameweeks = {
        "wildcard": wildcard_week,
        "free_hit": free_hit_week,
        "triple_captain": triple_captain_week,
        "bench_boost": bench_boost_week,
    }

    if not check_tag_valid(tag, gameweeks, season=season):
        logger.error(
            "Database does not contain predictions for all the specified "
            "optimsation gameweeks. Please run 'airsenal_run_prediction' first "
            "with the same input gameweeks and season you specified here."
        )
        sys.exit(1)

    set_multiprocessing_start_method()

    run_optimization(
        gameweeks,
        tag,
        season=season,
        fpl_team_id=fpl_team_id,
        chip_gameweeks=chip_gameweeks,
        num_free_transfers=num_free_transfers,
        max_total_hit=max_hit,
        allow_unused_transfers=allow_unused,
        max_opt_transfers=max_transfers,
        num_iterations=num_iterations,
        num_thread=num_thread,
        save_strategies=save_strategies,
        profile=profile,
        is_replay=is_replay,
    )
