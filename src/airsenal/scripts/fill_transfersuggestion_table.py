"""
usage:
python fill_transfersuggestions_table.py --weeks_ahead <num_weeks_ahead>
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
import os
import shutil
import sys
import threading
from collections.abc import Callable
from multiprocessing import Process, Queue

import regex as re
import requests
from rich.panel import Panel
from rich.text import Text
from sqlalchemy.orm import Session

from airsenal.core.concurrency import (
    CustomQueue,
    set_multiprocessing_start_method,
)
from airsenal.core.env import AIRSENAL_HOME
from airsenal.core.output import (
    console,
    get_logger,
    price_str,
    progress_bar,
    table,
)
from airsenal.db.session import get_session
from airsenal.fetch.fpl_api import get_fetcher
from airsenal.framework.optimization_transfers import make_best_transfers
from airsenal.framework.optimization_utils import (
    MAX_FREE_TRANSFERS,
    check_tag_valid,
    count_expected_outputs,
    fill_suggestion_table,
    fill_transaction_table,
    get_baseline_strat,
    get_discount_factor,
    get_num_increments,
    get_starting_squad,
    next_week_transfers,
)
from airsenal.framework.squad import Squad
from airsenal.framework.utils import (
    CURRENT_SEASON,
    get_entry_start_gameweek,
    get_free_transfers,
    get_gameweeks_array,
    get_latest_prediction_tag,
    get_player,
    get_player_name,
)
from airsenal.scripts.squad_builder import fill_initial_squad

logger = get_logger(__name__)

OUTPUT_DIR = os.path.join(AIRSENAL_HOME, "airsopt")


def optimize(
    queue: CustomQueue,
    pid: Process,
    gameweek_range: list[int],
    season: str,
    pred_tag: str,
    chips_gw_dict: dict,
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
    (
     num_transfers,
     free_transfers,
     hit_so_far,
     current_team,
     strat_dict,
     strat_id
    )
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
            num_transfers,
            free_transfers,
            hit_so_far,
            hit_this_gw,
            squad,
            strat_dict,
            sid,
        ) = status
        # num_transfers will be 0, 1, 2, OR 'W' or 'F', OR 'T0', T1', 'T2',
        # OR 'B0', 'B1', or 'B2' (the latter six represent triple captain or
        # bench boost along with 0, 1, or 2 transfers).

        # sid (status id) is just a string e.g. "0-0-2" representing how many
        # transfers to be made in each gameweek.
        # Only exception is the root node, where sid is "starting" - this
        # node only exists to add children to the queue.

        if sid == "starting":
            sid = ""
            depth = 0
            strat_dict["total_score"] = 0
            strat_dict["points_per_gw"] = {}
            strat_dict["free_transfers"] = {}
            strat_dict["num_transfers"] = {}
            strat_dict["points_hit"] = {}
            strat_dict["discount_factor"] = {}
            strat_dict["players_in"] = {}
            strat_dict["players_out"] = {}
            strat_dict["chips_played"] = {}
            strat_dict["bank"] = {}
            new_squad = squad
            gw = gameweek_range[0] - 1
            strat_dict["root_gw"] = gameweek_range[0]
        else:
            if len(sid) > 0:
                sid += "-"
            sid += str(num_transfers)
            if resetter is not None:
                resetter(pid, sid)

            # work out what gameweek we're in and how far down the tree we are.
            depth = len(strat_dict["points_per_gw"])

            # gameweeks from this point in strategy to end of window
            gameweeks = gameweek_range[depth:]

            # upcoming gameweek:
            gw = gameweeks[0]
            root_gw = strat_dict["root_gw"]

            # check whether we're playing a chip this gameweek
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

            # calculate best transfers to make this gameweek (to maximise points across
            # remaining gameweeks)
            num_increments_for_updater = get_num_increments(
                num_transfers, num_iterations
            )
            increment = 100 / num_increments_for_updater
            new_squad, transfers, points = make_best_transfers(
                num_transfers,
                squad,
                pred_tag,
                gameweeks,
                root_gw,
                season,
                num_iterations,
                (updater, increment, pid) if updater is not None else None,
            )

            discount_factor = get_discount_factor(root_gw, gw)
            points -= hit_this_gw * discount_factor
            strat_dict["total_score"] += points
            strat_dict["points_per_gw"][gw] = points
            strat_dict["free_transfers"][gw] = free_transfers
            strat_dict["num_transfers"][gw] = num_transfers
            strat_dict["points_hit"][gw] = hit_this_gw
            strat_dict["discount_factor"][gw] = discount_factor
            strat_dict["players_in"][gw] = transfers["in"]
            strat_dict["players_out"][gw] = transfers["out"]
            strat_dict["bank"][gw] = new_squad.budget
            depth += 1

        if depth >= len(gameweek_range):
            with open(
                os.path.join(OUTPUT_DIR, f"strategy_{pred_tag}_{sid}.json"),
                "w",
            ) as outfile:
                json.dump(strat_dict, outfile)
            # call function to update the main progress bar
            if updater is not None:
                updater()

            if profile and profiler is not None:
                profiler.dump_stats(f"process_strat_{pred_tag}_{sid}.pstat")

        else:
            # add children to the queue
            strategies = next_week_transfers(
                (free_transfers, hit_so_far, strat_dict),
                max_total_hit=max_total_hit,
                allow_unused_transfers=allow_unused_transfers,
                max_opt_transfers=max_transfers,
                chips=chips_gw_dict[gw + 1],
                max_free_transfers=max_free_transfers,
            )
            for strat in strategies:
                num_transfers, free_transfers, hit_so_far, hit_this_gw = strat

                queue.put(
                    (
                        num_transfers,
                        free_transfers,
                        hit_so_far,
                        hit_this_gw,
                        new_squad,
                        strat_dict,
                        sid,
                    )
                )

        # mark this task as done only now that any children have been queued,
        # so queue.join() can't return before the whole tree is processed.
        queue.task_done()


def find_best_strat_from_json(tag: str) -> dict | None:
    """
    Look through all the files in our tmp directory that
    contain the prediction tag in their filename.
    Load the json, and find the strategy with the best 'total_score'.
    """
    best_score = 0
    best_strat = None
    file_list = os.listdir(OUTPUT_DIR)
    for filename in file_list:
        if f"strategy_{tag}_" not in filename:
            continue
        full_filename = os.path.join(OUTPUT_DIR, filename)
        with open(full_filename) as strat_file:
            strat = json.load(strat_file)
            if strat["total_score"] > best_score:
                best_score = strat["total_score"]
                best_strat = strat

    return best_strat


def save_baseline_score(squad: Squad, gameweeks: list[int], tag: str) -> None:
    """When strategies with unused transfers are excluded the baseline strategy will
    normally not be part of the tree. In that case save it first with this function.
    """
    strat_dict = get_baseline_strat(squad, gameweeks, tag, root_gw=gameweeks[0])

    num_gameweeks = len(gameweeks)
    zeros = ("0-" * num_gameweeks)[:-1]
    filename = os.path.join(OUTPUT_DIR, f"strategy_{tag}_{zeros}.json")
    with open(filename, "w") as f:
        json.dump(strat_dict, f)


def find_baseline_score_from_json(tag: str, num_gameweeks: int) -> float:
    """
    The baseline score is the one where we make 0 transfers
    for all gameweeks.
    """
    # the strategy string we're looking for will be something like '0-0-0'.
    zeros = ("0-" * num_gameweeks)[:-1]
    filename = os.path.join(OUTPUT_DIR, f"strategy_{tag}_{zeros}.json")
    if not os.path.exists(filename):
        logger.warning("Couldn't find %s", filename)
        return 0.0
    with open(filename) as inputfile:
        strat = json.load(inputfile)
        return strat["total_score"]


def print_optimization_summary(
    strat: dict,
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
    gameweeks_as_str = strat["points_per_gw"].keys()
    gameweeks_as_int = sorted(int(gw) for gw in gameweeks_as_str)
    first_gw, last_gw = gameweeks_as_int[0], gameweeks_as_int[-1]

    total_score = strat["total_score"]
    total_hits = sum(strat["points_hit"][str(gw)] for gw in gameweeks_as_int)

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
    for gw in gameweeks_as_int:
        chip = strat["chips_played"][str(gw)] or "-"
        points_hit = strat["points_hit"][str(gw)]
        pred_pts = strat["points_per_gw"][str(gw)] / strat["discount_factor"][str(gw)]
        strategy_table.add_row(
            str(gw),
            str(strat["num_transfers"][str(gw)]),
            chip,
            f"-{points_hit}pts" if points_hit else "0pts",
            f"{pred_pts:.1f}pts",
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
    for gw in gameweeks_as_int:
        players_out = strat["players_out"][str(gw)]
        players_in = strat["players_in"][str(gw)]
        for pid_out, pid_in in zip(players_out, players_in, strict=True):
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


def discord_payload(strat: dict, lineup: list[str]) -> dict:
    """
    json formated discord webhook content.
    """
    gameweeks_as_str = strat["points_per_gw"].keys()
    gameweeks_as_int = sorted([int(gw) for gw in gameweeks_as_str])
    discord_embed = {
        "title": "AIrsenal webhook",
        "description": "Optimum strategy for gameweek(S)"
        f" {','.join(str(x) for x in gameweeks_as_int)}:",
        "color": 0x35A800,
        "fields": [],
    }
    fields: list[dict] = []
    for gw in gameweeks_as_int:
        fields.append(
            {
                "name": f"GW{gw} chips:",
                "value": f"Chips played:  {strat['chips_played'][str(gw)]}\n",
                "inline": False,
            }
        )
        pin = [str(get_player_name(p)) for p in strat["players_in"][str(gw)]]
        pout = [str(get_player_name(p)) for p in strat["players_out"][str(gw)]]
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
    strat: dict,
    season: str = CURRENT_SEASON,
    fpl_team_id: int | None = None,
    use_api: bool = False,
) -> Squad:
    """
    Display the team (inc. subs and captain) for the next gameweek
    """
    gameweeks_as_str = strat["points_per_gw"].keys()
    gameweeks_as_int = sorted([int(gw) for gw in gameweeks_as_str])
    next_gw = gameweeks_as_int[0]
    t = get_starting_squad(
        next_gw=next_gw, season=season, fpl_team_id=fpl_team_id, use_api=use_api
    )
    for pidout in strat["players_out"][str(next_gw)]:
        t.remove_player(pidout)
    for pidin in strat["players_in"][str(next_gw)]:
        t.add_player(pidin)
    tag = get_latest_prediction_tag(season=season)
    chip_played = strat["chips_played"].get(str(next_gw))
    console.print(
        t.formation_table(
            tag,
            next_gw,
            bench_boost=chip_played == "bench_boost",
            triple_captain=chip_played == "triple_captain",
        )
    )
    return t


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
    profile: bool = False,
    is_replay: bool = False,  # for replaying seasons
    max_free_transfers: int = MAX_FREE_TRANSFERS,
) -> tuple[Squad, dict[str, dict[str, int | list[int]]] | None]:
    """
    This is the actual main function that sets up the multiprocessing
    and calls the optimize function for every num_transfers/gameweek
    combination, to find the best strategy.
    The chip-related variables e.g. wildcard_week are -1 if that chip
    is not to be played, 0 for 'play it any week', or the gw in which
    it should be played.
    """
    if chip_gameweeks is None:
        chip_gameweeks = {}
    discord_webhook = get_fetcher().DISCORD_WEBHOOK
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
        squad = fill_initial_squad(
            tag=tag,
            gw_range=gameweeks,
            season=season,
            fpl_team_id=fpl_team_id,
            num_generations=num_iterations,
            population_size=num_iterations,
            chip_gameweeks=chip_gameweeks,
        )
        return squad, None

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
            squad = fill_initial_squad(
                tag=tag,
                gw_range=gameweeks,
                season=season,
                fpl_team_id=fpl_team_id,
                num_generations=num_iterations,
                population_size=num_iterations,
                chip_gameweeks=chip_gameweeks,
            )
            return squad, None
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

        # create the output directory for temporary json files
        # giving the points prediction for each strategy
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # first get a baseline prediction
        # baseline_score, baseline_dict = get_baseline_prediction(num_weeks_ahead, tag)

        # Get a dict of what chips we definitely or possibly will play
        # in each gw
        chip_gw_dict = construct_chip_dict(gameweeks, chip_gameweeks)

        # Specific fix (aka hack) for the 2022 World Cup, where everyone
        # gets a free wildcard
        if season == "2223" and gameweeks[0] == 17:
            chip_gw_dict[gameweeks[0]]["chip_to_play"] = "wildcard"
            num_free_transfers = 1

        # create a queue that we will add nodes to, and some processes to take
        # things off it
        squeue = CustomQueue()
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
            chip_gw_dict=chip_gw_dict,
            max_free_transfers=max_free_transfers,
        )

        with progress_bar(transient=True) as progress:
            # one progress bar per worker process, plus one for overall progress
            worker_tasks = [
                progress.add_task(f"Worker {i}: idle", total=100)
                for i in range(num_thread)
            ]
            total_task = progress.add_task(
                "Total strategies", total=num_expected_outputs
            )

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

            progress_thread = threading.Thread(
                target=consume_progress_updates, daemon=True
            )
            progress_thread.start()

            if baseline_excluded:
                # if we are excluding unused transfers the tree may not include the
                # baseline strategy. In those cases quickly calculate and save it
                # here first.
                save_baseline_score(starting_squad, gameweeks, tag)
                progress.advance(total_task, 1)

            # Add Processes to run the target 'optimize' function.
            # This target function needs to know:
            #  num_transfers
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
                        gameweeks,
                        season,
                        tag,
                        chip_gw_dict,
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
            squeue.put((0, num_free_transfers, 0, 0, starting_squad, {}, "starting"))

            # block until every node in the (dynamically-grown) strategy tree has
            # been processed - i.e. the queue is empty and no worker is still
            # processing an item that could enqueue further children.
            squeue.join()

            progress_queue.put(None)
            progress_thread.join()
            progress.update(total_task, description="Transfer optimization complete")

        # tell each worker to shut down, then wait for them to exit
        for _ in procs:
            squeue.put(None)
        for p in procs:
            p.join()

        # find the best from all the strategies tried
        best_strategy = find_best_strat_from_json(tag)

        baseline_score = find_baseline_score_from_json(tag, num_weeks)
        fill_suggestion_table(baseline_score, best_strategy, season, fpl_team_id)
        if is_replay:
            # simulating a previous season, so imitate applying transfers by adding
            # the suggestions to the Transaction table
            fill_transaction_table(
                starting_squad, best_strategy, season, fpl_team_id, tag
            )

    console.print()

    if best_strategy is None:
        msg = "Failed to find a strategy!"
        raise ValueError(msg)

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

    # If a valid discord webhook URL has been stored
    # in env variables, send a webhook message
    if discord_webhook:
        # Use regex to check the discord webhook url is correctly formatted
        if re.match(
            r"^.*(discord|discordapp)\.com\/api\/webhooks\/([\d]+)\/([a-zA-Z0-9_-]+)$",
            discord_webhook,
        ):
            # create a formatted team lineup message for the discord webhook
            lineup_strings = [
                f"__Strategy for Team ID: **{fpl_team_id}**__",
                f"Baseline score: *{int(baseline_score)}*",
                f"Best score: *{int(best_strategy['total_score'])}*",
                "\n__starting 11__",
            ]
            for position in ["GK", "DEF", "MID", "FWD"]:
                lineup_strings.append(f"== **{position}** ==\n```")
                for p in best_squad.players:
                    if p.position == position and p.is_starting:
                        player_line = f"{p} ({p.team})"
                        if p.is_captain:
                            player_line += "(C)"
                        elif p.is_vice_captain:
                            player_line += "(VC)"
                        lineup_strings.append(player_line)
                lineup_strings.append("```\n")
            lineup_strings.append("__subs__")
            lineup_strings.append("```")
            subs = [p for p in best_squad.players if not p.is_starting]
            subs.sort(key=lambda p: p.sub_position)
            for p in subs:
                lineup_strings.append(f"{p} ({p.team})")
            lineup_strings.append("```\n")

            # generate a discord embed json and send to webhook
            payload = discord_payload(best_strategy, lineup_strings)
            result = requests.post(discord_webhook, json=payload)
            if 200 <= result.status_code < 300:
                logger.info("Discord webhook sent, status code: %s", result.status_code)
            else:
                logger.warning(
                    "Not sent with %s, response:\n%s",
                    result.status_code,
                    result.json(),
                )
        else:
            logger.warning("Discord webhook url is malformed: %s", discord_webhook)

    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    return best_squad, best_strategy


def construct_chip_dict(gameweeks: list[int], chip_gameweeks: dict) -> dict:
    """
    Given a dict of form {<chip_name>: <chip_gw>,...}
    where <chip_name> is e.g. 'wildcard', and <chip_gw> is -1 if chip
    is not to be played, 0 if it is to be considered any week, or gw
    if it is definitely to be played that gw, return a dict
    { <gw>: {"chip_to_play": [<chip_name>],
             "chips_allowed": [<chip_name>,...]},...}
    """
    chip_dict: dict[int, dict[str, str | list[str] | None]] = {}
    # first fill in any allowed chips
    for gw in gameweeks:
        chip_to_play: str | None = None
        chips_allowed: list[str] = []
        for k, v in chip_gameweeks.items():
            if int(v) == 0:
                chips_allowed.append(k)
        chip_dict[gw] = {
            "chip_to_play": chip_to_play,
            "chips_allowed": chips_allowed,
        }
    # now go through again, for any definite ones, and remove
    # other allowed chips from those gameweeks
    for k, v in chip_gameweeks.items():
        if v > 0 and v in gameweeks:  # v is the gameweek
            # check we're not trying to play 2 chips
            if chip_dict[v]["chip_to_play"] is not None:
                msg = (
                    f"Cannot play {chip_dict[v]['chip_to_play']} and {k} in the "
                    "same week"
                )
                raise RuntimeError(msg)
            chip_dict[v]["chip_to_play"] = k
            chip_dict[v]["chips_allowed"] = []
    return chip_dict


def sanity_check_args(
    weeks_ahead: int | None,
    gameweek_start: int | None,
    gameweek_end: int | None,
    num_free_transfers: int | None,
) -> bool:
    """
    Check that command-line arguments are self-consistent.
    """
    if weeks_ahead and (gameweek_start or gameweek_end):
        msg = "Please only specify weeks_ahead OR gameweek_start/end"
        raise RuntimeError(msg)
    if (gameweek_start and not gameweek_end) or (gameweek_end and not gameweek_start):
        msg = "Need to specify both gameweek_start and gameweek_end"
        raise RuntimeError(msg)
    if num_free_transfers and num_free_transfers not in range(6):
        msg = "Number of free transfers must be 0 to 5"
        raise RuntimeError(msg)
    return True


def run_transfer_optimization(
    weeks_ahead: int | None,
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
) -> None:
    """Run transfer optimization for a gameweek range."""
    sanity_check_args(
        weeks_ahead,
        gameweek_start,
        gameweek_end,
        num_free_transfers,
    )
    gameweeks = get_gameweeks_array(
        weeks_ahead=weeks_ahead,
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
        season,
        fpl_team_id,
        chip_gameweeks,
        num_free_transfers,
        max_hit,
        allow_unused,
        max_transfers,
        num_iterations,
        num_thread,
        profile,
        is_replay=is_replay,
    )
