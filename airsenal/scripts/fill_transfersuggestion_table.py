#!/usr/bin/env python

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


import argparse
import cProfile
import json
import os
import shutil
import sys
import time
import warnings
from multiprocessing import Process, Queue
from typing import Callable, List, Optional

import regex as re
import requests
from tqdm import TqdmWarning, tqdm

from airsenal.framework.env import AIRSENAL_HOME
from airsenal.framework.multiprocessing_utils import (
    CustomQueue,
    set_multiprocessing_start_method,
)
from airsenal.framework.optimization_transfers import make_best_transfers
from airsenal.framework.optimization_utils import (
    calc_free_transfers,
    calc_points_hit,
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
    fetcher,
    get_entry_start_gameweek,
    get_free_transfers,
    get_gameweeks_array,
    get_latest_prediction_tag,
    get_player_name,
)
from airsenal.scripts.squad_builder import fill_initial_squad

OUTPUT_DIR = os.path.join(AIRSENAL_HOME, "airsopt")


def is_finished(final_expected_num: int) -> bool:
    """
    Count the number of json files in the output directory, and see if the number
    matches the final expected number, which should be pre-calculated by the
    count_expected_points function based on the number of weeks optimising for, chips
    available and other constraints.
    Return True if output files are all there, False otherwise.
    """

    # count the json files in the output dir
    json_count = len(os.listdir(OUTPUT_DIR))
    return json_count == final_expected_num


def optimize(
    queue: Queue,
    pid: Process,
    num_expected_outputs: int,
    gameweek_range: List[int],
    season: str,
    pred_tag: str,
    chips_gw_dict: dict,
    max_total_hit: Optional[int] = None,
    allow_unused_transfers: bool = False,
    max_transfers: int = 2,
    num_iterations: int = 100,
    updater: Optional[Callable] = None,
    resetter: Optional[Callable] = None,
    profile: bool = False,
) -> None:
    """
    Queue is the multiprocessing queue,
    pid is the Process that will execute this func,
    gameweeks will be a list of gameweeks to consider,
    season and prediction_tag are hopefully self-explanatory.

    The rest of the parameters needed for prediction are from the queue.

    Things on the queue will either be "FINISHED", or a tuple:
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
        if queue.qsize() > 0:
            status = queue.get()
        else:
            if is_finished(num_expected_outputs):
                break
            time.sleep(5)
            continue

        # now assume we have set of parameters to do an optimization
        # from the queue.

        # turn on the profiler if requested
        if profile:
            profiler = cProfile.Profile()
            profiler.enable()

        num_transfers, free_transfers, hit_so_far, squad, strat_dict, sid = status
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
            new_squad = squad
            gw = gameweek_range[0] - 1
            strat_dict["root_gw"] = gameweek_range[0]
        else:
            if len(sid) > 0:
                sid += "-"
            sid += str(num_transfers)
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
                (updater, increment, pid),
            )

            points_hit = calc_points_hit(num_transfers, free_transfers)
            discount_factor = get_discount_factor(root_gw, gw)
            points -= points_hit * discount_factor
            strat_dict["total_score"] += points
            strat_dict["points_per_gw"][gw] = points
            strat_dict["free_transfers"][gw] = free_transfers
            strat_dict["num_transfers"][gw] = num_transfers
            strat_dict["points_hit"][gw] = points_hit
            strat_dict["discount_factor"][gw] = discount_factor
            strat_dict["players_in"][gw] = transfers["in"]
            strat_dict["players_out"][gw] = transfers["out"]
            free_transfers = calc_free_transfers(num_transfers, free_transfers)

            depth += 1

        if depth >= len(gameweek_range):
            with open(
                os.path.join(OUTPUT_DIR, f"strategy_{pred_tag}_{sid}.json"),
                "w",
            ) as outfile:
                json.dump(strat_dict, outfile)
            # call function to update the main progress bar
            updater()

            if profile:
                profiler.dump_stats(f"process_strat_{pred_tag}_{sid}.pstat")

        else:
            # add children to the queue
            strategies = next_week_transfers(
                (free_transfers, hit_so_far, strat_dict),
                max_total_hit=max_total_hit,
                allow_unused_transfers=allow_unused_transfers,
                max_transfers=max_transfers,
                chips=chips_gw_dict[gw + 1],
            )

            for strat in strategies:
                # strat: (num_transfers, free_transfers, hit_so_far)
                num_transfers, free_transfers, hit_so_far = strat

                queue.put(
                    (
                        num_transfers,
                        free_transfers,
                        hit_so_far,
                        new_squad,
                        strat_dict,
                        sid,
                    )
                )


def find_best_strat_from_json(tag: str) -> dict:
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


def save_baseline_score(squad: Squad, gameweeks: List[int], tag: str) -> None:
    """When strategies with unused transfers are excluded the baseline strategy will
    normally not be part of the tree. In that case save it first with this function.
    """
    strat_dict = get_baseline_strat(squad, gameweeks, tag, root_gw=gameweeks[0])

    num_gameweeks = len(gameweeks)
    zeros = ("0-" * num_gameweeks)[:-1]
    filename = os.path.join(OUTPUT_DIR, f"strategy_{tag}_{zeros}.json")
    with open(filename, "w") as f:
        json.dump(strat_dict, f)


def find_baseline_score_from_json(tag: str, num_gameweeks: int) -> None:
    """
    The baseline score is the one where we make 0 transfers
    for all gameweeks.
    """
    # the strategy string we're looking for will be something like '0-0-0'.
    zeros = ("0-" * num_gameweeks)[:-1]
    filename = os.path.join(OUTPUT_DIR, f"strategy_{tag}_{zeros}.json")
    if not os.path.exists(filename):
        print(f"Couldn't find {filename}")
        return 0.0
    else:
        with open(filename) as inputfile:
            strat = json.load(inputfile)
            return strat["total_score"]


def print_strat(strat: dict) -> None:
    """
    nicely formatted printout as output of optimization.
    """
    gameweeks_as_str = strat["points_per_gw"].keys()
    gameweeks_as_int = sorted([int(gw) for gw in gameweeks_as_str])
    print(" ===============================================")
    print(" ========= Optimum strategy ====================")
    print(" ===============================================")
    for gw in gameweeks_as_int:
        print(f"\n =========== Gameweek {gw} ================\n")
        print(f"Chips played:  {strat['chips_played'][str(gw)]}\n")
        print("Players in:\t\t\tPlayers out:")
        print("-----------\t\t\t------------")
        for i in range(len(strat["players_in"][str(gw)])):
            pin = get_player_name(strat["players_in"][str(gw)][i])
            pout = get_player_name(strat["players_out"][str(gw)][i])
            if len(pin) < 20:
                subs = f"{pin}\t\t\t{pout}"
            else:
                subs = f"{pin}\t\t{pout}"
            print(subs)
    print("\n==========================")
    print(f" Total score: {int(strat['total_score'])} \n")


def discord_payload(strat: dict, lineup: List[str]) -> dict:
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
    for gw in gameweeks_as_int:
        discord_embed["fields"].append(
            {
                "name": f"GW{gw} chips:",
                "value": f"Chips played:  {strat['chips_played'][str(gw)]}\n",
                "inline": False,
            }
        )
        pin = [get_player_name(p) for p in strat["players_in"][str(gw)]]
        pout = [get_player_name(p) for p in strat["players_out"][str(gw)]]
        discord_embed["fields"].extend(
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
    payload = {
        "content": "\n".join(lineup),
        "username": "AIrsenal",
        "embeds": [discord_embed],
    }
    return payload


def print_team_for_next_gw(
    strat: dict, season: str = CURRENT_SEASON, fpl_team_id: Optional[int] = None
) -> Squad:
    """
    Display the team (inc. subs and captain) for the next gameweek
    """
    gameweeks_as_str = strat["points_per_gw"].keys()
    gameweeks_as_int = sorted([int(gw) for gw in gameweeks_as_str])
    next_gw = gameweeks_as_int[0]
    t = get_starting_squad(next_gw=next_gw, season=season, fpl_team_id=fpl_team_id)
    for pidout in strat["players_out"][str(next_gw)]:
        t.remove_player(pidout)
    for pidin in strat["players_in"][str(next_gw)]:
        t.add_player(pidin)
    tag = get_latest_prediction_tag(season=season)
    t.get_expected_points(next_gw, tag)
    print(t)
    return t


def run_optimization(
    gameweeks: List[int],
    tag: str,
    season: str = CURRENT_SEASON,
    fpl_team_id: Optional[int] = None,
    chip_gameweeks: dict = {},
    num_free_transfers: Optional[int] = None,
    max_total_hit: Optional[int] = None,
    allow_unused_transfers: bool = False,
    max_transfers: int = 2,
    num_iterations: int = 100,
    num_thread: int = 4,
    profile: bool = False,
    is_replay: bool = False,  # for replaying seasons
):
    """
    This is the actual main function that sets up the multiprocessing
    and calls the optimize function for every num_transfers/gameweek
    combination, to find the best strategy.
    The chip-related variables e.g. wildcard_week are -1 if that chip
    is not to be played, 0 for 'play it any week', or the gw in which
    it should be played.
    """
    discord_webhook = fetcher.DISCORD_WEBHOOK
    if fpl_team_id is None:
        fpl_team_id = fetcher.FPL_TEAM_ID

    # see if we are at the start of a season, or
    if gameweeks[0] == 1 or gameweeks[0] == get_entry_start_gameweek(
        fpl_team_id, apifetcher=fetcher
    ):
        print(
            "This is the start of the season or a new team - will make a squad "
            "from scratch"
        )
        fill_initial_squad(
            tag=tag,
            gw_range=gameweeks,
            season=season,
            fpl_team_id=fpl_team_id,
            num_iterations=num_iterations,
        )
        return

    print(f"Running optimization with fpl_team_id {fpl_team_id}")
    use_api = season == CURRENT_SEASON and not is_replay
    try:
        starting_squad = get_starting_squad(
            next_gw=gameweeks[0],
            season=season,
            fpl_team_id=fpl_team_id,
            use_api=use_api,
            apifetcher=fetcher,
        )
    except (ValueError, TypeError):
        # first week for this squad?
        print(f"No existing squad or transfers found for team_id {fpl_team_id}")
        print("Will suggest a new starting squad:")
        fill_initial_squad(
            tag=tag,
            gw_range=gameweeks,
            season=season,
            fpl_team_id=fpl_team_id,
            num_iterations=num_iterations,
        )
        return
    # if we got to here, we can assume we are optimizing an existing squad.

    # How many free transfers are we starting with?
    if not num_free_transfers:
        num_free_transfers = get_free_transfers(
            fpl_team_id,
            gameweeks[0],
            season=season,
            apifetcher=fetcher,
            is_replay=is_replay,
        )

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
    # create one progress bar for each thread
    progress_bars = []
    for i in range(num_thread):
        progress_bars.append(tqdm(total=100))

    # number of nodes in tree will be something like 3^num_weeks unless we allow
    # a "chip" such as wildcard or free hit, in which case it gets complicated
    num_weeks = len(gameweeks)
    num_expected_outputs = count_expected_outputs(
        num_weeks,
        next_gw=gameweeks[0],
        free_transfers=num_free_transfers,
        max_total_hit=max_total_hit,
        allow_unused_transfers=allow_unused_transfers,
        max_transfers=max_transfers,
        chip_gw_dict=chip_gw_dict,
    )
    total_progress = tqdm(total=num_expected_outputs, desc="Total progress")

    # functions to be passed to subprocess to update or reset progress bars
    def reset_progress(index, strategy_string):
        if strategy_string == "DONE":
            progress_bars[index].close()
        else:
            progress_bars[index].n = 0
            progress_bars[index].desc = "strategy: " + strategy_string
            progress_bars[index].refresh()

    def update_progress(increment=1, index=None):
        if index is None:
            # outer progress bar
            nfiles = len(os.listdir(OUTPUT_DIR))
            total_progress.n = nfiles
            total_progress.refresh()
            if nfiles == num_expected_outputs:
                total_progress.close()
                for pb in progress_bars:
                    pb.close()
        else:
            progress_bars[index].update(increment)
            progress_bars[index].refresh()

    if not allow_unused_transfers and (
        num_weeks > 1 or (num_weeks == 1 and num_free_transfers == 2)
    ):
        # if we are excluding unused transfers the tree may not include the baseline
        # strategy. In those cases quickly calculate and save it here first.
        save_baseline_score(starting_squad, gameweeks, tag)
        update_progress()

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
                num_expected_outputs,
                gameweeks,
                season,
                tag,
                chip_gw_dict,
                max_total_hit,
                allow_unused_transfers,
                max_transfers,
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
    squeue.put((0, num_free_transfers, 0, starting_squad, {}, "starting"))

    for i, p in enumerate(procs):
        progress_bars[i].close()
        progress_bars[i] = None
        p.join()

    # find the best from all the strategies tried
    best_strategy = find_best_strat_from_json(tag)

    baseline_score = find_baseline_score_from_json(tag, num_weeks)
    fill_suggestion_table(baseline_score, best_strategy, season, fpl_team_id)
    if is_replay:
        # simulating a previous season, so imitate applying transfers by adding
        # the suggestions to the Transaction table
        fill_transaction_table(starting_squad, best_strategy, season, fpl_team_id, tag)

    for i in range(len(procs)):
        print("\n")
    print("\n====================================\n")
    print(f"Strategy for Team ID: {fpl_team_id}")
    print(f"Baseline score: {baseline_score}")
    print(f"Best score: {best_strategy['total_score']}")
    print_strat(best_strategy)
    best_squad = print_team_for_next_gw(
        best_strategy, season=season, fpl_team_id=fpl_team_id
    )

    # If a valid discord webhook URL has been stored
    # in env variables, send a webhook message
    if discord_webhook != "MISSING_ID":
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
                        player_line = f"{p.name} ({p.team})"
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
                lineup_strings.append(f"{p.name} ({p.team})")
            lineup_strings.append("```\n")

            # generate a discord embed json and send to webhook
            payload = discord_payload(best_strategy, lineup_strings)
            result = requests.post(discord_webhook, json=payload)
            if 200 <= result.status_code < 300:
                print(f"Discord webhook sent, status code: {result.status_code}")
            else:
                print(f"Not sent with {result.status_code}, response:\n{result.json()}")
        else:
            print("Warning: Discord webhook url is malformed!\n", discord_webhook)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    return best_squad, best_strategy


def construct_chip_dict(gameweeks: List[int], chip_gameweeks: dict) -> dict:
    """
    Given a dict of form {<chip_name>: <chip_gw>,...}
    where <chip_name> is e.g. 'wildcard', and <chip_gw> is -1 if chip
    is not to be played, 0 if it is to be considered any week, or gw
    if it is definitely to be played that gw, return a dict
    { <gw>: {"chip_to_play": [<chip_name>],
             "chips_allowed": [<chip_name>,...]},...}
    """
    chip_dict = {}
    # first fill in any allowed chips
    for gw in gameweeks:
        chip_dict[gw] = {"chip_to_play": None, "chips_allowed": []}
        for k, v in chip_gameweeks.items():
            if int(v) == 0:
                chip_dict[gw]["chips_allowed"].append(k)
    # now go through again, for any definite ones, and remove
    # other allowed chips from those gameweeks
    for k, v in chip_gameweeks.items():
        if v > 0 and v in gameweeks:  # v is the gameweek
            # check we're not trying to play 2 chips
            if chip_dict[v]["chip_to_play"] is not None:
                raise RuntimeError(
                    (
                        f"Cannot play {chip_dict[v]['chip_to_play']} and {k} in the "
                        "same week"
                    )
                )
            chip_dict[v]["chip_to_play"] = k
            chip_dict[v]["chips_allowed"] = []
    return chip_dict


def sanity_check_args(args: argparse.Namespace) -> bool:
    """
    Check that command-line arguments are self-consistent.
    """
    if args.weeks_ahead and (args.gameweek_start or args.gameweek_end):
        raise RuntimeError("Please only specify weeks_ahead OR gameweek_start/end")
    elif (args.gameweek_start and not args.gameweek_end) or (
        args.gameweek_end and not args.gameweek_start
    ):
        raise RuntimeError("Need to specify both gameweek_start and gameweek_end")
    if args.num_free_transfers and args.num_free_transfers not in range(1, 3):
        raise RuntimeError("Number of free transfers must be 1 or 2")
    return True


def main():
    """
    The main function, to be used as entrypoint.
    """
    parser = argparse.ArgumentParser(
        description="Try some different transfer strategies"
    )
    parser.add_argument("--weeks_ahead", help="how many weeks ahead", type=int)
    parser.add_argument("--gameweek_start", help="first gameweek to consider", type=int)
    parser.add_argument("--gameweek_end", help="last gameweek to consider", type=int)
    parser.add_argument("--tag", help="specify a string identifying prediction set")
    parser.add_argument(
        "--wildcard_week",
        help="play wildcard in the specified week. Choose 0 for 'any week'.",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--free_hit_week",
        help="play free hit in the specified week. Choose 0 for 'any week'.",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--triple_captain_week",
        help="play triple captain in the specified week. Choose 0 for 'any week'.",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--bench_boost_week",
        help="play bench_boost in the specified week. Choose 0 for 'any week'.",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--num_free_transfers", help="how many free transfers do we have", type=int
    )
    parser.add_argument(
        "--max_hit",
        help="maximum number of points to spend on additional transfers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--allow_unused",
        help="if set, include strategies that waste free transfers",
        action="store_true",
    )
    parser.add_argument(
        "--num_iterations",
        help="how many iterations to use for Wildcard/Free Hit optimization",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--num_thread", help="how many threads to use", type=int, default=4
    )
    parser.add_argument(
        "--season",
        help="what season, in format e.g. '2021'",
        type=str,
        default=CURRENT_SEASON,
    )
    parser.add_argument(
        "--profile",
        help="For developers: Profile strategy execution time",
        action="store_true",
    )
    parser.add_argument(
        "--fpl_team_id",
        help="specify fpl team id",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--is_replay",
        help="Add suggested squad to the database (for replaying seasons)",
        action="store_true",
    )
    args = parser.parse_args()

    fpl_team_id = args.fpl_team_id or None

    sanity_check_args(args)
    season = args.season
    gameweeks = get_gameweeks_array(
        weeks_ahead=args.weeks_ahead,
        gameweek_start=args.gameweek_start,
        gameweek_end=args.gameweek_end,
        season=season,
    )

    num_iterations = args.num_iterations

    if args.num_free_transfers:
        num_free_transfers = args.num_free_transfers
    else:
        num_free_transfers = None  # will work it out in run_optimization
    tag = args.tag or get_latest_prediction_tag(season=season)
    max_total_hit = args.max_hit
    allow_unused_transfers = args.allow_unused
    num_thread = args.num_thread
    profile = args.profile or False
    chip_gameweeks = {
        "wildcard": args.wildcard_week,
        "free_hit": args.free_hit_week,
        "triple_captain": args.triple_captain_week,
        "bench_boost": args.bench_boost_week,
    }

    if not check_tag_valid(tag, gameweeks, season=season):
        print(
            "ERROR: Database does not contain predictions",
            "for all the specified optimsation gameweeks.\n",
            "Please run 'airsenal_run_prediction' first with the",
            "same input gameweeks and season you specified here.",
        )
        sys.exit(1)

    set_multiprocessing_start_method()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TqdmWarning)
        run_optimization(
            gameweeks,
            tag,
            season,
            fpl_team_id,
            chip_gameweeks,
            num_free_transfers,
            max_total_hit,
            allow_unused_transfers,
            2,
            num_iterations,
            num_thread,
            profile,
            is_replay=args.is_replay,
        )
