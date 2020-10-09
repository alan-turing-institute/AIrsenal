#!/usr/bin/env python

"""
usage:
python fill_transfersuggestsions_table.py --weeks_ahead <num_weeks_ahead> --num_iterations <num_iterations>
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

import os
import shutil
import sys
import time
import random

import json


from multiprocessing import Process
from tqdm import tqdm
import argparse

from airsenal.framework.multiprocessing_utils import CustomQueue
from airsenal.framework.optimization_utils import (
    get_starting_squad,
    calc_free_transfers,
    calc_points_hit,
    fill_suggestion_table,
    make_best_transfers
)

from airsenal.framework.utils import (
    CURRENT_SEASON,
    get_player_name,
    get_latest_prediction_tag,
    get_next_gameweek
)

if os.name == "posix":
    TMPDIR = "/tmp/"
else:
    TMPDIR = "%TMP%"

OUTPUT_DIR = os.path.join(TMPDIR, "airsopt")


def count_increments(strategy_string, num_iterations):
    """
    how many steps for the progress bar for this strategy
    """
    total = 0
    for s in strategy_string:
        if s=="W" or s=="F":
            ## wildcard or free hit - needs num_iterations iterations
            total+= num_iterations
        elif s=="1":
            ## single transfer - 15 increments (replace each player in turn)
            total+= 15
        elif s=="2":
            ## remove each pair of players - 15*7=105 combinations
            total += 105
        elif s=="3":
            total += num_iterations
    ## return at least 1, to avoid ZeroDivisionError
    return max(total,1)


def get_leaf_node_count(week, max_week, can_play_wildcard, can_play_free_hit):
    week += 1
    if week == max_week:
        return 3 + int(can_play_wildcard) + int(can_play_free_hit)
    for _ in range(3):
        total += get_leaf_node_count(week, max_week, can_play_wildcard, can_play_free_hit)
    if can_play_wildcard:
        total += get_leaf_node_count(week, max_week, False, can_play_free_hit)
    if can_play_free_hit:
        total += get_leaf_node_count(week, max_week, can_play_wildcard, False)
    return total


def is_finished(num_gameweeks, wildcard=False, free_hit=False, bench_boost=False, triple_captain=False):
    """
    We expected pow(3,num_gameweeks) json files in the output dir.
    Return True if they are all there, False otherwise.
    """
    possibilities_per_gameweek = 3
    if wildcard:
        possibilities_per_gameweek += 1
    if free_hit:
        possibilities_per_gameweek += 1
    if bench_boost:
        possibilities_per_gameweek += 3 # bench_boost plus zero, one or two transfers

    num_expected = pow(possibilities_per_gameweek, num_gameweeks)
    # count the json files in the output dir
    json_count = len(os.listdir(OUTPUT_DIR))
    if json_count == num_expected:
        return True
    return False


def optimize(queue, pid, gameweek_range, season, pred_tag):
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
     current_team,
     strat_dict,
     strat_id
    )
    """
    while True:
        if queue.qsize() > 0:
            print("PID {} - {} items on queue".format(queue.qsize(), pid))
            status = queue.get()
        else:
            if is_finished(len(gameweek_range)):
                print("All jobs done: process {} exiting.".format(pid))
                break
            else:
                time.sleep(5)
                continue

        # now assume we have set of parameters to do an optimization
        # from the queue.

        num_transfers, free_transfers, squad, strat_dict, sid = status

        # sid (status id) is just a string e.g. "002" representing how many
        # transfers to be made in each gameweek.
        # Only exception is the root node, where sid is "STARTING" - this
        # node only exists to add children to the queue.

        if sid == "STARTING":
            sid = ""
            depth = 0
            strat_dict["total_score"] = 0
            strat_dict["points_per_gw"] = {}
            strat_dict["players_in"] = {}
            strat_dict["players_out"] = {}
            strat_dict["cards_played"] = {}
        else:
            sid = sid + str(num_transfers)
            print("Process {} doing strategy {}".format(pid, sid))

            # work out what gameweek we're in and how far down the tree we are.
            depth = len(strat_dict["points_per_gw"])

            # gameweeks from this point in strategy to end of window
            gameweeks = gameweek_range[depth:]
            # next gameweek:
            gw = gameweeks[0]
            if num_transfers > 0:
                squad, transfers, points = make_best_transfers(num_transfers,
                                                               squad,
                                                               pred_tag,
                                                               gameweeks,
                                                               season)

                points += calc_points_hit(num_transfers, free_transfers)
                strat_dict["players_in"][gw] = transfers["in"]
                strat_dict["players_out"][gw] = transfers["out"]
            else:
                # no transfers
                strat_dict["players_in"][gw] = []
                strat_dict["players_out"][gw] = []
                strat_dict["cards_played"][gw] = []
                points = squad.get_expected_points(gw, pred_tag)

            free_transfers = calc_free_transfers(num_transfers, free_transfers)
            strat_dict["total_score"] += points
            strat_dict["points_per_gw"][gw] = points

            depth += 1
        if depth >= len(gameweek_range):
            print("Process {} Finished {}: {}".format(pid,
                                                      sid,
                                                      strat_dict["total_score"]))
            with open(
                    os.path.join(OUTPUT_DIR,
                                 "strategy_{}_{}.json".format(pred_tag, sid)),
                    "w") as outfile:
                json.dump(strat_dict, outfile)

        else:
            # add children to the queue
            for num_transfers in range(3):
                queue.put((num_transfers, free_transfers, squad, strat_dict, sid))


def find_best_strat_from_json(tag):
    """
    Look through all the files in our tmp directory that
    contain the prediction tag in their filename.
    Load the json, and find the strategy with the best 'total_score'.
    """
    best_score = 0
    best_strat = None
    file_list = os.listdir(OUTPUT_DIR)
    for filename in file_list:
        if not "strategy_{}_".format(tag) in filename:
            continue
        full_filename = os.path.join(OUTPUT_DIR, filename)
        with open(full_filename) as strat_file:
            strat = json.load(strat_file)
            if strat["total_score"] > best_score:
                best_score = strat["total_score"]
                best_strat = strat
        ## cleanup
#        os.remove(full_filename)
    return best_strat


def find_baseline_score_from_json(tag, num_gameweeks):
    """
    The baseline score is the one where we make 0 transfers
    for all gameweeks.
    """
    zeros = "0"*num_gameweeks
    filename = os.path.join(OUTPUT_DIR, "strategy_{}_{}.json"\
                            .format(tag, zeros))
    if not os.path.exists(filename):
        print("Couldn't find {}".format(filename))
        return 0.
    else:
        with open(filename) as inputfile:
            strat = json.load(inputfile)
            score = strat["total_score"]
            return score



def print_strat(strat):
    """
    nicely formated printout as output of optimization.
    """

    gameweeks_as_str = strat['points_per_gw'].keys()
    gameweeks_as_int = sorted([ int(gw) for gw in gameweeks_as_str])
    print(" ===============================================")
    print(" ========= Optimum strategy ====================")
    print(" ===============================================")
    for gw in gameweeks_as_int:
        print("\n =========== Gameweek {} ================\n".format(gw))
       # print("Cards played:  {}\n".format(strat['cards_played'][str(gw)]))
        print("Players in:\t\t\tPlayers out:")
        print("-----------\t\t\t------------")
        for i in range(len(strat['players_in'][str(gw)])):
            pin = get_player_name(strat['players_in'][str(gw)][i])
            pout = get_player_name(strat['players_out'][str(gw)][i])
            if len(pin) < 20:
                subs = "{}\t\t\t{}".format(pin,pout)
            else:
                subs = "{}\t\t{}".format(pin,pout)
            print(subs)
    print("\n==========================")
    print(" Total score: {} \n".format(int(strat['total_score'])))
    pass


def print_team_for_next_gw(strat):
    """
    Display the team (inc. subs and captain) for the next gameweek
    """
    t = get_starting_squad()
    gameweeks_as_str = strat['points_per_gw'].keys()
    gameweeks_as_int = sorted([ int(gw) for gw in gameweeks_as_str])
    next_gw = gameweeks_as_int[0]
    for pidout in strat['players_out'][str(next_gw)]:
        t.remove_player(pidout)
    for pidin in strat['players_in'][str(next_gw)]:
        t.add_player(pidin)
    tag = get_latest_prediction_tag()
    expected_points = t.get_expected_points(next_gw,tag)
    print(t)



def run_optimization(gameweeks,
                     tag,
                     season=CURRENT_SEASON,
                     wildcard=False,
                     free_hit=False,
                     num_free_transfers=1,
                     bank=0,
                     max_points_hit=4,
                     num_thread=4):
    """
    This is the actual main function that sets up the multiprocessing
    and calls the optimize function for every num_transfers/gameweek
    combination, to find the best strategy.
    """
    ## create the output directory for temporary json files
    ## giving the points prediction for each strategy
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ## first get a baseline prediction
    #baseline_score, baseline_dict = get_baseline_prediction(num_weeks_ahead, tag)

    ## create a queue that we will add strategies to, and some processes to take
    ## things off it
    squeue = CustomQueue()
    procs = []
    ## create one progress bar for each thread
    progress_bars = []
    for i in range(num_thread):
        progress_bars.append(tqdm(total=100))

    ## number of nodes in tree will be 3^num_weeks unless we allow
    ## wildcard or free hit, in which case it'll be 4^num_weeks
    num_weeks = len(gameweeks)
    if not (wildcard or free_hit):
        total_nodes = pow(3, num_weeks)
    elif (wildcard and free_hit):
        total_nodes = pow(5,num_weeks)
    else:
        total_nodes = pow(4,num_weeks)
    total_progress = tqdm(total=total_nodes, desc="Total progress")


    ## functions to be passed to subprocess to update or reset progress bars
    def reset_progress(index, strategy_string):
        if strategy_string == "DONE":
            progress_bars[index].close()
        else:
            progress_bars[index].n=0
            progress_bars[index].desc="strategy: "+strategy_string
            progress_bars[index].refresh()


    def update_progress(increment=1, index=None):
        if index==None:
            ## outer progress bar
            nfiles = len(os.listdir(OUTPUT_DIR))
            total_progress.n = nfiles
            total_progress.refresh()
            if nfiles == len(strategies):
                total_progress.close()
                for pb in progress_bars:
                    pb.close()
        else:
            progress_bars[index].update(increment)
            progress_bars[index].refresh()
    ## target function for process needs to know:
    ## num_transfers
    ## current_team (list of player_ids)
    ## transfer_dict {"gw":<gw>,"in":[],"out":[]}
    ## total_score
    ## num_free_transfers
    ## budget
    for i in range(num_thread):
        processor = Process(
            target=optimize,
            args=(squeue, i, gameweeks, season, tag)
        )
        processor.daemon = True
        processor.start()
        procs.append(processor)
    ## add starting node to the queue
    starting_squad = get_starting_squad()
    squeue.put((0,0,starting_squad, {}, "STARTING"))

    for i,p in enumerate(procs):
        progress_bars[i].close()
        progress_bars[i] = None
        p.join()

    ### find the best from all the strategies tried
    best_strategy = find_best_strat_from_json(tag)

    baseline_score = find_baseline_score_from_json(tag, num_weeks)
    fill_suggestion_table(baseline_score, best_strategy, season)
    for i in range(len(procs)):
        print("\n")
    print("\n====================================\n")
    print("Baseline score: {}".format(baseline_score))
    print("Best score: {}".format(best_strategy["total_score"]))
    print_strat(best_strategy)
    print_team_for_next_gw(best_strategy)


def sanity_check_args(args):
    """
    Check that command-line arguments are self-consistent.
    """
    if args.weeks_ahead and (args.gw_start or args.gw_end):
        raise RuntimeError("Please only specify weeks_ahead OR gw_start/end")
    elif (args.gw_start and not args.gw_end) or \
         (args.gw_end and not args.gw_start):
        raise RuntimeError("Need to specify both gw_start and gw_end")
    return True






def main():
    """
    The main function, to be used as entrypoint.
    """
    parser = argparse.ArgumentParser(
        description="Try some different transfer strategies"
    )
    parser.add_argument(
        "--weeks_ahead", help="how many weeks ahead", type=int
    )
    parser.add_argument(
        "--gw_start", help="first gameweek to consider", type=int
    )
    parser.add_argument(
        "--gw_end", help="last gameweek to consider", type=int
    )
    parser.add_argument("--tag", help="specify a string identifying prediction set")
    parser.add_argument(
        "--num_iterations", help="how many trials to run", type=int, default=100
    )
    parser.add_argument("--allow_wildcard",
                        help="include possibility of wildcarding in one of the weeks",
                        action="store_true")
    parser.add_argument("--allow_free_hit",
                        help="include possibility of playing free hit in one of the weeks",
                        action="store_true")
    parser.add_argument("--max_points_hit",
                        help="how many points are we prepared to lose on transfers",
                        type=int, default=4)
    parser.add_argument("--num_free_transfers",
                        help="how many free transfers do we have",
                        type=int, default=1)
    parser.add_argument("--bank",
                        help="how much money do we have in the bank (multiplied by 10)?",
                        type=int, default=0)
    parser.add_argument("--num_thread",
                        help="how many threads to use",
                        type=int, default=4)
    parser.add_argument("--season",
                        help="what season, in format e.g. '2021'",
                        type=str, default=CURRENT_SEASON)
    args = parser.parse_args()

    args_ok = sanity_check_args(args)
    season = args.season
    if args.weeks_ahead:
        gameweeks = list(range(get_next_gameweek(),
                               get_next_gameweek()+args.weeks_ahead))
    else:
        gameweeks = list(range(args.gw_start, args.gw_end))
    num_iterations = args.num_iterations
    if args.allow_wildcard:
        wildcard = True
    else:
        wildcard = False
    if args.allow_free_hit:
        free_hit = True
    else:
        free_hit = False
    num_free_transfers = args.num_free_transfers
    bank = args.bank
    max_points_hit = args.max_points_hit
    if args.tag:
        tag = args.tag
    else:
        ## get most recent set of predictions from DB table
        tag = get_latest_prediction_tag()
    num_thread = args.num_thread
    run_optimization(gameweeks,
                     tag,
                     season,
                     wildcard,
                     free_hit,
                     num_free_transfers,
                     bank,
                     max_points_hit,
                     num_thread)
