#!/usr/bin/env python

"""
usage:
python parallel_fill_transfersuggestsions_table.py --weeks_ahead <num_weeks_ahead> --num_iterations <num_iterations>
output for each strategy tried is going to be a dict
{ "total_points": <float>,
"points_per_gw": {<gw>: <float>, ...},
"players_sold" : {<gw>: [], ...},
"players_bought" : {<gw>: [], ...}
}
"""

import os
import sys
import time


import json


from multiprocessing import Process, Queue
import argparse

from ..framework.optimization_utils import *

NUM_THREAD = 4
OUTPUT_DIR = "../data"


def process_strat(queue, pid, num_iterations, method, baseline=None):
    while True:
        strat = queue.get()
        if strat == "DONE":
            break
        sid = make_strategy_id(strat)
        if not strategy_involves_2_or_more_transfers_in_gw(strat):
            num_iter = 1
        else:
            num_iter = num_iterations
        print("ID {} doing {} iterations for Strategy {}".format(pid, num_iter, strat))
        strat_output = apply_strategy(strat, method, baseline, num_iter)
        with open(
            os.path.join(OUTPUT_DIR, "strategy_{}_{}.json".format(method, sid)), "w"
        ) as outfile:
            json.dump(strat_output, outfile)


def find_best_strat_from_json(method):
    best_score = 0
    best_strat = None
    file_list = os.listdir(OUTPUT_DIR)
    for filename in file_list:
        if not "strategy_{}_".format(method) in filename:
            continue
        full_filename = os.path.join(OUTPUT_DIR, filename)
        with open(full_filename) as strat_file:
            strat = json.load(strat_file)
            if strat["total_score"] > best_score:
                best_score = strat["total_score"]
                best_strat = strat
        ## cleanup
        os.remove(full_filename)
    return best_strat


def main():

    parser = argparse.ArgumentParser(
        description="Try some different transfer strategies"
    )
    parser.add_argument(
        "--weeks_ahead", help="how many weeks ahead", type=int, default=3
    )
    parser.add_argument("--tag", help="specify a string identifying prediction set")
    parser.add_argument(
        "--num_iterations", help="how many trials to run", type=int, default=100
    )
    parser.add_argument("--exhaustive_double_transfer",
                        help="use exhaustive search when doing 2 transfers in gameweek",
                        action="store_true")
    args = parser.parse_args()

    num_weeks_ahead = args.weeks_ahead
    num_iterations = args.num_iterations

    if args.tag:
        method = args.tag
    else:
        ## get most recent set of predictions from DB table
        method = get_latest_prediction_tag()

    ## first get a baseline prediction
    baseline_score, baseline_dict = get_baseline_prediction(num_weeks_ahead, method)

    ## create a queue that we will add strategies to, and some processes to take
    ## things off it
    squeue = Queue()
    procs = []
    for i in range(NUM_THREAD):
        processor = Process(
            target=process_strat,
            args=(squeue, i, num_iterations, method, baseline_dict),
        )
        processor.daemon = True
        processor.start()
        procs.append(processor)

    ### add strategies to the queue
    strategies = generate_transfer_strategies(num_weeks_ahead)
    for strat in strategies:
        squeue.put(strat)
    for i in range(NUM_THREAD):
        squeue.put("DONE")
    ### now rejoin the main thread
    for p in procs:
        p.join()

    ### find the best from all the strategies tried
    best_strategy = find_best_strat_from_json(method)

    fill_suggestion_table(baseline_score, best_strategy)
    print("====================================\n")
    print("Baseline score: {}".format(baseline_score))
    print("Best score: {}".format(best_strategy["total_score"]))
    print(" best strategy")
    print(best_strategy)
