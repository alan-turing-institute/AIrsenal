#!/usr/bin/env python

"""
usage:
python team_optimizer.py <num_weeks_ahead> <num_iterations>
output for each strategy tried is going to be a dict
{ "total_points": <float>,
"points_per_gw": {<gw>: <float>, ...},
"players_sold" : {<gw>: [], ...},
"players_bought" : {<gw>: [], ...}
}
"""

import os
import sys

sys.path.append("..")
import argparse

from framework.optimization_utils import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Try some different transfer strategies"
    )
    parser.add_argument(
        "--weeks_ahead", help="how many weeks ahead", type=int, default=3
    )
    parser.add_argument("--tag", help="tag of the prediction method")
    parser.add_argument(
        "--num_iterations", help="how many trials to run", type=int, default=100
    )
    args = parser.parse_args()
    if args.tag:
        tag = args.tag
    else:
        tag = get_latest_prediction_tag()
    num_weeks_ahead = args.weeks_ahead
    num_iterations = args.num_iterations

    baseline, score, strategy = optimize_transfers(num_weeks_ahead, tag, num_iterations)

    print("====================================\n")
    print("Baseline score: {}".format(baseline))
    print("Best score: {}".format(score))
    print(" best strategy")
    print(strategy)
