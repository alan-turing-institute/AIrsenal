"""
Script to replay all or part of a season, to allow evaluation of different
code and strategies.
"""
import argparse

from airsenal.framework.schema import session_scope
from airsenal.scripts.fill_predictedscore_table import make_predictedscore_table


def rerun_predictions(season, gw_start, gw_end, weeks_ahead=3, num_thread=4):
    """
    Run the predictions each week for gw_start to gw_end in chosen season.
    """
    with session_scope() as session:
        for gw in range(gw_start, gw_end + 1):
            print(
                "======== Running predictions for {} week {} ======".format(season, gw)
            )
            tag_prefix = season + "_" + str(gw) + "_"
            make_predictedscore_table(
                gw_range=range(gw, gw + weeks_ahead),
                season=season,
                num_thread=num_thread,
                tag_prefix=tag_prefix,
                dbsession=session,
            )


def main():
    parser = argparse.ArgumentParser(description="fill player predictions")

    parser.add_argument(
        "--gameweek_start", help="first gameweek to look at", type=int, default=1
    )
    parser.add_argument(
        "--gameweek_end", help="last gameweek to look at", type=int, default=38
    )
    parser.add_argument(
        "--weeks_ahead", help="how many weeks ahead to fill", type=int, default=3
    )
    parser.add_argument(
        "--season", help="season, in format e.g. '1819'", type=str, required=True
    )
    parser.add_argument(
        "--num_thread",
        help="number of threads to parallelise over",
        type=int,
        default=4,
    )
    args = parser.parse_args()

    rerun_predictions(
        args.season,
        args.gameweek_start,
        args.gameweek_end,
        args.weeks_ahead,
        args.num_thread,
    )
