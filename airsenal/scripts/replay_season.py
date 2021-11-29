"""
Script to replay all or part of a season, to allow evaluation of different
code and strategies.
"""
import argparse
import warnings
from datetime import datetime

from tqdm import TqdmWarning, tqdm

from airsenal.framework.multiprocessing_utils import set_multiprocessing_start_method
from airsenal.framework.schema import Transaction, session_scope
from airsenal.framework.utils import get_max_gameweek
from airsenal.scripts.fill_predictedscore_table import make_predictedscore_table
from airsenal.scripts.fill_transfersuggestion_table import run_optimization
from airsenal.scripts.squad_builder import fill_initial_squad


def get_dummy_id(season, dbsession):
    team_ids = [
        item[0]
        for item in dbsession.query(Transaction.fpl_team_id)
        .filter_by(season=season)
        .distinct()
        .all()
    ]
    if not team_ids or min(team_ids) > 0:
        return -1
    return min(team_ids) - 1


def print_replay_params(season, gw_start, gw_end, tag_prefix, fpl_team_id):
    print("=" * 30)
    print(f"Replay {season} season from GW{gw_start} to GW{gw_end}")
    print(f"tag_prefix = {tag_prefix}")
    print(f"fpl_team_id = {fpl_team_id}")
    print("=" * 30)


def replay_season(
    season,
    gw_start=1,
    gw_end=None,
    new_squad=True,
    weeks_ahead=3,
    num_thread=4,
    transfers=True,
    tag_prefix="",
    fpl_team_id=None,
):
    if gw_end is None:
        gw_end = get_max_gameweek(season)
    if fpl_team_id is None:
        with session_scope() as session:
            fpl_team_id = get_dummy_id(season, dbsession=session)
    if not tag_prefix:
        start = datetime.now().strftime("%Y%m%d%H%M")
        tag_prefix = f"Replay_{season}_GW{gw_start}_GW{gw_end}_{start}"
    print_replay_params(season, gw_start, gw_end, tag_prefix, fpl_team_id)

    replay_range = range(gw_start, gw_end + 1)
    for idx, gw in enumerate(tqdm(replay_range)):
        print(f"GW{gw} ({idx+1} out of {len(replay_range)})...")
        gw_range = range(gw, gw + weeks_ahead)
        with session_scope() as session:
            tag = make_predictedscore_table(
                gw_range=gw_range,
                season=season,
                num_thread=num_thread,
                tag_prefix=tag_prefix,
                dbsession=session,
            )
        if not transfers:
            continue
        if gw == gw_start and new_squad:
            print("Creating initial squad...")
            fill_initial_squad(tag, gw_range, season, fpl_team_id)
        else:
            print("Optimising transfers...")
            run_optimization(
                gw_range,
                tag,
                season=season,
                fpl_team_id=fpl_team_id,
                num_thread=num_thread,
            )
        print("-" * 30)
    print_replay_params(season, gw_start, gw_end, tag_prefix, fpl_team_id)
    print("DONE!")


def main():
    parser = argparse.ArgumentParser(description="fill player predictions")

    parser.add_argument(
        "--gw_start", help="first gameweek to look at", type=int, default=1
    )
    parser.add_argument(
        "--gw_end", help="last gameweek to look at", type=int, default=None
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
    set_multiprocessing_start_method(args.num_thread)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TqdmWarning)
        replay_season(
            season=args.season,
            gw_start=args.gw_start,
            gw_end=args.gw_end,
            weeks_ahead=args.weeks_ahead,
            num_thread=args.num_thread,
        )


if __name__ == "__main__":
    main()
