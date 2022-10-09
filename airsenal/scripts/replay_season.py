"""
Script to replay all or part of a season, to allow evaluation of different
code and strategies.
"""
import argparse
import json
import warnings
from datetime import datetime

from tqdm import TqdmWarning, tqdm

from airsenal.framework.multiprocessing_utils import set_multiprocessing_start_method
from airsenal.framework.schema import Transaction, session_scope
from airsenal.framework.utils import get_gameweeks_array, get_max_gameweek
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


def print_replay_params(season, gameweek_start, gameweek_end, tag_prefix, fpl_team_id):
    print("=" * 30)
    print(f"Replay {season} season from GW{gameweek_start} to GW{gameweek_end}")
    print(f"tag_prefix = {tag_prefix}")
    print(f"fpl_team_id = {fpl_team_id}")
    print("=" * 30)


def replay_season(
    season,
    gameweek_start=1,
    gameweek_end=None,
    new_squad=True,
    weeks_ahead=3,
    num_thread=4,
    transfers=True,
    tag_prefix="",
    fpl_team_id=None,
):
    if gameweek_end is None:
        gameweek_end = get_max_gameweek(season)
    if fpl_team_id is None:
        with session_scope() as session:
            fpl_team_id = get_dummy_id(season, dbsession=session)
    if not tag_prefix:
        start = datetime.now().strftime("%Y%m%d%H%M")
        tag_prefix = f"Replay_{season}_GW{gameweek_start}_GW{gameweek_end}_{start}"
    print_replay_params(season, gameweek_start, gameweek_end, tag_prefix, fpl_team_id)

    # store results in a dictionary, which we will later save to a json file
    replay_results = {}
    replay_results["tag"] = tag_prefix
    replay_results["season"] = season
    replay_results["weeks_ahead"] = weeks_ahead
    replay_results["gameweeks"] = []
    replay_range = range(gameweek_start, gameweek_end + 1)
    for idx, gw in enumerate(tqdm(replay_range, desc="REPLAY PROGRESS")):
        print(f"GW{gw} ({idx+1} out of {len(replay_range)})...")
        with session_scope() as session:
            gw_range = get_gameweeks_array(
                weeks_ahead, gameweek_start=gw, season=season, dbsession=session
            )
            tag = make_predictedscore_table(
                gw_range=gw_range,
                season=season,
                num_thread=num_thread,
                tag_prefix=tag_prefix,
                dbsession=session,
            )
        gw_result = {"gameweek": gw, "predictions_tag": tag}

        if not transfers:
            continue
        if gw == gameweek_start and new_squad:
            print("Creating initial squad...")
            squad = fill_initial_squad(tag, gw_range, season, fpl_team_id)
        else:
            print("Optimising transfers...")
            squad = run_optimization(
                gw_range,
                tag,
                season=season,
                fpl_team_id=fpl_team_id,
                num_thread=num_thread,
            )
        gw_result["starting_11"] = []
        gw_result["subs"] = []
        for p in squad.players:
            if p.is_starting:
                gw_result["starting_11"].append(p.name)
            else:
                gw_result["subs"].append(p.name)
            if p.is_captain:
                gw_result["captain"] = p.name
            elif p.is_vice_captain:
                gw_result["vice_captain"] = p.name
        exp_points = squad.get_expected_points(gw, tag)
        gw_result["expected_points"] = exp_points
        actual_points = squad.get_actual_points(gw, season)
        gw_result["actual_points"] = actual_points
        replay_results["gameweeks"].append(gw_result)
        print("-" * 30)
    with open(f"{tag_prefix}.json", "w") as outfile:
        json.dump(replay_results, outfile)
    print_replay_params(season, gameweek_start, gameweek_end, tag_prefix, fpl_team_id)
    print("DONE!")


def main():
    parser = argparse.ArgumentParser(description="fill player predictions")

    parser.add_argument(
        "--gameweek_start", help="first gameweek to look at", type=int, default=1
    )
    parser.add_argument(
        "--gameweek_end", help="last gameweek to look at", type=int, default=None
    )
    parser.add_argument(
        "--weeks_ahead", help="how many weeks ahead to fill", type=int, default=3
    )
    parser.add_argument(
        "--season", help="season, in format e.g. '1819'", type=str, required=True
    )
    parser.add_argument(
        "--fpl_team_id",
        help="FPL team ID (defaults to a unique, negative value)",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--resume",
        help=(
            "If set, use a pre-existing squad and transactions in the database "
            "for this team ID as the starting point, rather than creating a new squad. "
            "fpl_team_id must be defined."
        ),
        action="store_true",
    )
    parser.add_argument(
        "--num_thread",
        help="number of threads to parallelise over",
        type=int,
        default=4,
    )
    args = parser.parse_args()
    if args.resume and not args.fpl_team_id:
        raise RuntimeError("fpl_team_id must be set to use the resume argument")

    set_multiprocessing_start_method()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TqdmWarning)
        replay_season(
            season=args.season,
            gameweek_start=args.gameweek_start,
            gameweek_end=args.gameweek_end,
            new_squad=not args.resume,
            weeks_ahead=args.weeks_ahead,
            num_thread=args.num_thread,
            fpl_team_id=args.fpl_team_id,
        )


if __name__ == "__main__":
    main()
