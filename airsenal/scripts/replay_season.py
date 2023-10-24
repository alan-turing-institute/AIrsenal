"""
Script to replay all or part of a season, to allow evaluation of different
code and strategies.
"""
import argparse
import json
import warnings
from datetime import datetime
from typing import Optional

from sqlalchemy.orm.session import Session
from tqdm import TqdmWarning, tqdm

from airsenal.framework.multiprocessing_utils import set_multiprocessing_start_method
from airsenal.framework.schema import Transaction, session_scope
from airsenal.framework.utils import (
    get_gameweeks_array,
    get_max_gameweek,
    get_player_name,
    parse_team_model_from_str,
)
from airsenal.scripts.fill_predictedscore_table import make_predictedscore_table
from airsenal.scripts.fill_transfersuggestion_table import run_optimization
from airsenal.scripts.squad_builder import fill_initial_squad


def get_dummy_id(season: str, dbsession: Session) -> int:
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


def print_replay_params(
    season: str,
    gameweek_start: int,
    gameweek_end: int,
    tag_prefix: str,
    fpl_team_id: int,
) -> None:
    print("=" * 30)
    print(f"Replay {season} season from GW{gameweek_start} to GW{gameweek_end}")
    print(f"tag_prefix = {tag_prefix}")
    print(f"fpl_team_id = {fpl_team_id}")
    print("=" * 30)


def replay_season(
    season: str,
    gameweek_start: int = 1,
    gameweek_end: Optional[int] = None,
    new_squad: bool = True,
    weeks_ahead: int = 3,
    num_thread: int = 4,
    transfers: bool = True,
    tag_prefix: str = "",
    team_model: str = "extended",
    team_model_args: dict = {"epsilon": 0.0},
    fpl_team_id: Optional[int] = None,
) -> None:
    start = datetime.now()
    if gameweek_end is None:
        gameweek_end = get_max_gameweek(season)
    if fpl_team_id is None:
        with session_scope() as session:
            fpl_team_id = get_dummy_id(season, dbsession=session)
    if not tag_prefix:
        start_str = start.strftime("%Y%m%d%H%M")
        tag_prefix = (
            f"Replay_{season}_GW{gameweek_start}_GW{gameweek_end}_"
            f"{start_str}_{team_model}"
        )
    print_replay_params(season, gameweek_start, gameweek_end, tag_prefix, fpl_team_id)

    team_model_class = parse_team_model_from_str(team_model)

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
                team_model=team_model_class,
                team_model_args=team_model_args,
                dbsession=session,
            )
        gw_result = {"gameweek": gw, "predictions_tag": tag}

        if not transfers:
            continue
        if gw == gameweek_start and new_squad:
            print("Creating initial squad...")
            squad = fill_initial_squad(
                tag, gw_range, season, fpl_team_id, is_replay=True
            )
            # no points hits due to unlimited transfers to initialise team
            best_strategy = {
                "points_hit": {str(gw): 0},
                "free_transfers": {str(gw): 0},
                "num_transfers": {str(gw): 0},
                "players_in": {str(gw): []},
                "players_out": {str(gw): []},
            }
        else:
            print("Optimising transfers...")
            # find best squad and the strategy for this gameweek
            squad, best_strategy = run_optimization(
                gw_range,
                tag,
                season=season,
                fpl_team_id=fpl_team_id,
                num_thread=num_thread,
                is_replay=True,
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
        # obtain information about the strategy used for gameweek
        gw_result["free_transfers"] = best_strategy["free_transfers"][str(gw)]
        gw_result["num_transfers"] = best_strategy["num_transfers"][str(gw)]
        gw_result["points_hit"] = best_strategy["points_hit"][str(gw)]
        gw_result["players_in"] = [
            get_player_name(p) for p in best_strategy["players_in"][str(gw)]
        ]
        gw_result["players_out"] = [
            get_player_name(p) for p in best_strategy["players_out"][str(gw)]
        ]
        # compute expected and actual points for gameweek
        exp_points = squad.get_expected_points(gw, tag)
        gw_result["expected_points"] = exp_points - gw_result["points_hit"]
        actual_points = squad.get_actual_points(gw, season)
        gw_result["actual_points"] = actual_points - gw_result["points_hit"]
        replay_results["gameweeks"].append(gw_result)
        print("-" * 30)

    end = datetime.now()
    elapsed = end - start
    replay_results["elapsed"] = elapsed.total_seconds()
    with open(f"{tag_prefix}.json", "w") as outfile:
        json.dump(replay_results, outfile)
    print_replay_params(season, gameweek_start, gameweek_end, tag_prefix, fpl_team_id)
    print("DONE!")


def main():
    """
    replay a particular FPL season
    """
    parser = argparse.ArgumentParser(description="replay a particular FPL season")

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
    parser.add_argument(
        "--loop",
        help="How many times to repeat repla (default 1, -1 to loop continuously)",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--team_model",
        help="Specify name of the team model.",
        type=str,
        default="extended",
        choices=["extended", "random"],
    )
    parser.add_argument(
        "--epsilon",
        help="how much to downweight games by in exponential time weighting",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()
    if args.resume and not args.fpl_team_id:
        raise RuntimeError("fpl_team_id must be set to use the resume argument")

    set_multiprocessing_start_method()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TqdmWarning)
        n_completed = 0
        while (args.loop == -1) or (n_completed < args.loop):
            print("*" * 15)
            print(f"RUNNING REPLAY {n_completed + 1}")
            print("*" * 15)
            replay_season(
                season=args.season,
                gameweek_start=args.gameweek_start,
                gameweek_end=args.gameweek_end,
                new_squad=not args.resume,
                weeks_ahead=args.weeks_ahead,
                num_thread=args.num_thread,
                fpl_team_id=args.fpl_team_id,
                team_model=args.team_model,
                team_model_args={"epsilon": args.epsilon},
            )
            n_completed += 1


if __name__ == "__main__":
    main()
