"""
Script to replay all or part of a season, to allow evaluation of different
code and strategies.
"""

import json
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm.session import Session

from airsenal.core.concurrency import set_multiprocessing_start_method
from airsenal.core.console import track
from airsenal.core.logging import get_logger
from airsenal.db.models import Transaction
from airsenal.db.queries.gameweeks import get_gameweeks_array, get_max_gameweek
from airsenal.db.queries.players import get_player_name
from airsenal.db.session import session_scope
from airsenal.optimization.moves import GameweekMove
from airsenal.optimization.run_squad import fill_initial_squad
from airsenal.optimization.run_transfers import run_optimization
from airsenal.optimization.strategy import GameweekOutcome, Strategy
from airsenal.prediction.registry import PLAYER_MODELS, TEAM_MODELS
from airsenal.prediction.run import make_predictedscore_table
from airsenal.prediction.team_models.dixon_coles import DEFAULT_TEAM_EPSILON

logger = get_logger(__name__)


def get_dummy_id(season: str, dbsession: Session) -> int:
    team_ids = dbsession.scalars(
        select(Transaction.fpl_team_id).where(Transaction.season == season).distinct()
    ).all()
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
    logger.info("=" * 30)
    logger.info(
        "Replay %s season from GW%s to GW%s", season, gameweek_start, gameweek_end
    )
    logger.info("tag_prefix = %s", tag_prefix)
    logger.info("fpl_team_id = %s", fpl_team_id)
    logger.info("=" * 30)


def replay_season(
    season: str,
    gameweek_start: int = 1,
    gameweek_end: int | None = None,
    new_squad: bool = True,
    n_gameweeks: int = 3,
    num_thread: int = 4,
    transfers: bool = True,
    tag_prefix: str = "",
    team_model: str = "extended",
    team_model_args: dict[str, Any] | None = None,
    fpl_team_id: int | None = None,
    max_opt_transfers: int = 2,
    player_model: str = "conjugate",
    player_model_options: dict[str, str] | None = None,
) -> None:
    if team_model_args is None:
        team_model_args = {"epsilon": DEFAULT_TEAM_EPSILON}
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

    fitted_player_model = PLAYER_MODELS.create_with(
        player_model, player_model_options or {}
    )
    fitted_team_model = TEAM_MODELS.create(team_model)

    # store results in a dictionary, which we will later save to a json file
    replay_results: dict[str, str | int | float | list[Any]] = {}
    replay_results["tag"] = tag_prefix
    replay_results["season"] = season
    replay_results["n_gameweeks"] = n_gameweeks
    replay_results["gameweeks"] = []
    replay_range = range(gameweek_start, gameweek_end + 1)
    for idx, gw in enumerate(track(replay_range, desc="REPLAY PROGRESS")):
        logger.info("GW%s (%s out of %s)...", gw, idx + 1, len(replay_range))
        with session_scope() as session:
            gameweeks = get_gameweeks_array(
                n_gameweeks, gameweek_start=gw, season=season, dbsession=session
            )
            tag = make_predictedscore_table(
                gameweeks=gameweeks,
                season=season,
                tag_prefix=tag_prefix,
                player_model=fitted_player_model,
                team_model=fitted_team_model,
                team_model_args=team_model_args,
                dbsession=session,
            )
        gw_result = {"gameweek": gw, "predictions_tag": tag}

        if not transfers:
            continue
        if gw == gameweek_start and new_squad:
            logger.info("Creating initial squad...")
            squad = fill_initial_squad(
                tag, gameweeks, season, fpl_team_id, is_replay=True
            )
            # no points hits due to unlimited transfers to initialise team
            best_strategy: Strategy | None = Strategy(
                root_gameweek=gw,
                outcomes=(
                    GameweekOutcome(
                        gameweek=gw,
                        move=GameweekMove(),
                        points=0.0,
                        discount_factor=1.0,
                        points_hit=0,
                        free_transfers=0,
                    ),
                ),
            )
        else:
            logger.info("Optimising transfers...")
            # find best squad and the strategy for this gameweek
            squad, best_strategy = run_optimization(
                gameweeks,
                tag,
                season=season,
                fpl_team_id=fpl_team_id,
                num_thread=num_thread,
                is_replay=True,
                max_opt_transfers=max_opt_transfers,
            )
        if best_strategy is None:
            msg = f"Failed to find a strategy for GW{gw}!"
            raise ValueError(msg)

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
        outcome = best_strategy.outcome(gw)
        gw_result["free_transfers"] = outcome.free_transfers
        gw_result["num_transfers"] = outcome.move.label()
        gw_result["points_hit"] = outcome.points_hit
        gw_result["players_in"] = [get_player_name(p) for p in outcome.players_in]
        gw_result["players_out"] = [get_player_name(p) for p in outcome.players_out]
        # compute expected and actual points for gameweek
        gw_result["expected_points"] = squad.get_expected_points(gw, tag)
        actual_points = squad.get_actual_points(gw, season)
        gw_result["actual_points"] = actual_points - gw_result["points_hit"]
        if not isinstance(replay_results["gameweeks"], list):
            msg = (
                f"replay_results['gameweeks'] should be a list, "
                f"got {type(replay_results['gameweeks'])}"
            )
            raise TypeError(msg)
        replay_results["gameweeks"].append(gw_result)
        logger.info("-" * 30)

    end = datetime.now()
    elapsed = end - start
    replay_results["elapsed"] = elapsed.total_seconds()
    with open(f"{tag_prefix}.json", "w") as outfile:
        json.dump(replay_results, outfile)
    print_replay_params(season, gameweek_start, gameweek_end, tag_prefix, fpl_team_id)
    logger.info("DONE!")


def run_replays(
    season: str,
    gameweek_start: int,
    gameweek_end: int | None,
    n_gameweeks: int,
    fpl_team_id: int | None,
    resume: bool,
    num_thread: int,
    loop: int,
    team_model: str,
    epsilon: float | None,
    max_transfers: int,
    player_model: str = "conjugate",
    player_model_options: dict[str, str] | None = None,
) -> None:
    """Replay a season one or more times."""
    if resume and not fpl_team_id:
        msg = "fpl_team_id must be set to use the resume argument"
        raise RuntimeError(msg)

    set_multiprocessing_start_method()

    n_completed = 0
    while (loop == -1) or (n_completed < loop):
        logger.info("*" * 15)
        logger.info("RUNNING REPLAY %s", n_completed + 1)
        logger.info("*" * 15)
        replay_season(
            season=season,
            gameweek_start=gameweek_start,
            gameweek_end=gameweek_end,
            new_squad=not resume,
            n_gameweeks=n_gameweeks,
            num_thread=num_thread,
            fpl_team_id=fpl_team_id,
            team_model=team_model,
            team_model_args=({"epsilon": epsilon} if epsilon is not None else {}),
            max_opt_transfers=max_transfers,
            player_model=player_model,
            player_model_options=player_model_options,
        )
        n_completed += 1
