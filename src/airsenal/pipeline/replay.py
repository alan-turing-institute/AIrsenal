"""
Replay all or part of a past season, to compare models and algorithms.

A separate driver rather than a method or a subclass of `AIrsenalPipeline`: what
replay needs is the predict and optimise stages, once per gameweek, and none of
the database setup, transfer applying or absence exporting that `run()` does. A
subclass would inherit five things in order to switch four of them off.

It shares the pipeline object, though, which is the point - replay used to build
its own team model with `TEAM_MODELS.create()` and so measured a differently
fitted model than the one `airsenal run` actually uses.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm.session import Session

from airsenal.core.concurrency import set_multiprocessing_start_method
from airsenal.core.console import track
from airsenal.core.logging import get_logger
from airsenal.db.models import Transaction
from airsenal.db.queries.gameweeks import get_max_gameweek
from airsenal.db.queries.players import get_player_name
from airsenal.db.session import session_scope
from airsenal.pipeline.run import AIrsenalPipeline

logger = get_logger(__name__)


@dataclass(frozen=True)
class ReplaySettings:
    """Which part of the season to replay, and how many times."""

    gameweek_start: int = 1
    gameweek_end: int | None = None
    tag_prefix: str = ""
    transfers: bool = True
    loop: int = 1
    # Carry on from the squad already in the database rather than building a new
    # one for the first gameweek.
    resume: bool = False


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


def replay_season(pipeline: AIrsenalPipeline, replay: ReplaySettings) -> None:
    """Replay one season once, writing the results to a JSON file."""
    start = datetime.now()
    season = pipeline.settings.season
    gameweek_end = replay.gameweek_end or get_max_gameweek(season)

    if pipeline.settings.fpl_team_id is None:
        with session_scope() as session:
            pipeline = pipeline.with_settings(
                fpl_team_id=get_dummy_id(season, dbsession=session)
            )
    fpl_team_id = pipeline.settings.fpl_team_id
    if fpl_team_id is None:
        msg = "Could not determine an fpl_team_id to replay under"
        raise RuntimeError(msg)

    tag_prefix = replay.tag_prefix or (
        f"Replay_{season}_GW{replay.gameweek_start}_GW{gameweek_end}_"
        f"{start.strftime('%Y%m%d%H%M')}"
    )
    print_replay_params(
        season, replay.gameweek_start, gameweek_end, tag_prefix, fpl_team_id
    )

    replay_results: dict[str, str | int | float | list[Any]] = {
        "tag": tag_prefix,
        "season": season,
        "n_gameweeks": pipeline.settings.n_gameweeks,
        "gameweeks": [],
    }
    gameweeks_log: list[Any] = replay_results["gameweeks"]  # type: ignore[assignment]

    replay_range = range(replay.gameweek_start, gameweek_end + 1)
    for idx, gw in enumerate(track(replay_range, desc="REPLAY PROGRESS")):
        logger.info("GW%s (%s out of %s)...", gw, idx + 1, len(replay_range))
        # One session per gameweek rather than one for the whole replay: holding
        # a session open across a whole season of model fitting is worse.
        with session_scope() as session:
            gameweeks = pipeline.gameweeks(session, gameweek_start=gw)
            tag = pipeline.predict(gameweeks, session, tag_prefix=tag_prefix)

        if not replay.transfers:
            continue

        # only the first gameweek can start from nothing; after that there is a
        # squad in the database to transfer from
        new_squad = gw == replay.gameweek_start and not replay.resume
        squad, plan = pipeline.with_settings(new_squad=new_squad).optimize(
            gameweeks, tag, fpl_team_id, is_replay=True
        )

        gw_result: dict[str, Any] = {"gameweek": gw, "predictions_tag": tag}
        gw_result["starting_11"] = [p.name for p in squad.players if p.is_starting]
        gw_result["subs"] = [p.name for p in squad.players if not p.is_starting]
        for p in squad.players:
            if p.is_captain:
                gw_result["captain"] = p.name
            elif p.is_vice_captain:
                gw_result["vice_captain"] = p.name

        # A squad built from scratch has no plan: there was nothing to
        # transfer from, and unlimited transfers means no points hit.
        outcome = plan.outcome(gw) if plan is not None else None
        gw_result["free_transfers"] = outcome.free_transfers if outcome else 0
        gw_result["num_transfers"] = outcome.move.label() if outcome else "0"
        gw_result["points_hit"] = outcome.points_hit if outcome else 0
        gw_result["players_in"] = (
            [get_player_name(p) for p in outcome.players_in] if outcome else []
        )
        gw_result["players_out"] = (
            [get_player_name(p) for p in outcome.players_out] if outcome else []
        )

        gw_result["expected_points"] = squad.get_expected_points(gw, tag)
        gw_result["actual_points"] = (
            squad.get_actual_points(gw, season) - gw_result["points_hit"]
        )
        gameweeks_log.append(gw_result)
        logger.info("-" * 30)

    replay_results["elapsed"] = (datetime.now() - start).total_seconds()
    with open(f"{tag_prefix}.json", "w") as outfile:
        json.dump(replay_results, outfile)
    print_replay_params(
        season, replay.gameweek_start, gameweek_end, tag_prefix, fpl_team_id
    )
    logger.info("DONE!")


def run_replays(pipeline: AIrsenalPipeline, replay: ReplaySettings) -> None:
    """Replay a season one or more times."""
    if replay.resume and not pipeline.settings.fpl_team_id:
        msg = "fpl_team_id must be set to use the resume argument"
        raise RuntimeError(msg)

    set_multiprocessing_start_method()

    n_completed = 0
    while (replay.loop == -1) or (n_completed < replay.loop):
        logger.info("*" * 15)
        logger.info("RUNNING REPLAY %s", n_completed + 1)
        logger.info("*" * 15)
        replay_season(pipeline, replay)
        n_completed += 1
