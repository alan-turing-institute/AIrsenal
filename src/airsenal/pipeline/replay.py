"""
Replay all or part of a past season, to compare models and algorithms.

A separate driver rather than a method or a subclass of `AIrsenalPipeline`: what
replay needs is the predict and optimise stages, once per gameweek, and none of
the database setup, transfer applying or absence exporting that `run()` does.

It takes the pipeline object itself, though, so that what it measures is the
same components `airsenal run` would use - and it records which ones they were,
because a score is only comparable against another score if you know what each
was produced with.
"""

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
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
from airsenal.game.enums import Chip
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
    # Where the result JSON is written. The process's working directory, as
    # before, when this is None.
    output_dir: Path | None = None


@dataclass(frozen=True)
class ReplayGameweek:
    """
    What one replayed gameweek picked, and what it scored.

    Named for the replay rather than the gameweek: `optimization.plan` already
    has a `GameweekOutcome`, which is what a *plan* expects a gameweek to do.
    This is what actually happened.
    """

    gameweek: int
    predictions_tag: str
    starting_11: list[str]
    subs: list[str]
    captain: str | None
    vice_captain: str | None
    free_transfers: int
    num_transfers: str
    points_hit: int
    players_in: list[str]
    players_out: list[str]
    expected_points: float
    # Already net of the points hit, so it is what the entry actually scored.
    actual_points: float
    # Which chip the plan played, if any. Both point totals above are scored with
    # it, so a replay that cannot say which chip it used cannot be read back.
    chip_played: str | None = None

    @property
    def prediction_error(self) -> float:
        """Signed points the prediction was out by, expected minus actual."""
        return self.expected_points - self.actual_points


@dataclass(frozen=True)
class ReplayResult:
    """
    What one replay of a season scored, and what produced it.

    The comparable summary of a run: `total_points` is the number two replays are
    judged on, and `config` says what each was run with, so a pair of results is
    self-describing without anyone having to remember which flags they used.
    """

    tag: str
    season: str
    n_gameweeks: int | None
    config: dict[str, str]
    gameweeks: list[ReplayGameweek] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def total_points(self) -> float:
        """Points scored across the replay, net of hits. The headline number."""
        return sum(gw.actual_points for gw in self.gameweeks)

    @property
    def total_points_hit(self) -> int:
        return sum(gw.points_hit for gw in self.gameweeks)

    @property
    def total_expected_points(self) -> float:
        return sum(gw.expected_points for gw in self.gameweeks)

    @property
    def mean_absolute_error(self) -> float:
        """
        Average size of the gap between a gameweek's prediction and its outcome.

        Zero when the replay covered no gameweeks. Unlike `total_points` this
        judges the prediction alone, so a model can be compared without the
        optimizer's choices being part of the answer.
        """
        if not self.gameweeks:
            return 0.0
        return sum(abs(gw.prediction_error) for gw in self.gameweeks) / len(
            self.gameweeks
        )

    def as_dict(self) -> dict[str, Any]:
        """
        The JSON payload, summary first.

        The per-gameweek keys are unchanged from when this was assembled by hand,
        so anything already reading a replay file keeps working.
        """
        return {
            "tag": self.tag,
            "season": self.season,
            "n_gameweeks": self.n_gameweeks,
            "config": self.config,
            "total_points": self.total_points,
            "total_points_hit": self.total_points_hit,
            "total_expected_points": self.total_expected_points,
            "mean_absolute_error": self.mean_absolute_error,
            "elapsed": self.elapsed,
            "gameweeks": [asdict(gw) for gw in self.gameweeks],
        }

    def write(self, directory: Path | None = None) -> Path:
        """Write the result as JSON named after the tag, and return the path."""
        directory = Path(directory) if directory is not None else Path.cwd()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.tag}.json"
        with path.open("w") as outfile:
            json.dump(self.as_dict(), outfile, indent=2)
        return path


def describe_pipeline(pipeline: AIrsenalPipeline) -> dict[str, str]:
    """
    The components a replay ran with, by class name.

    Class names rather than the table names they were built from: a component
    constructed in Python never had a table name, and the point is to be able to
    tell two results apart.
    """
    return {
        "team_model": type(pipeline.team_model).__name__,
        "player_model": type(pipeline.player_model).__name__,
        "transfer_optimizer": type(pipeline.transfer_optimizer).__name__,
        "squad_optimizer": type(pipeline.squad_optimizer).__name__,
    }


def default_tag_prefix(
    season: str, gameweek_start: int, gameweek_end: int, when: datetime
) -> str:
    """
    The tag a replay is named after when the caller does not supply one.

    Resolved here rather than inside `replay_season` so that `run_replays` can
    build it once and number the runs off it: the timestamp is only accurate to
    the minute, and two replays of a short window finish inside the same minute
    and would otherwise write over each other's results.
    """
    return (
        f"Replay_{season}_GW{gameweek_start}_GW{gameweek_end}_"
        f"{when.strftime('%Y%m%d%H%M')}"
    )


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


def _names(player_ids: list[int]) -> list[str]:
    """
    Player names for a transfer list, falling back to the id.

    `get_player_name` returns None for an id the database does not know, and a
    replay record that silently dropped a transfer would be worse than one
    naming a number.
    """
    return [get_player_name(pid) or f"player_{pid}" for pid in player_ids]


def _gameweek_outcome(
    tag: str,
    gameweek: int,
    squad: Any,
    plan: Any,
    season: str,
) -> ReplayGameweek:
    """One gameweek's row, from the squad and plan the pipeline produced."""
    # A squad built from scratch has no plan: there was nothing to transfer from,
    # and unlimited transfers means no points hit.
    outcome = plan.outcome(gameweek) if plan is not None else None
    points_hit = outcome.points_hit if outcome else 0
    # A bench boost scores the bench too and a triple captain trebles rather than
    # doubles. Scoring a chip gameweek as though no chip were played understates
    # exactly the weeks a chip was meant to win.
    chip = outcome.chip if outcome else None
    bench_boost = chip is Chip.BENCH_BOOST
    triple_captain = chip is Chip.TRIPLE_CAPTAIN
    return ReplayGameweek(
        gameweek=gameweek,
        predictions_tag=tag,
        starting_11=[p.name for p in squad.players if p.is_starting],
        subs=[p.name for p in squad.players if not p.is_starting],
        captain=next((p.name for p in squad.players if p.is_captain), None),
        vice_captain=next((p.name for p in squad.players if p.is_vice_captain), None),
        free_transfers=outcome.free_transfers if outcome else 0,
        num_transfers=outcome.move.label() if outcome else "0",
        points_hit=points_hit,
        players_in=_names(outcome.players_in) if outcome else [],
        players_out=_names(outcome.players_out) if outcome else [],
        expected_points=squad.get_expected_points(
            tag, gameweek, bench_boost=bench_boost, triple_captain=triple_captain
        ),
        actual_points=squad.get_actual_points(
            gameweek,
            season,
            bench_boost=bench_boost,
            triple_captain=triple_captain,
        )
        - points_hit,
        chip_played=str(chip) if chip else None,
    )


def replay_season(pipeline: AIrsenalPipeline, replay: ReplaySettings) -> ReplayResult:
    """
    Replay one season once, and return what it scored.

    Also writes the result as JSON, to `replay.output_dir` or the working
    directory. The object is returned as well as written so that a caller
    comparing two configurations does not have to read its own output back.
    """
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

    tag_prefix = replay.tag_prefix or default_tag_prefix(
        season, replay.gameweek_start, gameweek_end, start
    )
    print_replay_params(
        season, replay.gameweek_start, gameweek_end, tag_prefix, fpl_team_id
    )

    outcomes: list[ReplayGameweek] = []
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
        outcomes.append(_gameweek_outcome(tag, gw, squad, plan, season))
        logger.info("-" * 30)

    result = ReplayResult(
        tag=tag_prefix,
        season=season,
        n_gameweeks=pipeline.settings.n_gameweeks,
        config=describe_pipeline(pipeline),
        gameweeks=outcomes,
        elapsed=(datetime.now() - start).total_seconds(),
    )
    path = result.write(replay.output_dir)
    print_replay_params(
        season, replay.gameweek_start, gameweek_end, tag_prefix, fpl_team_id
    )
    logger.info("Scored %s points. Written to %s", result.total_points, path)
    return result


def run_replays(
    pipeline: AIrsenalPipeline, replay: ReplaySettings
) -> list[ReplayResult]:
    """Replay a season one or more times, and return what each one scored."""
    if replay.resume and not pipeline.settings.fpl_team_id:
        msg = "fpl_team_id must be set to use the resume argument"
        raise RuntimeError(msg)

    set_multiprocessing_start_method()

    base = replay.tag_prefix or default_tag_prefix(
        pipeline.settings.season,
        replay.gameweek_start,
        replay.gameweek_end or get_max_gameweek(pipeline.settings.season),
        datetime.now(),
    )

    results: list[ReplayResult] = []
    while (replay.loop == -1) or (len(results) < replay.loop):
        run = len(results) + 1
        logger.info("*" * 15)
        logger.info("RUNNING REPLAY %s", run)
        logger.info("*" * 15)
        # numbered only when there is more than one, so a single replay keeps the
        # name the caller asked for
        tag_prefix = base if replay.loop == 1 else f"{base}_run{run}"
        results.append(replay_season(pipeline, replace(replay, tag_prefix=tag_prefix)))
    return results
