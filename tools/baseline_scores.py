"""
Score every team and player model, to compare before and after a prediction change.

Every team and player model in the tables is scored by held-out log probability,
and with `--points` the whole points calculation is scored by the error in the
points it predicts. Written as JSON so two runs can be diffed.

`--points` writes predictions to the database, one tag per gameweek, so point it
at a copy rather than the database you run with.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from airsenal.core.console import track
from airsenal.core.logging import configure_logging, get_logger
from airsenal.db.session import session_scope
from airsenal.game.enums import Position
from airsenal.game.season import CURRENT_SEASON
from airsenal.prediction.evaluation import (
    backtest_player_model,
    backtest_points,
    backtest_team_model,
)
from airsenal.prediction.player_models import PLAYER_MODELS
from airsenal.prediction.team_models import TEAM_MODELS

logger = get_logger(__name__)


def score_models(
    horizon: int = 1,
    *,
    season: str,
    gameweeks: list[int],
) -> dict[str, dict[str, Any]]:
    """
    Every team and player model in the tables, by held-out log probability.

    Player models are scored one position at a time, because not every model can
    be fitted for every position: goalkeepers who scored no goals in the window
    leave the Dirichlet prior improper, so `numpyro` cannot be fitted there. A
    model that raises is recorded with its error rather than left out.
    """
    scores: dict[str, dict[str, Any]] = {}
    for name in track(sorted(TEAM_MODELS), description="Team models:"):
        with session_scope() as dbsession:
            score = backtest_team_model(
                TEAM_MODELS[name],
                season=season,
                dbsession=dbsession,
                gameweeks=gameweeks,
                horizon=horizon,
            )
        scores[f"team:{name}"] = {
            "mean_log_probability": score.mean_log_probability,
            "n_observations": score.n_observations,
            "n_skipped": score.n_skipped,
        }
        logger.info("team:%s %.5f", name, score.mean_log_probability)

    entries = [
        (name, position) for name in sorted(PLAYER_MODELS) for position in Position
    ]
    for name, position in track(entries, description="Player models:"):
        key = f"player:{name}:{position}"
        try:
            with session_scope() as dbsession:
                score = backtest_player_model(
                    PLAYER_MODELS[name],
                    season=season,
                    dbsession=dbsession,
                    gameweeks=gameweeks,
                    positions=[position],
                    horizon=horizon,
                )
        except ValueError as error:
            scores[key] = {"error": str(error)}
            logger.warning("%s could not be fitted: %s", key, error)
            continue
        scores[key] = {
            "mean_log_probability": score.mean_log_probability,
            "n_observations": score.n_observations,
            "n_skipped": score.n_skipped,
        }
        logger.info("%s %.5f", key, score.mean_log_probability)
    return scores


def score_points(
    horizon: int = 1,
    *,
    season: str,
    gameweeks: list[int],
) -> dict[str, float | int]:
    """The whole points calculation, with the default models."""
    with session_scope() as dbsession:
        score = backtest_points(
            season=season, dbsession=dbsession, gameweeks=gameweeks, horizon=horizon
        )
    logger.info(
        "points: MAE %.4f (%.4f for players who appeared) RMSE %.4f rank %.4f "
        "over %s performances",
        score.mean_absolute_error,
        score.mean_absolute_error_played,
        score.root_mean_squared_error,
        score.mean_rank_correlation,
        score.n_observations,
    )
    return {
        "mean_absolute_error": score.mean_absolute_error,
        "mean_absolute_error_played": score.mean_absolute_error_played,
        "root_mean_squared_error": score.root_mean_squared_error,
        "mean_rank_correlation": score.mean_rank_correlation,
        "n_observations": score.n_observations,
        "n_played": score.n_played,
        "n_skipped": score.n_skipped,
    }


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default=CURRENT_SEASON)
    parser.add_argument("--first-gameweek", type=int, default=5)
    parser.add_argument("--last-gameweek", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--points",
        action="store_true",
        help="also backtest the points calculation - this writes to the database",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    gameweeks = list(range(args.first_gameweek, args.last_gameweek + 1))
    result: dict[str, Any] = {
        "config": {
            "season": args.season,
            "gameweek_start": args.first_gameweek,
            "gameweek_end": args.last_gameweek,
            "horizon": args.horizon,
        },
        "models": score_models(args.horizon, season=args.season, gameweeks=gameweeks),
    }
    if args.points:
        result["points"] = score_points(
            args.horizon, season=args.season, gameweeks=gameweeks
        )

    out = args.out_json or Path(
        f"baseline_{args.season}_GW{args.first_gameweek}_GW{args.last_gameweek}.json"
    )
    with out.open("w") as handle:
        json.dump(result, handle, indent=2)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
