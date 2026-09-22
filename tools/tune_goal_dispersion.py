"""
Tune how widely goals scatter around the rate a team model predicts.

A sweep over the Conway-Maxwell-Poisson dispersion of `ConwayMaxwellScorelines`,
scored by `prediction.evaluation.backtest_team_model`: for each gameweek, fit on
the matches before it and score the scorelines of the next `--horizon`
gameweeks. One is exactly a Poisson, above one is narrower than a Poisson.

The dispersion is swept rather than fitted inside the model because one fitted
to the same matches as the ratings comes out too narrow. Each season is reported
as well as the pooled score, since their optima differ.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from airsenal.core.console import track
from airsenal.core.logging import configure_logging, get_logger
from airsenal.db.queries.gameweeks import get_max_gameweek
from airsenal.db.session import session_scope
from airsenal.game.season import CURRENT_SEASON
from airsenal.prediction.evaluation import ModelScore, backtest_team_model
from airsenal.prediction.protocols import ScorelineTeamModel
from airsenal.prediction.team_models.scorelines import (
    DEFAULT_GOAL_DISPERSION,
    ConwayMaxwellScorelines,
)
from airsenal.prediction.team_models.xg import XGTeamConfig, XGTeamModel

logger = get_logger(__name__)


def build(dispersion: float, epsilon: float | None = None) -> ScorelineTeamModel:
    """The xG model with one dispersion, built fresh because a fit is in place."""
    config = XGTeamConfig() if epsilon is None else XGTeamConfig(epsilon=epsilon)
    return ConwayMaxwellScorelines(XGTeamModel(config), dispersion=dispersion)


def evaluate_dispersion(
    dispersion: float,
    seasons: list[str],
    horizon: int,
    epsilon: float | None = None,
    first_gameweek: int = 5,
    last_gameweek: int | None = None,
) -> dict[str, ModelScore]:
    """Score one dispersion on each season separately, walking each forward."""
    scores = {}
    for season in seasons:
        with session_scope() as dbsession:
            max_gameweek = get_max_gameweek(season=season, dbsession=dbsession)
            gameweek_end = (
                (
                    min(last_gameweek, max_gameweek)
                    if last_gameweek is not None
                    else max_gameweek
                )
                - horizon
                + 1
            )
            scores[season] = backtest_team_model(
                lambda: build(dispersion, epsilon),
                season=season,
                dbsession=dbsession,
                gameweeks=range(first_gameweek, gameweek_end + 1),
                horizon=horizon,
            )
    return scores


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="*", default=[CURRENT_SEASON])
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--dispersions", type=float, nargs="*", default=None)
    parser.add_argument("--first-gameweek", type=int, default=5)
    parser.add_argument("--last-gameweek", type=int, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    grid = args.dispersions or list(np.arange(1.0, 1.35, 0.05))
    rows: list[dict[str, float | int]] = []
    for dispersion in track(grid, description="Dispersion:"):
        scores = evaluate_dispersion(
            float(dispersion),
            seasons=args.seasons,
            horizon=args.horizon,
            epsilon=args.epsilon,
            first_gameweek=args.first_gameweek,
            last_gameweek=args.last_gameweek,
        )
        pooled = sum(scores.values(), ModelScore())
        row: dict[str, float | int] = {
            "dispersion": float(dispersion),
            "pooled_log_prob": pooled.mean_log_probability,
            "n_fixtures": pooled.n_observations,
        }
        row.update({season: s.mean_log_probability for season, s in scores.items()})
        rows.append(row)
        logger.info(
            "dispersion=%.3f  pooled %.5f  %s",
            dispersion,
            pooled.mean_log_probability,
            "  ".join(f"{s} {v.mean_log_probability:.5f}" for s, v in scores.items()),
        )

    best = max(rows, key=lambda r: r["pooled_log_prob"])
    logger.info(
        "Best pooled dispersion: %.3f (%.5f); shipped default is %.3f",
        best["dispersion"],
        best["pooled_log_prob"],
        DEFAULT_GOAL_DISPERSION,
    )
    out = args.out_csv or Path(f"tune_dispersion_{'_'.join(args.seasons)}.csv")
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
