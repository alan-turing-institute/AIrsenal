"""
Tune the team model's time-weighting (epsilon) hyperparameter.

A sweep over epsilon, scored by `prediction.evaluation.backtest_team_model`:
for each gameweek, fit on the matches before it and score the scorelines of the
next `--horizon` gameweeks.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from airsenal.core.console import track
from airsenal.core.logging import get_logger
from airsenal.db.queries.gameweeks import get_max_gameweek
from airsenal.db.session import session_scope
from airsenal.game.season import CURRENT_SEASON
from airsenal.prediction.evaluation import ModelScore, backtest_team_model
from airsenal.prediction.team_models import DEFAULT_TEAM_MODEL, build_team_model

logger = get_logger(__name__)


@dataclass(frozen=True)
class EpsilonResult:
    """What one epsilon scored, over how many fixtures."""

    epsilon: float
    score: ModelScore

    def as_row(self) -> dict[str, float | int]:
        return {
            "epsilon": self.epsilon,
            "total_log_prob": self.score.total_log_probability,
            "num_fixtures": self.score.n_observations,
            "avg_log_prob": self.score.mean_log_probability,
        }


def evaluate_epsilon(
    epsilon: float,
    seasons: list[str],
    horizon: int,
    model_name: str = DEFAULT_TEAM_MODEL,
    first_gw: int | None = None,
    last_gw: int | None = None,
) -> EpsilonResult:
    """Score one epsilon across every season, walking each one forward."""
    total = ModelScore()
    for season in track(seasons, desc="Season"):
        with session_scope() as dbsession:
            max_gw = get_max_gameweek(season=season, dbsession=dbsession)
            start_gw = first_gw or 1
            end_gw = min(last_gw if last_gw is not None else max_gw, max_gw) - horizon
            if end_gw < start_gw:
                msg = (
                    f"Invalid gameweek window: start={start_gw}, end={end_gw}, "
                    f"max_gw={max_gw}, horizon={horizon}"
                )
                raise ValueError(msg)
            total += backtest_team_model(
                lambda: build_team_model(model_name, epsilon=epsilon),
                season=season,
                dbsession=dbsession,
                gameweeks=range(start_gw, end_gw + 1),
                horizon=horizon,
            )
    return EpsilonResult(epsilon=epsilon, score=total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune epsilon for the team model")
    parser.add_argument("--seasons", nargs="*", default=[CURRENT_SEASON])
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--model", default=DEFAULT_TEAM_MODEL)
    parser.add_argument("--epsilons", type=float, nargs="*", default=None)
    parser.add_argument("--epsilon-start", type=float, default=0.0)
    parser.add_argument("--epsilon-stop", type=float, default=0.1)
    parser.add_argument("--epsilon-num", type=int, default=11)
    parser.add_argument("--first-gw", type=int, default=None)
    parser.add_argument("--last-gw", type=int, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    grid = args.epsilons or list(
        np.linspace(args.epsilon_start, args.epsilon_stop, args.epsilon_num)
    )
    results = [
        evaluate_epsilon(
            epsilon=float(epsilon),
            seasons=args.seasons,
            horizon=args.horizon,
            model_name=args.model,
            first_gw=args.first_gw,
            last_gw=args.last_gw,
        )
        for epsilon in track(grid, desc="Epsilon")
    ]

    best = max(results, key=lambda r: r.score.mean_log_probability)
    for result in results:
        logger.info(
            "epsilon=%.4f  avg log prob=%.5f  fixtures=%s",
            result.epsilon,
            result.score.mean_log_probability,
            result.score.n_observations,
        )
    logger.info(
        "Best epsilon: %.4f (avg log prob %.5f)",
        best.epsilon,
        best.score.mean_log_probability,
    )

    out = args.out_csv or Path(
        f"tune_epsilon_results_{'_'.join(args.seasons)}_h{args.horizon}.csv"
    )
    rows = [r.as_row() for r in results]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
