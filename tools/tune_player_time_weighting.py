"""
Tune the conjugate player model's epsilon and goals-prior hyperparameters.

A grid over (epsilon, n_goals_prior), scored by
`prediction.evaluation.backtest_player_model`: for each gameweek, fit on the
matches before it and score who actually scored and assisted in the next
`--horizon` gameweeks. The pair with the highest mean log probability shared out
goals best among players it had not seen do it.

As with the team-model sweep, the scoring lives in the package and is typed
against `PlayerModel`, so this file is a grid and a CSV writer.
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
from airsenal.prediction.evaluation import ModelScore, backtest_player_model
from airsenal.prediction.player_models import (
    ConjugatePlayerConfig,
    ConjugatePlayerModel,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class ParameterResult:
    """What one (epsilon, n_goals_prior) pair scored."""

    epsilon: float
    n_goals_prior: int
    score: ModelScore

    def as_row(self) -> dict[str, float | int]:
        return {
            "epsilon": self.epsilon,
            "n_goals_prior": self.n_goals_prior,
            "total_log_prob": self.score.total_log_probability,
            "num_performances": self.score.n_observations,
            "avg_log_prob": self.score.mean_log_probability,
        }


def evaluate_params(
    epsilon: float,
    n_goals_prior: int,
    seasons: list[str],
    horizon: int,
    first_gw: int | None = None,
    last_gw: int | None = None,
) -> ParameterResult:
    """Score one parameter pair across every season, walking each one forward."""
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
            total += backtest_player_model(
                lambda: ConjugatePlayerModel(
                    ConjugatePlayerConfig(epsilon=epsilon, n_goals_prior=n_goals_prior)
                ),
                season=season,
                dbsession=dbsession,
                gameweeks=range(start_gw, end_gw + 1),
                horizon=horizon,
            )
    return ParameterResult(epsilon=epsilon, n_goals_prior=n_goals_prior, score=total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune the conjugate player model")
    parser.add_argument("--seasons", nargs="+", default=[CURRENT_SEASON])
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--epsilons", type=float, nargs="*", default=None)
    parser.add_argument("--epsilon-start", type=float, default=0.0)
    parser.add_argument("--epsilon-stop", type=float, default=1.0)
    parser.add_argument("--epsilon-num", type=int, default=11)
    parser.add_argument("--n-goals-priors", type=int, nargs="*", default=[35])
    parser.add_argument("--first-gw", type=int, default=None)
    parser.add_argument("--last-gw", type=int, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    epsilons = args.epsilons or list(
        np.linspace(args.epsilon_start, args.epsilon_stop, args.epsilon_num)
    )
    grid = [(float(e), int(n)) for e in epsilons for n in args.n_goals_priors]
    results = [
        evaluate_params(
            epsilon=epsilon,
            n_goals_prior=n_goals_prior,
            seasons=args.seasons,
            horizon=args.horizon,
            first_gw=args.first_gw,
            last_gw=args.last_gw,
        )
        for epsilon, n_goals_prior in track(grid, desc="Parameters")
    ]

    best = max(results, key=lambda r: r.score.mean_log_probability)
    for result in results:
        logger.info(
            "epsilon=%.4f n_goals_prior=%s  avg log prob=%.5f  performances=%s",
            result.epsilon,
            result.n_goals_prior,
            result.score.mean_log_probability,
            result.score.n_observations,
        )
    logger.info(
        "Best: epsilon=%.4f n_goals_prior=%s (avg log prob %.5f)",
        best.epsilon,
        best.n_goals_prior,
        best.score.mean_log_probability,
    )

    out = args.out_csv or Path(
        f"tune_player_results_{'_'.join(args.seasons)}_h{args.horizon}.csv"
    )
    rows = [r.as_row() for r in results]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
