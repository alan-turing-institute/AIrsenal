"""
Tune a player model's epsilon and goals-prior hyperparameters.

A grid over (epsilon, n_goals_prior), scored by
`prediction.evaluation.backtest_player_model`: for each gameweek, fit on the
matches before it and score who actually scored and assisted in the next
`--horizon` gameweeks.

`--model` chooses which of the two hyperparameter-taking models is swept. The
table in `player_models/__init__.py` maps a name to a model with its own
defaults, which is the wrong thing here: the point is to vary them.
"""

import argparse
import csv
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from airsenal.core.console import track
from airsenal.core.logging import configure_logging, get_logger
from airsenal.db.queries.gameweeks import get_max_gameweek
from airsenal.db.session import session_scope
from airsenal.game.season import CURRENT_SEASON
from airsenal.prediction.evaluation import ModelScore, backtest_player_model
from airsenal.prediction.player_models import (
    DEFAULT_PLAYER_MODEL,
    ConjugatePlayerConfig,
    ConjugatePlayerModel,
    XGPlayerConfig,
    XGPlayerModel,
)
from airsenal.prediction.protocols import PlayerModel

logger = get_logger(__name__)


def _conjugate(epsilon: float, n_goals_prior: int | None) -> PlayerModel:
    config = ConjugatePlayerConfig(epsilon=epsilon)
    if n_goals_prior is not None:
        config = replace(config, n_goals_prior=n_goals_prior)
    return ConjugatePlayerModel(config)


def _xg(epsilon: float, n_goals_prior: int | None) -> PlayerModel:
    config = XGPlayerConfig(epsilon=epsilon)
    if n_goals_prior is not None:
        config = replace(config, n_goals_prior=n_goals_prior)
    return XGPlayerModel(config)


# `n_goals_prior` of None means the model's own default, which for `xg` is one
# value per position - so sweeping epsilon alone does not have to flatten it to
# a single number first.
MODELS: dict[str, Callable[[float, int | None], PlayerModel]] = {
    "conjugate": _conjugate,
    "xg": _xg,
}


@dataclass(frozen=True)
class ParameterResult:
    """What one (epsilon, n_goals_prior) pair scored."""

    model_name: str
    epsilon: float
    n_goals_prior: int | None
    score: ModelScore

    def as_row(self) -> dict[str, str | float | int]:
        return {
            "model": self.model_name,
            "epsilon": self.epsilon,
            "n_goals_prior": (
                "model default" if self.n_goals_prior is None else self.n_goals_prior
            ),
            "total_log_prob": self.score.total_log_probability,
            "num_performances": self.score.n_observations,
            "avg_log_prob": self.score.mean_log_probability,
        }


def evaluate_params(
    epsilon: float,
    n_goals_prior: int | None,
    seasons: list[str],
    horizon: int,
    model_name: str = DEFAULT_PLAYER_MODEL,
    first_gameweek: int | None = None,
    last_gameweek: int | None = None,
) -> ParameterResult:
    """Score one parameter pair across every season, walking each one forward."""
    total = ModelScore()
    for season in track(seasons, desc="Season"):
        with session_scope() as dbsession:
            max_gameweek = get_max_gameweek(season=season, dbsession=dbsession)
            gameweek_start = first_gameweek or 1
            gameweek_end = (
                min(
                    last_gameweek if last_gameweek is not None else max_gameweek,
                    max_gameweek,
                )
                - horizon
            )
            if gameweek_end < gameweek_start:
                msg = (
                    f"Invalid gameweek window: start={gameweek_start}, "
                    f"end={gameweek_end}, "
                    f"max_gameweek={max_gameweek}, horizon={horizon}"
                )
                raise ValueError(msg)
            total += backtest_player_model(
                lambda: MODELS[model_name](epsilon, n_goals_prior),
                season=season,
                dbsession=dbsession,
                gameweeks=range(gameweek_start, gameweek_end + 1),
                horizon=horizon,
            )
    return ParameterResult(
        model_name=model_name,
        epsilon=epsilon,
        n_goals_prior=n_goals_prior,
        score=total,
    )


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Tune a player model")
    parser.add_argument("--seasons", nargs="+", default=[CURRENT_SEASON])
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--model", default=DEFAULT_PLAYER_MODEL, choices=sorted(MODELS))
    parser.add_argument("--epsilons", type=float, nargs="*", default=None)
    parser.add_argument("--epsilon-start", type=float, default=0.0)
    parser.add_argument("--epsilon-stop", type=float, default=1.0)
    parser.add_argument("--epsilon-num", type=int, default=11)
    parser.add_argument(
        "--n-goals-priors",
        type=int,
        nargs="*",
        default=None,
        help="Shrinkage values to sweep. Defaults to the model's own.",
    )
    parser.add_argument("--first-gameweek", type=int, default=None)
    parser.add_argument("--last-gameweek", type=int, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    epsilons = args.epsilons or list(
        np.linspace(args.epsilon_start, args.epsilon_stop, args.epsilon_num)
    )
    priors: list[int | None] = (
        [int(n) for n in args.n_goals_priors] if args.n_goals_priors else [None]
    )
    grid = [(float(e), n) for e in epsilons for n in priors]
    results = [
        evaluate_params(
            epsilon=epsilon,
            n_goals_prior=n_goals_prior,
            seasons=args.seasons,
            horizon=args.horizon,
            model_name=args.model,
            first_gameweek=args.first_gameweek,
            last_gameweek=args.last_gameweek,
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
        "Best for %s: epsilon=%.4f n_goals_prior=%s (avg log prob %.5f)",
        args.model,
        best.epsilon,
        best.n_goals_prior,
        best.score.mean_log_probability,
    )

    out = args.out_csv or Path(
        f"tune_player_results_{args.model}_{'_'.join(args.seasons)}_h{args.horizon}.csv"
    )
    rows = [r.as_row() for r in results]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
