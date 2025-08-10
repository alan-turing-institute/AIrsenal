"""
Tune the time-weighting (epsilon) hyperparameter for the team model by
maximising the summed log-probability of actual scorelines on a rolling
"train up to GW t, evaluate on GW t+1..t+H" basis.

Steps per epsilon:
 1) For each gameweek in the season (up to max_gw - horizon), fit the team model
    using only results strictly before that gameweek (as if we were at GW t).
 2) For the next `horizon` gameweeks, compute the model probability of the
    actual (home_goals, away_goals) observed for each fixture.
 3) Accumulate the sum of log probabilities across all evaluated fixtures.
 4) Report the epsilon with the highest total log-probability.
"""

import argparse
import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from bpl import ExtendedDixonColesMatchPredictor
from sqlalchemy.orm.session import Session
from tqdm import tqdm

from airsenal.framework.bpl_interface import get_fitted_team_model
from airsenal.framework.schema import Fixture, session_scope
from airsenal.framework.utils import (
    get_fixtures_for_gameweek,
    get_max_gameweek,
)


@dataclass
class EpsilonResult:
    epsilon: float
    total_log_prob: float
    num_fixtures: int


def _safe_log(p: float) -> float:
    # guard against zero probabilities
    return float(np.log(max(p, 1e-12)))


def _score_fixtures(
    fixtures: Iterable[Fixture], model: ExtendedDixonColesMatchPredictor
) -> tuple[float, int]:
    """Sum log-probability of actual scorelines for a collection of fixtures.

    Returns (sum_log_prob, n_fixtures_scored)
    """
    sum_logp = 0.0
    n = 0
    for f in fixtures:
        if f.result is None:
            continue
        hg = int(f.result.home_score)
        ag = int(f.result.away_score)

        # predict_score_proba returns joint probability of exact scoreline
        proba = model.predict_score_proba(
            np.array([f.home_team]),
            np.array([f.away_team]),
            np.array([hg]),
            np.array([ag]),
        )
        # coerce to array and extract scalar
        arr = np.asarray(proba).ravel()
        prob = float(arr[0])
        sum_logp += _safe_log(prob)
        n += 1
    return sum_logp, n


def evaluate_epsilon(
    epsilon: float,
    seasons: list[str],
    horizon: int,
    dbsession: Session,
    first_gw: int | None = None,
    last_gw: int | None = None,
) -> EpsilonResult:
    """Evaluate a single epsilon value across the specified gameweek window."""

    total_logp = 0.0
    total_n = 0

    for season in tqdm(seasons, desc="Season"):
        max_gw = get_max_gameweek(season=season, dbsession=dbsession)
        start_gw = first_gw or 1
        # ensure we leave room for horizon weeks ahead
        end_gw = last_gw if last_gw is not None else max_gw - horizon
        end_gw = min(end_gw, max_gw - horizon)
        if end_gw < start_gw:
            msg = (
                "Invalid GW window: "
                f"start={start_gw}, end={end_gw}, max_gw={max_gw}, horizon={horizon}"
            )
            raise ValueError(msg)

        for gw in tqdm(range(start_gw, end_gw + 1), desc="GW"):
            print(f"\nFitting model for season {season} GW {gw}")
            # Fit the model using data strictly before (season, gw+horizon start)
            if not get_fixtures_for_gameweek(gw, season=season, dbsession=dbsession):
                print(f"No fixtures found for season {season} GW {gw}, skipping")
                continue
            fitted = get_fitted_team_model(
                season=season,
                gameweek=gw,
                dbsession=dbsession,
                model=ExtendedDixonColesMatchPredictor(),
                epsilon=epsilon,
            )

            # Evaluate on the next `horizon` gameweeks
            eval_gws = list(range(gw + 1, gw + 1 + horizon))
            fixtures = get_fixtures_for_gameweek(
                eval_gws, season=season, dbsession=dbsession
            )
            logp, n = _score_fixtures(fixtures, fitted)  # type: ignore[arg-type]
            total_logp += logp
            total_n += n
            print(f"\nGW {gw}: logp={logp:.3f}, n={n}")
            print("------")

    return EpsilonResult(
        epsilon=epsilon, total_log_prob=total_logp, num_fixtures=total_n
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune epsilon for the team model")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=str,
        help="Seasons to evaluate on, e.g. '2425'",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="Number of future gameweeks to score for each fit (default: 3)",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="*",
        help="Explicit list of epsilon values to try (overrides start/stop/num)",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=0.0,
        help=(
            "Start of epsilon range (inclusive) if --epsilons not given (default: 0.0)"
        ),
    )
    parser.add_argument(
        "--epsilon-stop",
        type=float,
        default=0.1,
        help="End of epsilon range (inclusive) if --epsilons not given (default: 0.1)",
    )
    parser.add_argument(
        "--epsilon-num",
        type=int,
        default=11,
        help=(
            "Number of grid points between start and stop if --epsilons "
            "not given (default: 11)"
        ),
    )
    parser.add_argument(
        "--first-gw",
        type=int,
        default=None,
        help="First gameweek to include for fitting (default: 1)",
    )
    parser.add_argument(
        "--last-gw",
        type=int,
        default=None,
        help=(
            "Last gameweek to use as the 'fit as of' point (default: max_gw - horizon)"
        ),
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help=(
            "Optional path to save a CSV with one row per epsilon containing "
            "epsilon,total_log_prob,num_fixtures,avg_log_prob. If not provided, "
            "a file named tune_epsilon_results_<season>_h<horizon>.csv will be created "
            "in the current directory."
        ),
    )

    args = parser.parse_args()

    eps_grid = (
        args.epsilons
        if args.epsilons and len(args.epsilons) > 0
        else list(np.linspace(args.epsilon_start, args.epsilon_stop, args.epsilon_num))
    )

    # Save results
    out_path = (
        Path(args.out_csv)
        if args.out_csv is not None
        else Path(f"tune_epsilon_results_{'_'.join(args.seasons)}_h{args.horizon}.csv")
    )
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epsilon", "total_log_prob", "num_fixtures", "avg_log_prob"])

    results: list[EpsilonResult] = []

    with session_scope() as db:
        db.expire_on_commit = False
        for eps in tqdm(eps_grid, desc="Epsilon"):
            res = evaluate_epsilon(
                epsilon=eps,
                seasons=args.seasons,
                horizon=args.horizon,
                dbsession=db,
                first_gw=args.first_gw,
                last_gw=args.last_gw,
            )
            print("\n========================================\n")
            print(
                "epsilon="
                f"{res.epsilon:.5f}  total_log_prob={res.total_log_prob:.3f}  "
                f"fixtures={res.num_fixtures}"
            )
            avg = (
                res.total_log_prob / res.num_fixtures
                if res.num_fixtures
                else float("nan")
            )
            with out_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        f"{res.epsilon:.6f}",
                        f"{res.total_log_prob:.6f}",
                        res.num_fixtures,
                        f"{avg:.6f}",
                    ]
                )

            print("\n========================================")
            results.append(res)

    if not results:
        msg = "No results computed - check DB contents and arguments."
        raise RuntimeError(msg)

    best = max(results, key=lambda r: r.total_log_prob)
    print()
    print("=" * 60)
    print(
        "Best epsilon = "
        f"{best.epsilon:.5f} with total log-probability "
        f"{best.total_log_prob:.3f}"
    )
    print("=" * 60)

    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
