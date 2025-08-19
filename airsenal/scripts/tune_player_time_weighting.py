import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import multinomial
from sqlalchemy.orm.session import Session
from tqdm import tqdm

from airsenal.framework.prediction_utils import get_all_fitted_player_data
from airsenal.framework.schema import Fixture, PlayerScore, session
from airsenal.framework.utils import (
    get_fixtures_for_gameweek,
    get_max_gameweek,
)


@dataclass
class EpsilonResult:
    epsilon: float
    total_log_prob: float
    num_goals: int


def get_player_outcome_prob(
    player_goals: tuple[int, int, int],
    player_minutes: int,
    player_prob: tuple[float, float, float],
    team_goals: int,
) -> float:
    """
    Given an actual match result, a player's goal involvements, and the player's
    modelled goal involvement parameters, get the probability of the player achieving
    their actual goal involvements, according to the model.
    """
    if player_minutes == 0.0 or team_goals == 0:
        msg = "Function only valid if player played and team scored."
        raise ValueError(msg)
    if sum(player_goals) != team_goals:
        msg = "Player goals must sum to team goals."
        raise ValueError(msg)

    # compute multinomial probabilities given time spent on pitch
    pr_score = (player_minutes / 90.0) * player_prob[0]
    pr_assist = (player_minutes / 90.0) * player_prob[1]
    pr_neither = 1.0 - pr_score - pr_assist
    multinom_probs = (pr_score, pr_assist, pr_neither)

    return multinomial.pmf(player_goals, n=[team_goals], p=multinom_probs)[0]


def _eval_player_scores(player_scores, player_probs):
    logp = 0.0
    n = 0
    for ps in player_scores:
        if ps.fixture.home_team == ps.opponent:
            team_goals = ps.result.away_score
        elif ps.fixture.away_team == ps.opponent:
            team_goals = ps.result.home_score
        else:
            msg = f"opponent {ps.opponent} not in fixture {ps.fixture}"
            raise ValueError(msg)
        if team_goals == 0:
            continue
        if ps.player_id not in player_probs.index:
            continue

        player_goals = (ps.goals, ps.assists, team_goals - ps.goals - ps.assists)
        player_prob = (
            player_probs.loc[ps.player_id, "prob_score"],
            player_probs.loc[ps.player_id, "prob_assist"],
            player_probs.loc[ps.player_id, "prob_neither"],
        )
        outcome_prob = get_player_outcome_prob(
            player_goals=player_goals,
            player_minutes=ps.minutes,
            player_prob=player_prob,
            team_goals=team_goals,
        )
        logp += float(np.log(max(outcome_prob, 1e-12)))
        n += team_goals

    return logp, n


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
            player_probs = pd.concat(
                get_all_fitted_player_data(
                    season=season,
                    gameweek=gw,
                    dbsession=dbsession,
                    epsilon=epsilon,
                ).values()
            )
            # Evaluate on the next `horizon` gameweeks
            player_scores = (
                dbsession.query(PlayerScore)
                .join(Fixture)
                .filter(Fixture.season == season)
                .filter(Fixture.gameweek >= gw)
                .filter(Fixture.gameweek < gw + horizon)
                .filter(PlayerScore.minutes > 0)
            ).all()
            print(len(player_scores), "player scores found")
            logp, n = _eval_player_scores(player_scores, player_probs)
            total_logp += logp
            total_n += n
            print(f"\nGW {gw}: logp={logp:.3f}, n={n}")
            print("------")

    return EpsilonResult(
        epsilon=epsilon,
        total_log_prob=total_logp,
        num_goals=total_n,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune epsilon for the player model")
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
        default=2.0,
        help="End of epsilon range (inclusive) if --epsilons not given (default: 2.0)",
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
        else Path(
            f"tune_player_epsilon_results_{'_'.join(args.seasons)}_h{args.horizon}.csv"
        )
    )
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epsilon",
                "total_log_prob",
                "num_goals",
                "avg_log_prob",
            ]
        )

    results: list[EpsilonResult] = []

    for eps in tqdm(eps_grid, desc="Epsilon"):
        res = evaluate_epsilon(
            epsilon=eps,
            seasons=args.seasons,
            horizon=args.horizon,
            dbsession=session,
            first_gw=args.first_gw,
            last_gw=args.last_gw,
        )
        print("\n========================================\n")
        print(
            f"epsilon={res.epsilon:.5f}, "
            f"total_log_prob={res.total_log_prob:.3f}  "
            f"num_goals={res.num_goals}"
        )
        avg = res.total_log_prob / res.num_goals if res.num_goals else float("nan")
        with out_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    f"{res.epsilon:.6f}",
                    f"{res.total_log_prob:.6f}",
                    res.num_goals,
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
