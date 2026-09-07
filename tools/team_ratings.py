"""
Print a team model's fitted attack and defence ratings.

For `--model xg`, which rates every team relative to a league average of one.
Sorted by what a team does to a match: creating more and conceding less is
better.
"""

import argparse

from airsenal.core.logging import configure_logging, get_logger
from airsenal.db.queries.gameweeks import get_max_gameweek
from airsenal.db.session import session_scope
from airsenal.game.season import CURRENT_SEASON
from airsenal.prediction.team_models import build_team_model
from airsenal.prediction.team_models.fitting import get_fitted_team_model

logger = get_logger(__name__)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default=CURRENT_SEASON)
    parser.add_argument("--gameweek", type=int, default=None)
    parser.add_argument("--model", default="xg")
    parser.add_argument("--epsilon", type=float, default=None)
    args = parser.parse_args()

    with session_scope() as dbsession:
        gameweek = args.gameweek or get_max_gameweek(
            season=args.season, dbsession=dbsession
        )
        fitted = get_fitted_team_model(
            gameweek,
            args.season,
            dbsession,
            model=build_team_model(args.model, args.epsilon),
        )

    # The ratings live on the model the table entry wrapped, if it wrapped one.
    # Read by name rather than by type: a model that has them need not be any
    # particular class, and one that has none is told so.
    model = getattr(fitted, "model", fitted)
    attack: dict[str, float] | None = getattr(model, "attack", None)
    defence: dict[str, float] | None = getattr(model, "defence", None)
    home_mean: float | None = getattr(model, "home_mean", None)
    away_mean: float | None = getattr(model, "away_mean", None)
    if not attack or not defence or home_mean is None or away_mean is None:
        logger.error(
            "The %s model has no attack and defence ratings to show", args.model
        )
        return

    print(
        f"{args.model} fitted as at {args.season} GW{gameweek}: "
        f"league {home_mean:.2f} xG at home, {away_mean:.2f} away"
    )
    print(
        f"{'team':6} {'attack':>7} {'defence':>8} {'net':>6} {'xG for':>8} {'xG vs':>7}"
    )
    for team in sorted(attack, key=lambda t: -attack[t] / defence[t]):
        print(
            f"{team:6} {attack[team]:7.3f} {defence[team]:8.3f} "
            f"{attack[team] / defence[team]:6.2f} "
            f"{home_mean * attack[team]:8.2f} "
            f"{away_mean * defence[team]:7.2f}"
        )


if __name__ == "__main__":
    main()
