"""
Print a team model's fitted attack and defence ratings.

For `--model xg`, which rates every team relative to a league average of one.
Sorted by what a team does to a match: creating more and conceding less is
better.

The ratings are fitted to every result before `--gameweek`, which defaults to
the next one without a result. That window reaches back through earlier
seasons, so relegated teams are rated too, and are listed unranked unless
`--in-league-only` leaves them out.
"""

import argparse

from airsenal.core.logging import configure_logging, get_logger
from airsenal.db.queries.gameweeks import (
    get_last_complete_gameweek_in_db,
    get_max_gameweek,
)
from airsenal.db.queries.teams import get_teams_for_season
from airsenal.db.session import session_scope
from airsenal.game.season import CURRENT_SEASON
from airsenal.prediction.team_models import DEFAULT_TEAM_MODEL, build_team_model
from airsenal.prediction.team_models.fitting import get_fitted_team_model

logger = get_logger(__name__)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default=CURRENT_SEASON)
    parser.add_argument("--gameweek", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument(
        "--in-league-only",
        action="store_true",
        help="leave out teams that are not in this season's league",
    )
    args = parser.parse_args()

    with session_scope() as dbsession:
        complete = get_last_complete_gameweek_in_db(
            season=args.season, dbsession=dbsession
        )
        gameweek = args.gameweek or (
            complete + 1
            if complete
            else get_max_gameweek(season=args.season, dbsession=dbsession)
        )
        in_league = set(get_teams_for_season(season=args.season, dbsession=dbsession))
        fitted = get_fitted_team_model(
            gameweek,
            args.season,
            dbsession,
            model=build_team_model(args.model or DEFAULT_TEAM_MODEL, args.epsilon),
        )

    # The ratings live on the model the table entry wrapped, if it wrapped one.
    model = getattr(fitted, "model", fitted)
    attack: dict[str, float] | None = getattr(model, "attack", None)
    defence: dict[str, float] | None = getattr(model, "defence", None)
    home_mean: float | None = getattr(model, "home_mean", None)
    away_mean: float | None = getattr(model, "away_mean", None)
    if not attack or not defence or home_mean is None or away_mean is None:
        logger.error(
            "The %s model has no attack and defence ratings to show",
            args.model or "default",
        )
        return

    teams = sorted(attack, key=lambda t: -attack[t] / defence[t])
    if args.in_league_only:
        teams = [team for team in teams if team in in_league]
    print(
        f"{args.model or 'xg'} fitted on every result before {args.season} "
        f"GW{gameweek}: league {home_mean:.2f} xG at home, {away_mean:.2f} away"
    )
    print(
        f"{'team':6} {'attack':>7} {'defence':>8} {'net':>6} {'xG for':>8} {'xG vs':>7}"
    )
    position = 0
    for team in teams:
        # only teams in this season's league take a place in the ranking
        in_it = team in in_league
        position += in_it
        rank = f"{position:2}." if in_it else "  -"
        print(
            f"{rank} {team:6} {attack[team]:7.3f} {defence[team]:8.3f} "
            f"{attack[team] / defence[team]:6.2f} "
            f"{home_mean * attack[team]:8.2f} "
            f"{away_mean * defence[team]:7.2f}"
            f"{'' if in_it else '   (not in the league this season)'}"
        )


if __name__ == "__main__":
    main()
