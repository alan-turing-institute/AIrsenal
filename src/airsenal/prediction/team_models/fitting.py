"""Getting a team model fitted, and reading predictions off it."""

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.orm.session import Session

from airsenal.core.logging import get_logger
from airsenal.db.models import FifaTeamRating, Fixture, Result
from airsenal.db.queries.fixtures import (
    get_fixture_teams,
    get_fixtures_for_gameweeks,
)
from airsenal.db.queries.gameweeks import is_future_gameweek
from airsenal.db.queries.scores import get_expected_goals_by_fixture
from airsenal.db.queries.teams import get_teams_for_season
from airsenal.db.session import get_session
from airsenal.game.scoring import MAX_GOALS
from airsenal.game.season import CURRENT_SEASON
from airsenal.prediction.protocols import (
    ScorelineTeamModel,
    TeamFitData,
    TeamModel,
)
from airsenal.prediction.team_models import build_team_model

logger = get_logger(__name__)


def get_result_dict(gameweek: int, season: str, dbsession: Session) -> TeamFitData:
    """Every result before a gameweek, in the shape a team model is fitted to."""
    results = [
        s
        for s in dbsession.scalars(
            select(Result).options(selectinload(Result.fixture))
        ).all()
        if s.fixture.gameweek
        and not is_future_gameweek(
            s.fixture.gameweek,
            s.fixture.season,
            current_season=season,
            current_gameweek=gameweek,
        )
    ]
    # compute the time difference for each fixture in results to the first fixture of
    # the next gameweek
    result_dates = np.array(
        [
            pd.Timestamp(r.fixture.date).replace(tzinfo=None)
            for r in results
            if r.fixture.date is not None
        ]
    )
    end_date = np.array(
        [
            pd.Timestamp(f.date).replace(tzinfo=None)
            for f in get_fixtures_for_gameweeks([gameweek], season, dbsession)
            if f.date is not None
        ]
    ).min()
    time_diff = (end_date - result_dates) / pd.Timedelta(days=365)
    expected_goals = get_expected_goals_by_fixture(dbsession)

    def side_expected_goals(home: bool) -> np.ndarray:
        """Expected goals for one side of every match, `nan` where unrecorded."""
        return np.array(
            [
                expected_goals.get(
                    (
                        r.fixture.fixture_id,
                        r.fixture.home_team if home else r.fixture.away_team,
                    ),
                    np.nan,
                )
                for r in results
            ]
        )

    return {
        "home_team": np.array([r.fixture.home_team for r in results]),
        "away_team": np.array([r.fixture.away_team for r in results]),
        "home_goals": np.array([r.home_score for r in results]),
        "away_goals": np.array([r.away_score for r in results]),
        "home_expected_goals": side_expected_goals(home=True),
        "away_expected_goals": side_expected_goals(home=False),
        "time_diff": time_diff,
        "neutral_venue": np.zeros(len(results)),
        "game_weights": np.ones(len(results)),
    }


def get_ratings_dict(
    season: str, teams: list[str], dbsession: Session
) -> dict[str, np.ndarray]:
    """The FIFA ratings (attack, midfield, defence, overall) of each team."""
    ratings = dbsession.scalars(
        select(FifaTeamRating).where(FifaTeamRating.season == season)
    ).all()
    if len(ratings) == 0:
        msg = f"No FIFA ratings found for season {season}"
        raise ValueError(msg)

    ratings_dict = {
        s.team: np.array([s.att, s.mid, s.defn, s.ovr])
        for s in ratings
        if s.team in teams
    }
    if len(ratings_dict) != len(teams):
        msg = (
            f"Must have FIFA ratings and results for all teams. {len(ratings_dict)} "
            f"teams with FIFA ratings but {len(teams)} teams with results."
            " The teams involved are "
            f"{set(ratings_dict.keys()).symmetric_difference(teams)}"
        )
        raise ValueError(msg)
    return ratings_dict


def get_training_data(
    gameweek: int,
    season: str,
    dbsession: Session,
    ratings: bool = True,
) -> TeamFitData:
    """
    Training data for the team model: every match up to a gameweek and season.

    Args:
        ratings: If True, include the FIFA team ratings as covariates.
    """
    training_data = get_result_dict(gameweek, season, dbsession)
    if ratings:
        teams = list(set(training_data["home_team"]) | set(training_data["away_team"]))
        training_data["team_covariates"] = get_ratings_dict(
            season=season, teams=teams, dbsession=dbsession
        )
    return training_data


def add_new_teams_to_model[ModelT: TeamModel](
    team_model: ModelT,
    season: str,
    dbsession: Session,
    ratings: bool = True,
) -> ModelT:
    """
    Add teams with no previous results, such as promoted ones, to the model.

    Their FIFA ratings stand in as covariates for the results we do not have.
    """
    teams = get_teams_for_season(season=season, dbsession=dbsession)
    for t in teams:
        if team_model.teams is None or t not in team_model.teams:
            if ratings:
                logger.info("Adding %s to team model with covariates", t)
                covariates = get_ratings_dict(season, [t], dbsession)
                team_model.add_new_team(t, team_covariates=covariates[t])
            else:
                logger.info("Adding %s to team model without covariates", t)
                team_model.add_new_team(t)
    return team_model


def get_fitted_team_model(
    gameweek: int,
    season: str,
    dbsession: Session,
    ratings: bool = True,
    model: ScorelineTeamModel | None = None,
) -> ScorelineTeamModel:
    """
    Fit a team model to past results and FIFA ratings, and return it.

    Fits `model` in place if one is given; otherwise builds the default one.
    """
    if model is None:
        model = build_team_model()
    logger.info("Fitting team model...")
    training_data = get_training_data(
        season=season,
        gameweek=gameweek,
        dbsession=dbsession,
        ratings=ratings,
    )
    model.fit(training_data)
    return add_new_teams_to_model(
        team_model=model, season=season, dbsession=dbsession, ratings=ratings
    )


def fixture_probabilities(
    gameweek: int,
    season: str = CURRENT_SEASON,
    model: ScorelineTeamModel | None = None,
    dbsession: Session | None = None,
    ratings: bool = True,
) -> pd.DataFrame:
    """
    Win, draw and loss probabilities for every fixture in a gameweek.

    One row per fixture, with columns home_team, away_team,
    home_win_probability, draw_probability and away_win_probability. Without a
    model, the default one is built and fitted first.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if model is None:
        model = build_team_model()
    if model.teams is None:
        model = get_fitted_team_model(
            season=season,
            gameweek=gameweek,
            dbsession=dbsession,
            ratings=ratings,
            model=model,
        )

    fixtures = get_fixture_teams(
        get_fixtures_for_gameweeks(
            gameweeks=[gameweek], season=season, dbsession=dbsession
        )
    )
    home_teams, away_teams = zip(*fixtures, strict=True)
    probabilities = model.predict_outcome_proba(home_teams, away_teams)
    return pd.DataFrame(
        {
            "home_team": home_teams,
            "away_team": away_teams,
            "home_win_probability": probabilities["home_win"],
            "draw_probability": probabilities["draw"],
            "away_win_probability": probabilities["away_win"],
        }
    )


def get_goal_probabilities_for_fixtures(
    fixtures: list[Fixture],
    team_model: ScorelineTeamModel,
    max_goals: int = MAX_GOALS,
) -> dict[int, dict[str, dict[int, float]]]:
    """Probability of each team in a fixture scoring 0 to `max_goals` goals."""
    goals = np.arange(0, max_goals + 1)
    probs = {}
    for f in fixtures:
        home_team_goal_prob = team_model.predict_score_n_proba(
            goals, f.home_team, f.away_team, home=True
        )
        away_team_goal_prob = team_model.predict_score_n_proba(
            goals, f.away_team, f.home_team, home=False
        )
        probs[f.fixture_id] = {
            f.home_team: dict(zip(goals, home_team_goal_prob, strict=True)),
            f.away_team: dict(zip(goals, away_team_goal_prob, strict=True)),
        }
    return probs
