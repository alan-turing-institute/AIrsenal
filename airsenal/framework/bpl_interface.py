"""
Interface to the Stan models
"""
from bpl import ExtendedDixonColesMatchPredictor

import numpy as np
import pandas as pd

from airsenal.framework.schema import Result, FifaTeamRating, session
from airsenal.framework.utils import (
    get_fixtures_for_gameweek,
    is_future_gameweek,
)
from airsenal.framework.season import (
    CURRENT_SEASON,
    CURRENT_TEAMS,
    get_teams_for_season,
)

np.random.seed(42)


def get_result_dict(season, gameweek, dbsession):
    """
    query the match table and put results into pandas dataframe,
    to train the team-level model.
    """
    results = [
        s
        for s in dbsession.query(Result).all()
        if s.fixture
        and not is_future_gameweek(
            s.fixture.season,
            s.fixture.gameweek,
            current_season=season,
            next_gameweek=gameweek,
        )
    ]
    return {
        "home_team": np.array([r.fixture.home_team for r in results]),
        "away_team": np.array([r.fixture.away_team for r in results]),
        "home_goals": np.array([r.home_score for r in results]),
        "away_goals": np.array([r.away_score for r in results]),
    }


def get_ratings_dict(season, teams, dbsession):
    """Create a dataframe containing the fifa team ratings."""

    ratings = dbsession.query(FifaTeamRating).filter_by(season=season).all()
    if len(ratings) == 0:
        raise ValueError("No FIFA ratings found for season {}".format(season))

    ratings_dict = {
        s.team: np.array([s.att, s.mid, s.defn, s.ovr])
        for s in ratings
        if s.team in teams
    }
    if len(ratings_dict) != len(teams):
        raise ValueError(
            f"Must have FIFA ratings and results for all teams. {len(ratings_dict)} "
            + f"teams with FIFA ratings but {len(teams)} teams with results."
        )
    return ratings_dict


def get_training_data(season, gameweek, dbsession, ratings=True):
    """Get training data for team model, optionally including FIFA ratings
    as covariates if ratings is True. Data returned is for all matches up
    to specified gameweek and season.
    """
    training_data = get_result_dict(season, gameweek, dbsession)
    if ratings:
        teams = set(training_data["home_team"]) | set(training_data["away_team"])
        training_data["team_covariates"] = get_ratings_dict(season, teams, dbsession)
    return training_data


def create_and_fit_team_model(training_data, teams=CURRENT_TEAMS):
    """
    Get the team-level stan model, which can give probabilities of
    each potential scoreline in a given fixture.
    """
    model_team = ExtendedDixonColesMatchPredictor().fit(training_data)

    # TODO: Add teams (w/covariates) without match results.
    for t in teams:
        if t not in model_team.teams:
            print(f"No model for {t}")

    return model_team


def get_fitted_team_model(season, gameweek, dbsession):
    """
    get the fitted team model using the past results and the FIFA rankings
    """
    print("Fitting team model...")
    training_data = get_training_data(season, gameweek, dbsession)
    teams = get_teams_for_season(season, dbsession=dbsession)
    return create_and_fit_team_model(training_data, teams=teams)


def fixture_probabilities(
    gameweek, season=CURRENT_SEASON, model_team=None, dbsession=session
):
    """
    Returns probabilities for all fixtures in a given gameweek and season, as a data
    frame with a row for each fixture and columns being home_team,
    away_team, home_win_probability, draw_probability, away_win_probability.
    """
    if model_team is None:
        model_team = get_fitted_team_model(season, gameweek, dbsession)
    fixtures = get_fixtures_for_gameweek(gameweek, season=season, dbsession=dbsession)
    home_teams, away_teams = zip(*fixtures)
    probabilities = model_team.predict_outcome_proba(home_teams, away_teams)

    return pd.DataFrame(
        {
            "home_team": home_teams,
            "away_team": away_teams,
            "home_win_probability": probabilities["home_win"],
            "draw_probability": probabilities["draw"],
            "away_win_probability": probabilities["away_win"],
        }
    )
