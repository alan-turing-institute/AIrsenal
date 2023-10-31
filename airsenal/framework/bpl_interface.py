"""
Interface to the NumPyro team model in bpl-next:
https://github.com/anguswilliams91/bpl-next
"""
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from bpl import ExtendedDixonColesMatchPredictor, NeutralDixonColesMatchPredictor
from sqlalchemy.orm.session import Session

from airsenal.framework.schema import FifaTeamRating, Fixture, Result, session
from airsenal.framework.season import CURRENT_SEASON, get_teams_for_season
from airsenal.framework.utils import (
    get_fixture_teams,
    get_fixtures_for_gameweek,
    is_future_gameweek,
)

np.random.seed(42)


def get_result_dict(
    season: str, gameweek: int, dbsession: Session
) -> Dict[str, np.array]:
    """
    Query the match table and put results into pandas dataframe,
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
    # compute the time difference for each fixture in results
    # to the first fixture of the next gameweek
    result_dates = np.array(
        [pd.Timestamp(r.fixture.date).replace(tzinfo=None) for r in results]
    )
    end_date = pd.to_datetime(
        [f.date for f in get_fixtures_for_gameweek(gameweek, season, dbsession)]
    ).min()
    end_date = end_date.replace(tzinfo=None)
    time_diff = (end_date - result_dates) / pd.Timedelta(days=365)
    return {
        "home_team": np.array([r.fixture.home_team for r in results]),
        "away_team": np.array([r.fixture.away_team for r in results]),
        "home_goals": np.array([r.home_score for r in results]),
        "away_goals": np.array([r.away_score for r in results]),
        "time_diff": time_diff,
        "neutral_venue": np.zeros(len(results)),
        "time_diff": time_diff,
        "game_weights": np.ones(len(results)),
    }


def get_ratings_dict(
    season: str, teams: List[str], dbsession: Session
) -> Dict[str, np.array]:
    """
    Create a dataframe containing the fifa team ratings.
    """
    ratings = dbsession.query(FifaTeamRating).filter_by(season=season).all()
    if len(ratings) == 0:
        raise ValueError(f"No FIFA ratings found for season {season}")

    ratings_dict = {
        s.team: np.array([s.att, s.mid, s.defn, s.ovr])
        for s in ratings
        if s.team in teams
    }
    if len(ratings_dict) != len(teams):
        raise ValueError(
            f"Must have FIFA ratings and results for all teams. {len(ratings_dict)} "
            + f"teams with FIFA ratings but {len(teams)} teams with results."
            + " The teams involved are "
            + f"{set(ratings_dict.keys()).symmetric_difference(teams)}"
        )
    return ratings_dict


def get_training_data(
    season: str,
    gameweek: int,
    dbsession: Session,
    ratings: bool = True,
):
    """Get training data for team model, optionally including FIFA ratings
    as covariates if ratings is True. If time_decay is None, do not include
    exponential time decay in model.
    Data returned is for all matches up to specified gameweek and season.
    """
    training_data = get_result_dict(season, gameweek, dbsession)
    if ratings:
        teams = set(training_data["home_team"]) | set(training_data["away_team"])
        training_data["team_covariates"] = get_ratings_dict(
            season=season, teams=teams, dbsession=dbsession
        )
    return training_data


def create_and_fit_team_model(
    training_data: dict,
    model: Union[
        ExtendedDixonColesMatchPredictor, NeutralDixonColesMatchPredictor
    ] = ExtendedDixonColesMatchPredictor(),
    **fit_args,
) -> Union[ExtendedDixonColesMatchPredictor, NeutralDixonColesMatchPredictor]:
    """
    Get the team-level stan model, which can give probabilities of
    each potential scoreline in a given fixture.
    """
    if not fit_args:
        fit_args = {}
    if "epsilon" in fit_args:
        print(f"Fitting {type(model)} model with epsilon = {fit_args['epsilon']}")
    else:
        print(
            f"Fitting {type(model)} model but no epsilon passed, "
            "so using the default epsilon = 0"
        )

    return model.fit(training_data=training_data, **fit_args)


def add_new_teams_to_model(
    team_model: Union[
        ExtendedDixonColesMatchPredictor, NeutralDixonColesMatchPredictor
    ],
    season: str,
    dbsession: Session,
    ratings: bool = True,
) -> Union[ExtendedDixonColesMatchPredictor, NeutralDixonColesMatchPredictor]:
    """
    Add teams that we don't have previous results for (e.g. promoted teams) to the model
    using their FIFA ratings as covariates.
    """
    teams = get_teams_for_season(season=season, dbsession=dbsession)
    for t in teams:
        if t not in team_model.teams:
            if ratings:
                print(f"Adding {t} to team model with covariates")
                ratings = get_ratings_dict(season, [t], dbsession)
                team_model.add_new_team(t, team_covariates=ratings[t])
            else:
                print(f"Adding {t} to team model without covariates")
                team_model.add_new_team(t)
    return team_model


def get_fitted_team_model(
    season: str,
    gameweek: int,
    dbsession: Session,
    ratings: bool = True,
    model: Union[
        ExtendedDixonColesMatchPredictor, NeutralDixonColesMatchPredictor
    ] = ExtendedDixonColesMatchPredictor(),
    **fit_args,
) -> Union[ExtendedDixonColesMatchPredictor, NeutralDixonColesMatchPredictor]:
    """
    Get the fitted team model using the past results and the FIFA rankings.
    """
    print(f"Fitting team model ({type(model)})...")
    training_data = get_training_data(
        season=season,
        gameweek=gameweek,
        dbsession=dbsession,
        ratings=ratings,
    )
    team_model = create_and_fit_team_model(
        training_data=training_data, model=model, **fit_args
    )
    return add_new_teams_to_model(
        team_model=team_model, season=season, dbsession=dbsession, ratings=ratings
    )


def fixture_probabilities(
    gameweek: int,
    season: str = CURRENT_SEASON,
    model: Optional[
        Union[ExtendedDixonColesMatchPredictor, NeutralDixonColesMatchPredictor]
    ] = None,
    dbsession: Session = session,
    ratings: bool = True,
    **fit_args,
) -> pd.DataFrame:
    """
    Returns probabilities for all fixtures in a given gameweek and season, as a data
    frame with a row for each fixture and columns being home_team,
    away_team, home_win_probability, draw_probability, away_win_probability.

    If no model is passed, it will fit a ExtendedDixonColesMatchPredictor model
    by default.
    """

    # fit team model if none is passed or if it is not fitted yet
    # (model.teams will be None if so)
    if model is None:
        # fit extended model by default
        model = get_fitted_team_model(
            season=season,
            gameweek=gameweek,
            dbsession=dbsession,
            ratings=ratings,
            model=ExtendedDixonColesMatchPredictor(),
            **fit_args,
        )
    elif model.teams is None:
        # model is not fit yet, so will need to fit
        model = get_fitted_team_model(
            season=season,
            gameweek=gameweek,
            dbsession=dbsession,
            ratings=ratings,
            model=model,
            **fit_args,
        )

    # obtain fixtures
    fixtures = get_fixture_teams(
        get_fixtures_for_gameweek(gameweek=gameweek, season=season, dbsession=dbsession)
    )
    home_teams, away_teams = zip(*fixtures)
    # obtain match probabilities
    if isinstance(model, ExtendedDixonColesMatchPredictor):
        probabilities = model.predict_outcome_proba(home_teams, away_teams)
    elif isinstance(model, NeutralDixonColesMatchPredictor):
        probabilities = model.predict_outcome_proba(
            home_teams, away_teams, neutral_venue=np.zeros(len(home_teams))
        )
    else:
        raise NotImplementedError(
            "model must be either of type "
            "'ExtendedDixonColesMatchPredictor' or "
            "'NeutralDixonColesMatchPredictor'"
        )
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
    fixtures: List[Fixture],
    team_model: Union[
        ExtendedDixonColesMatchPredictor, NeutralDixonColesMatchPredictor
    ],
    max_goals: int = 10,
) -> Dict[int, Dict[str, Dict[int, float]]]:
    """
    Get the probability that each team in a fixture scores any number of goals up
    to max_goals.
    """
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
            f.home_team: {g: p for g, p in zip(goals, home_team_goal_prob)},
            f.away_team: {g: p for g, p in zip(goals, away_team_goal_prob)},
        }
    return probs
