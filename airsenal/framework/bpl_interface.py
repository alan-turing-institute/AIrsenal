"""
Interface to the Stan models
"""


import bpl
import numpy as np
import pandas as pd

from airsenal.framework.schema import Result, FifaTeamRating
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


def get_result_df(season, gameweek, dbsession):
    """
    query the match table and put results into pandas dataframe,
    to train the team-level model.
    """
    df_past = pd.DataFrame(
        np.array(
            [
                [
                    s.fixture.date,
                    s.fixture.home_team,
                    s.fixture.away_team,
                    s.home_score,
                    s.away_score,
                ]
                for s in dbsession.query(Result).all()
                if s.fixture
                and not is_future_gameweek(
                    s.fixture.season,
                    s.fixture.gameweek,
                    current_season=season,
                    next_gameweek=gameweek,
                )
            ]
        ),
        columns=["date", "home_team", "away_team", "home_goals", "away_goals"],
    )
    df_past["home_goals"] = df_past["home_goals"].astype(int)
    df_past["away_goals"] = df_past["away_goals"].astype(int)
    df_past["date"] = pd.to_datetime(df_past["date"])
    return df_past


def get_ratings_df(season, dbsession):
    """Create a dataframe containing the fifa team ratings."""

    ratings = dbsession.query(FifaTeamRating).filter_by(season=season).all()

    if len(ratings) == 0:
        raise ValueError("No FIFA ratings found for season {}".format(season))

    return pd.DataFrame(
        np.array([[s.team, s.att, s.mid, s.defn, s.ovr] for s in ratings]),
        columns=["team", "att", "mid", "defn", "ovr"],
    )


def create_and_fit_team_model(df, df_X, teams=CURRENT_TEAMS, n_attempts=3):
    """
    Get the team-level stan model, which can give probabilities of
    each potential scoreline in a given fixture.
    """
    model_team = bpl.BPLModel(df, X=df_X)

    for i in range(n_attempts):
        print(f"attempt {i + 1} of {n_attempts}...", end=" ", flush=True)
        try:
            model_team.fit()
            print("SUCCESS!")
            break
        except RuntimeError:
            print("FAILED.")
            if i + 1 == n_attempts:
                raise
            else:
                continue

    # check if each team is known to the model, and if not, add it using FIFA rankings
    for team in teams:
        if team not in model_team.team_indices.keys():
            try:
                strvals = df_X.loc[
                    (df_X["team"] == team), ["att", "mid", "defn", "ovr"]
                ].values
                intvals = [int(v) for v in strvals[0]]
                model_team.add_new_team(team, intvals)
                print("Adding new team {} with covariates".format(team))
            except Exception:
                model_team.add_new_team(team)
                print("Adding new team {} without covariates".format(team))

    return model_team


def get_fitted_team_model(season, gameweek, dbsession):
    """
    get the fitted team model using the past results and the FIFA rankings
    """
    print("Fitting team model...")
    df_team = get_result_df(season, gameweek, dbsession)
    df_X = get_ratings_df(season, dbsession=dbsession)
    teams = get_teams_for_season(season, dbsession=dbsession)
    return create_and_fit_team_model(df_team, df_X, teams=teams)


def fixture_probabilities(gameweek, season=CURRENT_SEASON, dbsession=None):
    """
    Returns probabilities for all fixtures in a given gameweek and season, as a data
    frame with a row for each fixture and columns being fixture_id, home_team,
    away_team, home_win_probability, draw_probability, away_win_probability.
    """
    model_team = get_fitted_team_model(season, gameweek, dbsession)
    fixture_probabilities_list = []
    fixture_id_list = []
    for i, fixture in enumerate(
        get_fixtures_for_gameweek(gameweek, season=season, dbsession=dbsession)
    ):
        probabilities = model_team.overall_probabilities(fixture[0], fixture[1])
        fixture_probabilities_list.append(
            [
                i,
                fixture[0],
                fixture[1],
                probabilities[0],
                probabilities[1],
                probabilities[2],
            ]
        )
        fixture_id_list.append(i)
    return pd.DataFrame(
        fixture_probabilities_list,
        columns=[
            "fixture_id",
            "home_team",
            "away_team",
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
        ],
        index=fixture_id_list,
    )
