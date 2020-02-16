"""
Interface to the Stan models
"""
import os
import sys

import bpl
import numpy as np
import pandas as pd
import pystan


from .utils import *

np.random.seed(42)


def get_result_df(session):
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
                for s in session.query(Result).all()
            ]
        ),
        columns=["date", "home_team", "away_team", "home_goals", "away_goals"],
    )
    df_past["home_goals"] = df_past["home_goals"].astype(int)
    df_past["away_goals"] = df_past["away_goals"].astype(int)
    df_past["date"] = pd.to_datetime(df_past["date"])
    return df_past


def get_ratings_df(session):
    """Create a dataframe containing the fifa team ratings."""
    df = pd.DataFrame(
        np.array(
            [
                [s.team, s.att, s.mid, s.defn, s.ovr]
                for s in session.query(FifaTeamRating).all()
            ]
        ),
        columns=["team", "att", "mid", "defn", "ovr"],
    )
    return df


def create_and_fit_team_model(df, df_X, teams=CURRENT_TEAMS):
    """
    Get the team-level stan model, which can give probabilities of
    each potential scoreline in a given fixture.
    """
    model_team = bpl.BPLModel(df, X=df_X)

    model_team.fit()
    # check if each team is known to the model, and if not, add it using FIFA rankings
    for team in teams:
        if not team in model_team.team_indices.keys():
            try:
                strvals = df_X.loc[
                    (df_X["team"] == team), ["att", "mid", "defn", "ovr"]
                ].values
                intvals = [int(v) for v in strvals[0]]
                model_team.add_new_team(team, intvals)
                print("Adding new team {} with covariates".format(team))
            except:
                model_team.add_new_team(team)
                print("Adding new team {} without covariates".format(team))

    return model_team


def get_fitted_team_model(season, session):
    """
    get the fitted team model using the past results and the FIFA rankings
    """
    df_team = get_result_df(session)
    df_X = get_ratings_df(session)
    model_team = create_and_fit_team_model(df_team, df_X)
    return model_team
