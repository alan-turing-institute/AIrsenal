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


def get_result_df():
    """
    query the match table and put results into pandas dataframe,
    to train the team-level model.
    """
    df_past = pd.DataFrame(
        np.array(
            [
                [s.date, s.home_team, s.away_team, s.home_score, s.away_score]
                for s in session.query(Match).all()
            ]
        ),
        columns=["date", "home_team", "away_team", "home_goals", "away_goals"],
    )
    df_past["home_goals"] = df_past["home_goals"].astype(int)
    df_past["away_goals"] = df_past["away_goals"].astype(int)
    df_past["date"] = pd.to_datetime(df_past["date"])
    df_past = df_past[df_past["date"] > "2016-08-01"]
    return df_past


def get_ratings_df():
    """Create a dataframe containing the fifa team ratings."""
    df = pd.DataFrame(
        np.array(
            [
                [s.team, s.att, s.mid, s.defn, s.ovr]
                for s in session.query(FifaTeamRating).all()
            ]
        ),
        columns=["team", "att", "mid", "defn", "ovr"]
    )
    return df


def get_player_history_df(position="all"):
    """
    Query the player_score table to get goals/assists/minutes, and then
    get the team_goals from the match table.
    """

    col_names = [
        "player_id",
        "player_name",
        "match_id",
        "date",
        "goals",
        "assists",
        "minutes",
        "team_goals",
    ]
    df = pd.DataFrame(columns=col_names)
    player_ids = list_players(position)
    max_matches_per_player = get_max_matches_per_player(position)
    for counter, pid in enumerate(player_ids):
        player_name = get_player_name(pid)
        print(
            "Filling history dataframe for {}: {}/{} done".format(
                player_name, counter, len(player_ids)
            )
        )
        results = session.query(PlayerScore).filter_by(player_id=pid).all()
        row_count = 0
        for row in results:
            minutes = row.minutes
            opponent = row.opponent
            match_id = row.match_id
            goals = row.goals
            assists = row.assists
            # find the match, in order to get team goals
            match = session.query(Match).filter_by(match_id=row.match_id).first()
            match_date = dateparser.parse(match.date)
            if match.home_team == row.opponent:
                team_goals = match.away_score
            elif match.away_team == row.opponent:
                team_goals = match.home_score
            else:
                print("Unknown opponent!")
                team_goals = -1
            df.loc[len(df)] = [
                pid,
                player_name,
                match_id,
                match_date,
                goals,
                assists,
                minutes,
                team_goals,
            ]
            row_count += 1

        ## fill blank rows so they are all the same size
        if row_count < max_matches_per_player:
            for i in range(row_count, max_matches_per_player):
                df.loc[len(df)] = [pid, player_name, 0, 0, 0, 0, 0, 0]

    return df


def get_team_model(df, df_X):
    """
    Get the team-level stan model, which can give probabilities of
    each potential scoreline in a given fixture.
    """
    model_team = bpl.BPLModel(df, X=df_X)
    model_team.fit()
    return model_team


def get_player_model():
    """
    load the player-level model, which will give the probability that
    a given player scored/assisted/did-neither when their team scores a goal.
    """
    stan_filepath = "stan/player_forecasts.stan"
    if not os.path.exists(stan_filepath):
        stan_filepath = "../stan/player_forecasts.stan"
        if not os.path.exists(stan_filepath):
            raise RuntimeError("Can't find player_forecasts.stan")

    model_player = pystan.StanModel(file=stan_filepath)
    return model_player


def get_empirical_bayes_estimates(df_emp):
    """
    Get starting values for the model based on averaging goals/assists/neither
    over all players in that postition
    """
    # still not sure about this...
    df = df_emp.copy()
    df = df[df["match_id"] != 0]
    goals = df["goals"].sum()
    assists = df["assists"].sum()
    neither = df["neither"].sum()
    minutes = df["minutes"].sum()
    team = df["team_goals"].sum()
    total_minutes = 90 * len(df)
    neff = df.groupby("player_name").count()["goals"].mean()
    a0 = neff * (goals / team) * (total_minutes / minutes)
    a1 = neff * (assists / team) * (total_minutes / minutes)
    a2 = (
        neff
        * ((neither / team) - (total_minutes - minutes) / total_minutes)
        * (total_minutes / minutes)
    )
    alpha = np.array([a0, a1, a2])
    return alpha


def process_player_data(prefix):
    """
    transform the player dataframe, basically giving a list (for each player)
    of lists of minutes (for each match, and a list (for each player) of
    lists of ["goals","assists","neither"] (for each match)
    """
    df = get_player_history_df(prefix)
    df["neither"] = df["team_goals"] - df["goals"] - df["assists"]
    alpha = get_empirical_bayes_estimates(df)
    y = df.sort_values("player_id")[["goals", "assists", "neither"]].values.reshape(
        (
            df["player_id"].nunique(),
            df.groupby("player_id").count().iloc[0]["player_name"],
            3,
        )
    )

    minutes = df.sort_values("player_id")["minutes"].values.reshape(
        (
            df["player_id"].nunique(),
            df.groupby("player_id").count().iloc[0]["player_name"],
        )
    )

    nplayer = df["player_id"].nunique()
    nmatch = df.groupby("player_id").count().iloc[0]["player_name"]
    player_ids = np.sort(df["player_id"].unique())
    return (
        dict(
            nplayer=nplayer,
            nmatch=nmatch,
            minutes=minutes.astype("int64"),
            y=y.astype("int64"),
            alpha=alpha,
        ),
        player_ids,
    )


def fit_data(prefix, model):
    """
    fit the data for a particular position (FWD, MID, DEF)
    """
    data, names = process_player_data(prefix)
    fit = model.optimizing(data)
    df = (
        pd.DataFrame(fit["theta"], columns=["pr_score", "pr_assist", "pr_neither"])
        .set_index(names)
        .reset_index()
    )
    df["pos"] = prefix
    return df, fit, data


def fit_all_data(model):
    df = pd.DataFrame()
    fits = []
    dfs = []
    reals = []
    for prefix in ["FWD", "MID", "DEF"]:
        d, f, r = fit_data(prefix, model)
        fits.append(f)
        dfs.append(d)
        reals.append(r)
    df = (
        pd.concat(dfs)
        .rename(columns={"index": "player_id"})
        .sort_values("player_id")
        .set_index("player_id")
    )
    return df, fits, reals
