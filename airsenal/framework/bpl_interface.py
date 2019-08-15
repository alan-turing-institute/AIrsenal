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
                [s.fixture.date, s.fixture.home_team, s.fixture.away_team, s.home_score, s.away_score]
                for s in session.query(Result).all()
            ]
        ),
        columns=["date", "home_team", "away_team", "home_goals", "away_goals"],
    )
    df_past["home_goals"] = df_past["home_goals"].astype(int)
    df_past["away_goals"] = df_past["away_goals"].astype(int)
    df_past["date"] = pd.to_datetime(df_past["date"])
    df_past = df_past[df_past["date"] > pd.to_datetime("2016-08-01")]
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
        columns=["team", "att", "mid", "defn", "ovr"]
    )
    return df


def get_player_history_df(position="all", season=CURRENT_SEASON, session=None):
    """
    Query the player_score table to get goals/assists/minutes, and then
    get the team_goals from the match table.
    The 'season' argument defined the set of players that will be considered, but
    for those players, all results will be used.
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
    players = list_players(position=position,season=season,dbsession=session)
    max_matches_per_player = get_max_matches_per_player(position, season, dbsession=session)
    for counter, player in enumerate(players):
        print(
            "Filling history dataframe for {}: {}/{} done".format(
                player.name, counter, len(players)
            )
        )
        results = player.scores
        row_count = 0
        for row in results:
            match_id = row.result_id
            if not match_id:
                print(" Couldn't find result for {} {} {}"\
                      .format(row.fixture.home_team,
                              row.fixture.away_team,
                              row.fixture.date))
                continue
            minutes = row.minutes
            opponent = row.opponent
            goals = row.goals
            assists = row.assists
            # find the match, in order to get team goals
            match_result = row.result
            match_date = dateparser.parse(row.fixture.date)
            if row.fixture.home_team == row.opponent:
                team_goals = match_result.away_score
            elif row.fixture.away_team == row.opponent:
                team_goals = match_result.home_score
            else:
                print("Unknown opponent!")
                team_goals = -1
            df.loc[len(df)] = [
                player.player_id,
                player.name,
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
                df.loc[len(df)] = [player.player_id, player.name, 0, 0, 0, 0, 0, 0]

    return df


def get_team_model(df, df_X, teams=CURRENT_TEAMS):
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
                strvals = df_X.loc[(df_X["team"]==team),["att","mid","defn","ovr"]].values
                intvals = [int(v) for v in strvals[0]]
                model_team.add_new_team(team, intvals)
                print("Adding new team {} with covariates".format(team))
            except:
                model_team.add_new_team(team)
                print("Adding new team {} without covariates".format(team))

    return model_team


def get_player_model():
    """
    load the player-level model, which will give the probability that
    a given player scored/assisted/did-neither when their team scores a goal.
    """
    stan_filepath = os.path.join(os.path.dirname(__file__), "../stan/player_forecasts.stan")
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
    print("Alpha is {}".format(alpha))
    return alpha


def process_player_data(prefix, season=CURRENT_SEASON, session=session):
    """
    transform the player dataframe, basically giving a list (for each player)
    of lists of minutes (for each match, and a list (for each player) of
    lists of ["goals","assists","neither"] (for each match)
    """
    df = get_player_history_df(prefix, season=season, session=session)
    df["neither"] = df["team_goals"] - df["goals"] - df["assists"]
    df.loc[(df["neither"]<0),["neither","team_goals","goals","assists"]]=[0.,0.,0.,0.]
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


def fit_data(prefix, model, season, session):
    """
    fit the data for a particular position (FWD, MID, DEF)
    """
    data, names = process_player_data(prefix, season, session)
    fit = model.optimizing(data)
    df = (
        pd.DataFrame(fit["theta"], columns=["pr_score", "pr_assist", "pr_neither"])
        .set_index(names)
        .reset_index()
    )
    df["pos"] = prefix
    return df, fit, data


def fit_all_data(model, season, session):
    df = pd.DataFrame()
    fits = []
    dfs = []
    reals = []
    for prefix in ["FWD", "MID", "DEF"]:
        d, f, r = fit_data(prefix, model, season, session)
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
