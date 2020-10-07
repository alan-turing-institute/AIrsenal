"""
Use the BPL models to predict scores for upcoming fixtures.
"""

import os
import sys

from collections import defaultdict
import dateparser
import pandas as pd
import numpy as np
import pystan

from .mappings import alternative_team_names, alternative_player_names, positions

from scipy.stats import multinomial

from sqlalchemy import create_engine, and_, or_
from sqlalchemy.orm import sessionmaker

from .schema import Player, PlayerPrediction, Fixture, Base, engine

from .utils import (
    NEXT_GAMEWEEK,
    get_fixtures_for_player,
    estimate_minutes_from_prev_season,
    get_recent_minutes_for_player,
    get_return_gameweek_for_player,
    get_max_matches_per_player,
    get_player_name,
    get_player_from_api_id,
    list_players,
    fetcher,
    session,
    CURRENT_SEASON,
)
from .bpl_interface import get_fitted_team_model
from .FPL_scoring_rules import (
    points_for_goal,
    points_for_assist,
    points_for_cs,
    get_appearance_points,
)

np.random.seed(42)


def get_player_history_df(
    position="all", season=CURRENT_SEASON, session=session, gameweek=NEXT_GAMEWEEK
):
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
    players = list_players(
        position=position, season=season, dbsession=session, gameweek=gameweek
    )
    max_matches_per_player = get_max_matches_per_player(
        position, season, dbsession=session
    )
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
                print(
                    " Couldn't find result for {} {} {}".format(
                        row.fixture.home_team, row.fixture.away_team, row.fixture.date
                    )
                )
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


def get_attacking_points(
    player_id, position, team, opponent, is_home, minutes, model_team, df_player
):
    """
    use team-level and player-level models.
    """
    if position == "GK" or minutes == 0.0:
        # don't bother with GKs as they barely ever get points like this
        # if no minutes are played, can't score any points
        return 0.0

    # compute multinomial probabilities given time spent on pitch
    pr_score = (minutes / 90.0) * df_player.loc[player_id]["pr_score"]
    pr_assist = (minutes / 90.0) * df_player.loc[player_id]["pr_assist"]
    pr_neither = 1.0 - pr_score - pr_assist
    multinom_probs = (pr_score, pr_assist, pr_neither)

    def _get_partitions(n):
        # partition n goals into possible combinations of [n_goals, n_assists, n_neither]
        partitions = []
        for i in range(0, n + 1):
            for j in range(0, n - i + 1):
                partitions.append([i, j, n - i - j])
        return partitions

    def _get_partition_score(partition):
        # calculate the points scored for a given partition
        return (
            points_for_goal[position] * partition[0] + points_for_assist * partition[1]
        )

    # compute the weighted sum of terms like: points(ng, na, nn) * p(ng, na, nn | Ng, T) * p(Ng)
    exp_points = 0.0
    for ngoals in range(1, 11):
        partitions = _get_partitions(ngoals)
        probabilities = multinomial.pmf(
            partitions, n=[ngoals] * len(partitions), p=multinom_probs
        )
        scores = map(_get_partition_score, partitions)
        exp_score_inner = sum(pi * si for pi, si in zip(probabilities, scores))
        team_goal_prob = model_team.score_n_probability(ngoals, team, opponent, is_home)
        exp_points += exp_score_inner * team_goal_prob
    return exp_points


def get_defending_points(position, team, opponent, is_home, minutes, model_team):
    """
    only need the team-level model
    """
    if position == "FWD" or minutes == 0.0:
        # forwards don't get defending points
        # if no minutes are played, can't get any points
        return 0.0
    defending_points = 0
    if minutes >= 60:
        # TODO - what about if the team concedes only after player comes off?
        team_cs_prob = model_team.concede_n_probability(0, team, opponent, is_home)
        defending_points = points_for_cs[position] * team_cs_prob
    if position == "DEF" or position == "GK":
        # lose 1 point per 2 goals conceded if player is on pitch for both
        # lets simplify, say that its only the last goal that matters, and
        # chance that player was on pitch for that is expected_minutes/90
        for n in range(7):
            defending_points -= (
                (n // 2)
                * (minutes / 90)
                * model_team.concede_n_probability(n, team, opponent, is_home)
            )
    return defending_points


def calc_predicted_points(
    player,
    model_team,
    df_player,
    season,
    tag,
    session,
    gw_range=None,
    fixtures_behind=3,
):
    """
    Use the team-level model to get the probs of scoring or conceding
    N goals, and player-level model to get the chance of player scoring
    or assisting given that their team scores.
    """

    message = "Points prediction for player {}".format(player.name)

    if not gw_range:
        # by default, go for next three matches
        gw_range = list(
            range(NEXT_GAMEWEEK, min(NEXT_GAMEWEEK + 3, 38))
        )  # don't go beyond gw 38!
    team = player.team(
        season, gw_range[0]
    )  # assume player stays with same team from first gameweek in range
    position = player.position(season)
    fixtures = get_fixtures_for_player(
        player, season, gw_range=gw_range, dbsession=session
    )

    # use same recent_minutes from previous gameweeks for all predictions
    recent_minutes = get_recent_minutes_for_player(
        player,
        num_match_to_use=fixtures_behind,
        season=season,
        last_gw=min(gw_range) - 1,
        dbsession=session,
    )
    if len(recent_minutes) == 0:
        # e.g. for gameweek 1
        # this should now be dealt with in get_recent_minutes_for_player, so
        # throw error if not.
        # recent_minutes = estimate_minutes_from_prev_season(
        #    player, season=season, dbsession=session
        # )
        raise ValueError("Recent minutes is empty.")

    expected_points = defaultdict(float)  # default value is 0.
    predictions = []  # list that will hold PlayerPrediction objects

    for fixture in fixtures:
        gameweek = fixture.gameweek
        is_home = fixture.home_team == team
        opponent = fixture.away_team if is_home else fixture.home_team
        home_or_away = "at home" if is_home else "away"
        message += "\ngameweek: {} vs {}  {}".format(gameweek, opponent, home_or_away)
        points = 0.0
        expected_points[gameweek] = points

        if sum(recent_minutes) == 0:
            # 'recent_minutes' contains the number of minutes that player played
            # for in the past few matches. If these are all zero, we will for sure
            # predict zero points for this player, so we don't need to call all the
            # functions to calculate appearance points, defending points, attacking points.
            points = 0.0

        elif is_injured_or_suspended(player.fpl_api_id, gameweek, season, session):
            # Points for fixture will be zero if suspended or injured
            points = 0.0

        else:
            # now loop over recent minutes and average
            points = (
                sum(
                    [
                        get_appearance_points(mins)
                        + get_attacking_points(
                            player.player_id,
                            position,
                            team,
                            opponent,
                            is_home,
                            mins,
                            model_team,
                            df_player,
                        )
                        + get_defending_points(
                            position, team, opponent, is_home, mins, model_team
                        )
                        for mins in recent_minutes
                    ]
                )
                / len(recent_minutes)
            )
        # create the PlayerPrediction for this player+fixture
        predictions.append(make_prediction(player, fixture, points, tag))
        expected_points[gameweek] += points
        # and return the per-gameweek predictions as a dict
        message += "\nExpected points: {:.2f}".format(points)

    print(message)
    return predictions


def make_prediction(player, fixture, points, tag):
    """
    fill one row in the player_prediction table
    """
    pp = PlayerPrediction()
    pp.predicted_points = points
    pp.tag = tag
    pp.player = player
    pp.fixture = fixture
    return pp


#    session.add(pp)


def get_fitted_player_model(player_model, position, season, session):
    """
    Get the fitted player model for a given position
    """
    print("Generating player history dataframe - slow")
    df_player, fits, reals = fit_player_data(player_model, position, season, session)
    return df_player


# def get_fitted_models(season, session):
#    """
#    Retrieve match and player models, and fit player model to the playerscore data.
#    """
#    model_team = get_fited_team_model(season, session)
#    model_player = get_player_model()
#    print("Generating player history dataframe - slow")
#    df_player, fits, reals = fit_all_player_data(model_player, season, session)
#    return model_team, df_player


def is_injured_or_suspended(player_api_id, gameweek, season, session):
    """
    Query the API for 'chance of playing next round', and if this
    is <=50%, see if we can find a return date.
    """
    if season != CURRENT_SEASON:  # no API info for past seasons
        return False
    ## check if a player is injured or suspended
    pdata = fetcher.get_player_summary_data()[player_api_id]
    if (
        "chance_of_playing_next_round" in pdata.keys()
        and pdata["chance_of_playing_next_round"] is not None
        and pdata["chance_of_playing_next_round"] <= 50
    ):
        ## check if we have a return date
        return_gameweek = get_return_gameweek_for_player(player_api_id, session)
        if return_gameweek is None or return_gameweek > gameweek:
            return True
    return False


def fill_ep(csv_filename):
    """
    fill the database with FPLs ep_next prediction, and also
    write output to a csv.
    """
    if not os.path.exists(csv_filename):
        outfile = open(csv_filename, "w")
        outfile.write("player_id,gameweek,EP\n")
    else:
        outfile = open(csv_filename, "a")

    summary_data = fetcher.get_player_summary_data()
    gameweek = NEXT_GAMEWEEK
    for k, v in summary_data.items():
        player = get_player_from_api_id(k)
        player_id = player.player_id
        outfile.write("{},{},{}\n".format(player_id, gameweek, v["ep_next"]))
        pp = PlayerPrediction()
        pp.player_id = player_id
        pp.gameweek = gameweek
        pp.predicted_points = v["ep_next"]
        pp.method = "EP"
        session.add(pp)
    session.commit()
    outfile.close()


def get_player_model():
    """
    load the player-level model, which will give the probability that
    a given player scored/assisted/did-neither when their team scores a goal.
    """
    stan_filepath = os.path.join(
        os.path.dirname(__file__), "../stan/player_forecasts.stan"
    )
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
    df.loc[(df["neither"] < 0), ["neither", "team_goals", "goals", "assists"]] = [
        0.0,
        0.0,
        0.0,
        0.0,
    ]
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


def fit_player_data(model, prefix, season, session):
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
    df = (
        df.rename(columns={"index": "player_id"})
        .sort_values("player_id")
        .set_index("player_id")
    )
    return df, fit, data


def fit_all_player_data(model, season, session):
    df = pd.DataFrame()
    fits = []
    dfs = []
    reals = []
    for prefix in ["FWD", "MID", "DEF"]:
        d, f, r = fit_data(prefix, model, season, session)
        fits.append(f)
        dfs.append(d)
        reals.append(r)
    df = pd.concat(dfs)
    return df, fits, reals
