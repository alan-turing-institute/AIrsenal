"""
Use the BPL models to predict scores for upcoming fixtures.
"""

import os
import pickle
import pkg_resources
from collections import defaultdict
import pandas as pd
import numpy as np
import pystan

from scipy.stats import multinomial

from airsenal.framework.schema import Player, PlayerPrediction, PlayerScore, engine

from airsenal.framework.utils import (
    NEXT_GAMEWEEK,
    get_fixtures_for_player,
    get_recent_minutes_for_player,
    get_return_gameweek_for_player,
    get_max_matches_per_player,
    get_player_from_api_id,
    list_players,
    fetcher,
    session,
    CURRENT_SEASON,
    is_future_gameweek,
)

from airsenal.framework.FPL_scoring_rules import (
    points_for_goal,
    points_for_assist,
    points_for_cs,
    get_appearance_points,
    saves_for_point,
    points_for_yellow_card,
    points_for_red_card,
)

np.random.seed(42)


def get_player_history_df(
        position="all", season=CURRENT_SEASON, gameweek=NEXT_GAMEWEEK, dbsession=session
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
    player_data = []
    players = list_players(
        position=position, season=season, gameweek=gameweek, dbsession=dbsession
    )
    max_matches_per_player = get_max_matches_per_player(
        position, season=season, gameweek=gameweek, dbsession=dbsession
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
            if is_future_gameweek(
                row.fixture.season,
                row.fixture.gameweek,
                current_season=season,
                next_gameweek=gameweek,
            ):
                continue

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
            match_date = row.fixture.date
            if row.fixture.home_team == row.opponent:
                team_goals = match_result.away_score
            elif row.fixture.away_team == row.opponent:
                team_goals = match_result.home_score
            else:
                print("Unknown opponent!")
                team_goals = -1
            player_data.append(
                [
                    player.player_id,
                    player.name,
                    match_id,
                    match_date,
                    goals,
                    assists,
                    minutes,
                    team_goals,
                ]
            )
            row_count += 1

        ## fill blank rows so they are all the same size
        if row_count < max_matches_per_player:
            player_data += [[player.player_id, player.name, 0, 0, 0, 0, 0, 0]] * (
                max_matches_per_player - row_count
            )

    df = pd.DataFrame(player_data, columns=col_names)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.reset_index(drop=True, inplace=True)

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


def get_bonus_points(player_id, minutes, df_bonus):
    """
    Returns expected bonus points scored by player_id when playing minutes minutes.

    df_bonus : list containing df of average bonus pts scored when playing at least
    60 minutes in 1st index, and when playing between 30 and 60 minutes in 2nd index
    (as calculated by fit_bonus_points()).

    NOTE: Minutes values are currently hardcoded - this function and fit_bonus_points
    must be changed together.
    """
    if minutes >= 60:
        if player_id in df_bonus[0].index:
            return df_bonus[0].loc[player_id]
        else:
            return 0
    elif minutes >= 30:
        if player_id in df_bonus[1].index:
            return df_bonus[1].loc[player_id]
        else:
            return 0
    else:
        return 0


def get_save_points(position, player_id, minutes, df_saves):
    """Returns average save points scored by player_id when playing minutes minutes (or
    zero if this player's position is not GK).

    df_saves - as calculated by fit_save_points()
    """
    if position != "GK":
        return 0
    if minutes >= 60:
        if player_id in df_saves.index:
            return df_saves.loc[player_id]
        else:
            return 0
    else:
        return 0


def get_card_points(player_id, minutes, df_cards):
    """Returns average points lost by player_id due to yellow and red cards in matches
    they played at least 1 minute.

    df_cards - as calculated by fit_card_points().
    """
    if minutes >= 1:
        if player_id in df_cards.index:
            return df_cards.loc[player_id]
        else:
            return 0
    else:
        return 0


def calc_predicted_points_for_player(
    player,
    team_model,
    df_player,
    df_bonus,
    df_saves,
    df_cards,
    season,
    gw_range=None,
    fixtures_behind=3,
    tag="",
    dbsession=session
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
        player, season, gw_range=gw_range, dbsession=dbsession
    )

    # use same recent_minutes from previous gameweeks for all predictions
    recent_minutes = get_recent_minutes_for_player(
        player,
        num_match_to_use=fixtures_behind,
        season=season,
        last_gw=min(gw_range) - 1,
        dbsession=dbsession,
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

        elif is_injured_or_suspended(player.fpl_api_id, gameweek, season, dbsession):
            # Points for fixture will be zero if suspended or injured
            points = 0.0

        else:
            # now loop over recent minutes and average
            points = 0
            for mins in recent_minutes:
                points += (
                    get_appearance_points(mins)
                    + get_attacking_points(
                        player.player_id,
                        position,
                        team,
                        opponent,
                        is_home,
                        mins,
                        team_model,
                        df_player,
                    )
                    + get_defending_points(
                        position, team, opponent, is_home, mins, team_model
                    )
                )
                if df_bonus is not None:
                    points += get_bonus_points(player.player_id, mins, df_bonus)
                if df_cards is not None:
                    points += get_card_points(player.player_id, mins, df_cards)
                if df_saves is not None:
                    points += get_save_points(
                        position, player.player_id, mins, df_saves
                    )

            points = points / len(recent_minutes)

        # create the PlayerPrediction for this player+fixture
        predictions.append(make_prediction(player, fixture, points, tag))
        expected_points[gameweek] += points
        # and return the per-gameweek predictions as a dict
        message += "\nExpected points: {:.2f}".format(points)

    print(message)
    return predictions


def calc_predicted_points_for_pos(
    pos,
    team_model,
    player_model,
    df_bonus,
    df_saves,
    df_cards,
    season,
    gw_range,
    tag,
    dbsession=session
):
    """
    Calculate points predictions for all players in a given position and
    put into the DB
    """
    predictions = {}
    df_player = None
    if pos != "GK":  # don't calculate attacking points for keepers.
        df_player = get_fitted_player_model(
            player_model, pos, season, min(gw_range), dbsession
        )
    for player in list_players(
            position=pos, season=season, gameweek=min(gw_range), dbsession=dbsession
    ):
        predictions[player.player_id] = calc_predicted_points_for_player(
            player=player,
            team_model=team_model,
            df_player=df_player,
            df_bonus=df_bonus,
            df_saves=df_saves,
            df_cards=df_cards,
            season=season,
            gw_range=gw_range,
            tag=tag,
            dbsession=dbsession
        )

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


def get_fitted_player_model(player_model, position, season, gameweek, dbsession=session):
    """
    Get the fitted player model for a given position
    """
    print("Generating player history dataframe - slow")
    df_player, fits, reals = fit_player_data(
        player_model, position, season, gameweek, dbsession
    )
    return df_player


def is_injured_or_suspended(player_api_id, gameweek, season, dbsession=session):
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
        return_gameweek = get_return_gameweek_for_player(player_api_id, dbsession)
        if return_gameweek is None or return_gameweek > gameweek:
            return True
    return False


def fill_ep(csv_filename, dbsession=session):
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
        dbsession.add(pp)
    dbsession.commit()
    outfile.close()


def get_player_model():
    """
    load the player-level model, which will give the probability that
    a given player scored/assisted/did-neither when their team scores a goal.
    """
    ## old method - compile model at runtime
    stan_filepath = os.path.join(
        os.path.dirname(__file__), "../../stan/player_forecasts.stan"
    )
    if not os.path.exists(stan_filepath):
        raise RuntimeError("Can't find player_forecasts.stan")

    model_player = pystan.StanModel(file=stan_filepath)
    return model_player

    # new method - get pre-compiled pickle, BUT - how to ensure it looks
    # in site-packages rather than local directory?
#    model_file = pkg_resources.resource_filename(
#        "airsenal", "stan_model/player_forecasts.pkl"
#    )
#    with open(model_file, "rb") as f:
#        model_player = pickle.load(f)
#    return model_player


def get_empirical_bayes_estimates(df_emp):
    """
    Get starting values for the model based on averaging goals/assists/neither
    over all players in that position
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


def process_player_data(
        prefix, season=CURRENT_SEASON, gameweek=NEXT_GAMEWEEK, dbsession=session
):
    """
    transform the player dataframe, basically giving a list (for each player)
    of lists of minutes (for each match, and a list (for each player) of
    lists of ["goals","assists","neither"] (for each match)
    """
    df = get_player_history_df(
        prefix, season=season, gameweek=gameweek, dbsession=dbsession
    )
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


def fit_player_data(model, prefix, season, gameweek, dbsession=session):
    """
    fit the data for a particular position (FWD, MID, DEF)
    """
    data, names = process_player_data(prefix, season, gameweek, dbsession)
    print("Fitting player model for", prefix, "...")
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


def fit_all_player_data(model, season, gameweek, dbsession=session):
    df = pd.DataFrame()
    fits = []
    dfs = []
    reals = []
    for prefix in ["FWD", "MID", "DEF"]:
        d, f, r = fit_player_data(model, prefix, season, gameweek, dbsession)
        fits.append(f)
        dfs.append(d)
        reals.append(r)
    df = pd.concat(dfs)
    return df, fits, reals


def fit_bonus_points(gameweek=NEXT_GAMEWEEK, season=CURRENT_SEASON, min_matches=10):
    """Calculate the average bonus points scored by each player for matches they play
    between 60 and 90 minutes, and matches they play between 30 and 59 minutes.
    Mean is calculated as sum of all bonus points divided by either the number of
    maches the player has played in or min_matches, whichever is greater.

    Returns tuple of dataframes - first index bonus points for 60 to 90 mins, second
    index bonus points for 30 to 59 mins.

    NOTE: Minutes values are currently hardcoded - this function and fit_bonus_points
    must be changed together.
    """

    def get_bonus_df(min_minutes, max_minutes):
        query = (
            session.query(PlayerScore)
            .filter(PlayerScore.minutes <= max_minutes)
            .filter(PlayerScore.minutes >= min_minutes)
        )
        # TODO filter on gw and season
        df = pd.read_sql(query.statement, engine)

        match_counts = df.groupby("player_id").bonus.count()
        match_counts[match_counts < min_matches] = min_matches

        sum_bonus = df.groupby("player_id").bonus.sum()

        avg_bonus = sum_bonus / match_counts
        return avg_bonus

    df_90 = get_bonus_df(60, 90)
    df_60 = get_bonus_df(30, 59)

    return (df_90, df_60)


def fit_save_points(
    gameweek=NEXT_GAMEWEEK, season=CURRENT_SEASON, min_matches=10, min_minutes=90
):
    """Calculate the average save points scored by each goalkeeper for matches they
    played at least min_minutes in.
    Mean is calculated as sum of all save points divided by either the number of
    matches the player has played in or min_matches, whichever is greater.

    Returns pandas series index by player ID, values average save points.
    """
    goalkeepers = list_players(position="GK", gameweek=gameweek, season=season)
    goalkeepers = [gk.player_id for gk in goalkeepers]

    query = (
        session.query(PlayerScore)
        .join(Player)
        .filter(Player.player_id.in_(goalkeepers))
        .filter(PlayerScore.minutes >= min_minutes)
    )
    # TODO filter on gw and season
    df = pd.read_sql(query.statement, engine)

    #  1pt per 3 saves
    df["save_pts"] = (df["saves"] / saves_for_point).astype(int)

    match_counts = df.groupby("player_id").save_pts.count()
    match_counts[match_counts < min_matches] = min_matches

    sum_saves = df.groupby("player_id").save_pts.sum()

    avg_saves = sum_saves / match_counts
    return avg_saves


def fit_card_points(
    gameweek=NEXT_GAMEWEEK, season=CURRENT_SEASON, min_matches=10, min_minutes=1
):
    """Calculate the average points per match lost to yellow or red cards
    for each player.
    Mean is calculated as sum of all card points divided by either the number of
    matches the player has played in or min_matches, whichever is greater.

    Returns pandas series index by player ID, values average card points.
    """
    query = session.query(PlayerScore).filter(PlayerScore.minutes >= min_minutes)
    # TODO filter on gw and season
    # TODO: different values for different minutes (remember minutes < 90 for red cards though)
    df = pd.read_sql(query.statement, engine)

    df["card_pts"] = (
        points_for_yellow_card * df["yellow_cards"]
        + points_for_red_card * df["red_cards"]
    )
    match_counts = df.groupby("player_id").card_pts.count()
    match_counts[match_counts < min_matches] = min_matches

    sum_cards = df.groupby("player_id").card_pts.sum()
    avg_cards = sum_cards / match_counts

    return avg_cards
