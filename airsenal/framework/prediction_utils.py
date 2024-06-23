"""
Use the BPL models to predict scores for upcoming fixtures.
"""

import os
from collections import defaultdict
from functools import partial
import time
from sqlalchemy.orm import joinedload
from sqlalchemy import and_
import pandas as pd


import numpy as np
import pandas as pd
from scipy.stats import multinomial

from airsenal.framework.FPL_scoring_rules import (
    get_appearance_points,
    points_for_assist,
    points_for_cs,
    points_for_goal,
    points_for_red_card,
    points_for_yellow_card,
    saves_for_point,
)
from airsenal.framework.player_model import (
    ConjugatePlayerModel,
    get_empirical_bayes_estimates,
)
from airsenal.framework.schema import (
    Absence,
    Fixture,
    Player,
    PlayerAttributes,
    PlayerPrediction,
    PlayerScore,
)
from airsenal.framework.utils import (
    CURRENT_SEASON,
    NEXT_GAMEWEEK,
    fastcopy,
    fetcher,
    get_fixtures_for_player,
    get_max_matches_per_player,
    get_player,
    get_player_from_api_id,
    get_recent_minutes_for_player,
    is_future_gameweek,
    list_players,
    session,
    was_historic_absence,
)

np.random.seed(42)
# consider probabilities of scoring/conceding up to this many goals
MAX_GOALS = 10


def check_absence(player_id, gameweek, season, dbsession=session):
    """
    Query the Absence table for a given player and season to see if the
    gameweek is within the period of absence. If so, return the details of absence.

    Returns: the reasons and details of absence (which may be empty if there was no absence)
    """
    absence_records = dbsession.query(Absence).join(Player).filter(
        Absence.season == season,
        Player.player_id == player_id,
        Absence.gw_from < gameweek,
        Absence.gw_until > gameweek
    ).all()

    reasons = [ab.reason for ab in absence_records] if len(absence_records) > 0 else None
    details = [ab.details for ab in absence_records] if len(absence_records) > 0 else None

    # If there's only one reason or detail, return it directly instead of a list
    if reasons is not None:
        reasons = reasons[0] if len(reasons) == 1 else reasons
    if details is not None:
        details = details[0] if len(details) == 1 else details

    return reasons, details


def get_player_history_df(
    position="all",
    all_players=False,
    fill_blank=True,
    season=CURRENT_SEASON,
    gameweek=NEXT_GAMEWEEK,
    dbsession=session,
):
    start = time.time()
    col_names = [
        "player_id",
        "player_name",
        "match_id",
        "date",
        "season",
        "gameweek",
        "goals",
        "assists",
        "minutes",
        "team_goals",
        "absence_reason",
        "absence_detail",
    ]
    player_data = []

    # Query players based on conditions
    if all_players:
        players = session.query(PlayerAttributes.player_id, PlayerAttributes.name).distinct().all()
    else:
        players = list_players(position=position, season=season, gameweek=gameweek, dbsession=dbsession)

    # Fetch maximum matches per player
    max_matches_per_player = get_max_matches_per_player(
        position, season=season, gameweek=gameweek, dbsession=dbsession
    )

    # Batch query all player scores for the given season and gameweek
    player_ids = [player.player_id for player in players]
    player_scores = (
        dbsession.query(PlayerScore)
        .join(Fixture)
        .filter(
            PlayerScore.player_id.in_(player_ids),
            Fixture.season == season,
            Fixture.gameweek <= gameweek
        )
        .options(joinedload(PlayerScore.fixture))  # Eager loading of related Fixture objects
        .all()
    )

    # Map player scores by player_id for faster access
    player_scores_by_id = {player_id: [] for player_id in player_ids}
    for score in player_scores:
        player_scores_by_id[score.player_id].append(score)

    # Iterate through players
    for counter, player in enumerate(players):
        # If players are queried directly
        if all_players:
            player_id, player_name = player
        # If players are obtained from list_players
        else:
            player_id = player.player_id
            player_name = player.name

        # Query player scores directly to avoid lazy loading
        player_scores = player_scores_by_id[player_id]

        # Processing player scores
        row_count = 0
        for score in player_scores:
            if is_future_gameweek(score.fixture.season, score.fixture.gameweek, current_season=season, next_gameweek=gameweek):
                continue

            match_id = score.result_id
            if not match_id:
                continue

            minutes = score.minutes
            goals = score.goals
            assists = score.assists

            # Get team goals from the match
            match_result = score.result
            if score.fixture.home_team == score.opponent:
                team_goals = match_result.away_score
            elif score.fixture.away_team == score.opponent:
                team_goals = match_result.home_score
            else:
                team_goals = -1

            absence_reason, absence_detail = check_absence(player_id, score.fixture.gameweek, score.fixture.season, session)

            player_data.append([
                player_id,
                player_name,
                match_id,
                score.fixture.date,
                score.fixture.season,
                score.fixture.gameweek,
                goals,
                assists,
                minutes,
                team_goals,
                absence_reason,
                absence_detail,
            ])
            row_count += 1

        # Fill blank rows if required
        if fill_blank and row_count < max_matches_per_player:
            player_data += [
                [player_id, player_name, 0, 0, 0, 0, 0, 0, 0, 0, None, None]
            ] * (max_matches_per_player - row_count)

    # Create DataFrame
    df = pd.DataFrame(player_data, columns=col_names)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.reset_index(drop=True, inplace=True)
    end = time.time()
    print(f"Time taken to filling history: {end - start:.2f} seconds")

    return df


def get_attacking_points(position, minutes, team_score_prob, player_prob):
    """
    use team-level and player-level models.
    """
    if position == "GK" or minutes == 0.0:
        # don't bother with GKs as they barely ever get points like this
        # if no minutes are played, can't score any points
        return 0.0

    # compute multinomial probabilities given time spent on pitch
    pr_score = (minutes / 90.0) * player_prob["prob_score"]
    pr_assist = (minutes / 90.0) * player_prob["prob_assist"]
    pr_neither = 1.0 - pr_score - pr_assist
    multinom_probs = (pr_score, pr_assist, pr_neither)

    def _get_partitions(n):
        # partition n goals into possible combinations of
        # [n_goals, n_assists, n_neither]
        partitions = []
        for i in range(n + 1):
            for j in range(n - i + 1):
                partitions.append([i, j, n - i - j])
        return partitions

    def _get_partition_score(partition):
        # calculate the points scored for a given partition
        return (
            points_for_goal[position] * partition[0] + points_for_assist * partition[1]
        )

    # compute the weighted sum of terms like:
    #   points(ng, na, nn) * p(ng, na, nn | Ng, T) * p(Ng)
    exp_points = 0.0
    for ngoals, score_n_prob in team_score_prob.items():
        if ngoals > 0:
            partitions = _get_partitions(ngoals)
            probabilities = multinomial.pmf(
                partitions, n=[ngoals] * len(partitions), p=multinom_probs
            )
            scores = map(_get_partition_score, partitions)
            exp_score_inner = sum(pi * si for pi, si in zip(probabilities, scores))
            exp_points += exp_score_inner * score_n_prob
    return exp_points


def get_defending_points(position, minutes, team_concede_prob):
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
        defending_points = points_for_cs[position] * team_concede_prob[0]
    if position in ["DEF", "GK"]:
        # lose 1 point per 2 goals conceded if player is on pitch for both
        # lets simplify, say that its only the last goal that matters, and
        # chance that player was on pitch for that is expected_minutes/90
        defending_points -= sum(
            (ngoals // 2) * (minutes / 90) * concede_n_prob
            for ngoals, concede_n_prob in team_concede_prob.items()
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
    if minutes >= 60 and player_id in df_bonus[0].index:
        return df_bonus[0].loc[player_id]
    elif (
        minutes >= 60
        or minutes >= 30
        and player_id not in df_bonus[1].index
        or minutes < 30
    ):
        return 0
    else:
        return df_bonus[1].loc[player_id]


def get_save_points(position, player_id, minutes, df_saves):
    """Returns average save points scored by player_id when playing minutes minutes (or
    zero if this player's position is not GK).

    df_saves - as calculated by fit_save_points()
    """
    if position != "GK":
        return 0
    if minutes >= 60 and player_id in df_saves.index:
        return df_saves.loc[player_id]
    else:
        return 0


def get_card_points(player_id, minutes, df_cards):
    """Returns average points lost by player_id due to yellow and red cards in matches
    they played at least 1 minute.

    df_cards - as calculated by fit_card_points().
    """
    if minutes >= 30 and player_id in df_cards.index:
        return df_cards.loc[player_id]
    else:
        return 0


def calc_predicted_points_for_player(
    player,
    fixture_goal_probs,
    df_player,
    df_bonus,
    df_saves,
    df_cards,
    season,
    gw_range=None,
    fixtures_behind=None,
    min_fixtures_behind=3,
    tag="",
    dbsession=session,
):
    """
    Use the team-level model to get the probs of scoring or conceding
    N goals, and player-level model to get the chance of player scoring
    or assisting given that their team scores.
    """
    if isinstance(player, int):
        player = get_player(player, dbsession=dbsession)

    message = f"Points prediction for player {player}"

    if not gw_range:
        # by default, go for next three matches
        gw_range = list(
            range(NEXT_GAMEWEEK, min(NEXT_GAMEWEEK + 3, 38))
        )  # don't go beyond gw 38!

    if fixtures_behind is None:
        # default to getting recent minutes from the same number of matches we're
        # predicting for
        fixtures_behind = len(gw_range)

    fixtures_behind = max(fixtures_behind, min_fixtures_behind)

    team = player.team(
        season, gw_range[0]
    )  # assume player stays with same team from first gameweek in range
    position = player.position(season)
    fixtures = get_fixtures_for_player(
        player, season, gw_range=gw_range, dbsession=dbsession
    )
    player_prob = (
        # fitted probability of scoring/assisting for this player
        # (we don't calculate this for goalkeepers)
        df_player[position].loc[player.player_id]
        if position != "GK"
        else None
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
        message += f"\ngameweek: {gameweek} vs {opponent}  {home_or_away}"
        team_score_prob = fixture_goal_probs[fixture.fixture_id][team]
        team_concede_prob = fixture_goal_probs[fixture.fixture_id][opponent]

        points = 0.0
        expected_points[gameweek] = points

        if sum(recent_minutes) == 0:
            # 'recent_minutes' contains the number of minutes that player played for
            # in the past few matches. If these are all zero, we will for sure predict
            # zero points for this player, so we don't need to call all the functions to
            # calculate appearance points, defending points, attacking points.
            points = 0.0

        elif player.is_injured_or_suspended(season, gw_range[0], gameweek):
            # Points for fixture will be zero if suspended or injured
            points = 0.0
        elif was_historic_absence(
            player,
            gameweek=gameweek,
            season=season,
            dbsession=dbsession,
        ):
            # Points will be zero if player was suspended or injured (in past season)
            points = 0.0
        else:
            # now loop over recent minutes and average
            points = 0
            for mins in recent_minutes:
                points += (
                    get_appearance_points(mins)
                    + get_attacking_points(
                        position,
                        mins,
                        team_score_prob,
                        player_prob,
                    )
                    + get_defending_points(position, mins, team_concede_prob)
                )
                if df_bonus is not None:
                    points += get_bonus_points(player.player_id, mins, df_bonus)
                if df_cards is not None:
                    points += get_card_points(player.player_id, mins, df_cards)
                if df_saves is not None:
                    points += get_save_points(
                        position, player.player_id, mins, df_saves
                    )

            points /= len(recent_minutes)

        # create the PlayerPrediction for this player+fixture
        if np.isnan(points):
            raise ValueError(f"nan points for {player} {fixture} {points} {tag}")
        predictions.append(make_prediction(player, fixture, points, tag))
        expected_points[gameweek] += points
        # and return the per-gameweek predictions as a dict
        message += f"\nExpected points: {points:.2f}"

    print(message)
    return predictions


def calc_predicted_points_for_pos(
    pos,
    fixture_goal_probs,
    df_bonus,
    df_saves,
    df_cards,
    season,
    gw_range,
    tag,
    model=ConjugatePlayerModel(),
    dbsession=session,
):
    """
    Calculate points predictions for all players in a given position and
    put into the DB
    """
    df_player = None
    if pos != "GK":  # don't calculate attacking points for keepers.
        df_player = fit_player_data(pos, season, min(gw_range), model, dbsession)
    return {
        player.player_id: calc_predicted_points_for_player(
            player=player,
            fixture_goal_probs=fixture_goal_probs,
            df_player=df_player,
            df_bonus=df_bonus,
            df_saves=df_saves,
            df_cards=df_cards,
            season=season,
            gw_range=gw_range,
            tag=tag,
            dbsession=dbsession,
        )
        for player in list_players(
            position=pos, season=season, gameweek=min(gw_range), dbsession=dbsession
        )
    }


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
        outfile.write(f"{player_id},{gameweek},{v['ep_next']}\n")
        pp = PlayerPrediction()
        pp.player_id = player_id
        pp.gameweek = gameweek
        pp.predicted_points = v["ep_next"]
        pp.method = "EP"
        dbsession.add(pp)
    dbsession.commit()
    outfile.close()


def process_player_data(
    prefix, season=CURRENT_SEASON, gameweek=NEXT_GAMEWEEK, dbsession=session
):
    """
    transform the player dataframe, basically giving a list (for each player)
    of lists of minutes (for each match, and a list (for each player) of
    lists of ["goals","assists","neither"] (for each match)
    """
    start = time.time()
    df = get_player_history_df(
        prefix, season=season, gameweek=gameweek, dbsession=dbsession
    )
    
    # Calculate 'neither' goals
    df["neither"] = df["team_goals"] - df["goals"] - df["assists"]
    df["neither"] = df["neither"].clip(lower=0)  # Ensure 'neither' is non-negative
    
    # Get unique player IDs and counts
    player_ids = np.sort(df["player_id"].unique())
    nplayer = len(player_ids)
    nmatch = df.groupby("player_id").count().iloc[0]["player_name"]
    
    # Calculate empirical Bayes estimates
    alpha = get_empirical_bayes_estimates(df)
    
    # Reshape goals, assists, and 'neither' columns
    y_values = df[["goals", "assists", "neither"]].values
    y = y_values.reshape((nplayer, nmatch, 3)).astype("int64")
    
    # Reshape minutes column
    minutes_values = df["minutes"].values
    minutes = minutes_values.reshape((nplayer, nmatch)).astype("int64")
    end = time.time()
    print(f"Time taken to process player data: {end - start:.2f} seconds")

    return {
        "player_ids": player_ids,
        "nplayer": nplayer,
        "nmatch": nmatch,
        "minutes": minutes,
        "y": y,
        "alpha": alpha,
    }



def fit_player_data(
    position, season, gameweek, model=ConjugatePlayerModel(), dbsession=session
):
    """
    fit the data for a particular position (FWD, MID, DEF)
    """
    start = time.time()
    data = process_player_data(position, season, gameweek, dbsession)
    print("Fitting player model for", position, "...")
    model = fastcopy(model)
    fitted_model = model.fit(data)
    df = pd.DataFrame(fitted_model.get_probs())

    df["pos"] = position
    df = (
        df.rename(columns={"index": "player_id"})
        .sort_values("player_id")
        .set_index("player_id")
    )
    end = time.time()
    print(f"Time taken to fit player model for {position}: {end - start:.2f}seconds")
    return df


def get_all_fitted_player_data(
    season, gameweek, model=ConjugatePlayerModel(), dbsession=session
):
    start = time.time()
    df_positions = {"GK": None}
    for pos in ["DEF", "MID", "FWD"]:
        df_positions[pos] = fit_player_data(pos, season, gameweek, model, dbsession)
        end = time.time()
        print(f"Time taken to fit all player models: {end - start:.2f}seconds")
    return df_positions


def get_player_scores(
    season, gameweek, min_minutes=0, max_minutes=90, dbsession=session
):
    """Utility function to get player scores rows up to (or the same as) season and
    gameweek as a dataframe"""

    query = (
        dbsession.query(PlayerScore, Fixture.season, Fixture.gameweek)
        .filter(PlayerScore.minutes >= min_minutes)
        .filter(PlayerScore.minutes <= max_minutes)
        .join(Fixture)
    )
    df = pd.read_sql(query.statement, dbsession.bind)

    is_fut = partial(is_future_gameweek, current_season=season, next_gameweek=gameweek)
    exclude = df.apply(lambda r: is_fut(r["season"], r["gameweek"]), axis=1)
    df = df[~exclude]
    return df


def mean_group_min_count(df, group_col, mean_col, min_count=10):
    """Calculate mean of column col in df, grouped by group_col, but normalising the
    sum by either the actual number of rows in the group or min_count, whichever is
    larger
    """
    counts = df.groupby(group_col)[mean_col].count()
    counts[counts < min_count] = min_count
    sums = df.groupby(group_col)[mean_col].sum()
    return sums / counts


def fit_bonus_points(
    gameweek=NEXT_GAMEWEEK, season=CURRENT_SEASON, min_matches=10, dbsession=session
):
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
        df = get_player_scores(
            season,
            gameweek,
            min_minutes=min_minutes,
            max_minutes=max_minutes,
            dbsession=dbsession,
        )
        return mean_group_min_count(df, "player_id", "bonus", min_count=min_matches)

    df_90 = get_bonus_df(60, 90)
    df_60 = get_bonus_df(30, 59)

    return (df_90, df_60)


def fit_save_points(
    gameweek=NEXT_GAMEWEEK,
    season=CURRENT_SEASON,
    min_matches=10,
    min_minutes=90,
    dbsession=session,
):
    """Calculate the average save points scored by each goalkeeper for matches they
    played at least min_minutes in.
    Mean is calculated as sum of all save points divided by either the number of
    matches the player has played in or min_matches, whichever is greater.

    Returns pandas series index by player ID, values average save points.
    """
    df = get_player_scores(
        season, gameweek, min_minutes=min_minutes, dbsession=dbsession
    )

    goalkeepers = list_players(
        position="GK", gameweek=gameweek, season=season, dbsession=dbsession
    )
    goalkeepers = [gk.player_id for gk in goalkeepers]
    df = df[df["player_id"].isin(goalkeepers)]

    #  1pt per 3 saves
    df["save_pts"] = (df["saves"] / saves_for_point).astype(int)

    return mean_group_min_count(df, "player_id", "save_pts", min_count=min_matches)


def fit_card_points(
    gameweek=NEXT_GAMEWEEK,
    season=CURRENT_SEASON,
    min_matches=10,
    min_minutes=1,
    dbsession=session,
):
    """Calculate the average points per match lost to yellow or red cards
    for each player.
    Mean is calculated as sum of all card points divided by either the number of
    matches the player has played in or min_matches, whichever is greater.

    Returns pandas series index by player ID, values average card points.
    """
    df = get_player_scores(
        season, gameweek, min_minutes=min_minutes, dbsession=dbsession
    )

    # TODO: different values for different minutes (remember minutes < 90 for red cards
    # though)
    df["card_pts"] = (
        points_for_yellow_card * df["yellow_cards"]
        + points_for_red_card * df["red_cards"]
    )

    return mean_group_min_count(df, "player_id", "card_pts", min_count=min_matches)
