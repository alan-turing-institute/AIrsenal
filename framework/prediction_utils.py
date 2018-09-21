"""
Use the BPL models to predict scores for upcoming fixtures.
"""

import os
import sys

sys.path.append("..")

from collections import defaultdict

from framework.mappings import (
    alternative_team_names,
    alternative_player_names,
    positions,
)

from scipy.stats import multinomial

from sqlalchemy import create_engine, and_, or_
from sqlalchemy.orm import sessionmaker

from .schema import Player, PlayerPrediction, Fixture, Base, engine

from .utils import (
    get_fixtures_for_player,
    get_recent_minutes_for_player,
    get_return_gameweek_for_player,
    get_player_name,
    list_players
)
from .bpl_interface import (
    get_player_model,
    get_team_model,
    get_result_df,
    get_ratings_df,
    fit_all_data
    list_players,
    fetcher,
)

DBSession = sessionmaker(bind=engine)
session = DBSession()

points_for_goal = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
points_for_cs = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
points_for_assist = 3


def get_appearance_points(minutes):
    """
    get 1 point for appearance, 2 for >60 mins
    """
    app_points = 0.
    if minutes > 0:
        app_points = 1
        if minutes >= 60:
            app_points += 1
    return app_points


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


def get_predicted_points(
    player_id, model_team, df_player, fixtures_ahead=1, fixures_behind=3
):
    """
    Use the team-level model to get the probs of scoring or conceding
    N goals, and player-level model to get the chance of player scoring
    or assisting given that their team scores.
    """

    print("Getting points prediction for player {}".format(get_player_name(player_id)))
    player = session.query(Player).filter_by(player_id=player_id).first()
    team = player.team
    position = player.position
    fixtures = get_fixtures_for_player(player_id)[0:fixtures_ahead]
    expected_points = defaultdict(float)  # default value is 0.0

    for fid in fixtures:
        fixture = session.query(Fixture).filter_by(fixture_id=fid).first()
        gameweek = fixture.gameweek
        is_home = fixture.home_team == team
        opponent = fixture.away_team if is_home else fixture.home_team
        print("gameweek: {} vs {} home? {}".format(gameweek, opponent, is_home))
        recent_minutes = get_recent_minutes_for_player(
            player_id, num_match_to_use=fixures_behind
        )
        points = 0.
        # points for fixture will be zero if suspended or injured
        if not is_injured_or_suspended(player_id, gameweek):
        # now loop over recent minutes and average
            points = sum(
                [
                    get_appearance_points(mins)
                    + get_attacking_points(
                        player_id,
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
            ) / len(recent_minutes)
            expected_points[gameweek] += points

        print("Expected points: {:.2f}".format(points))

    return expected_points


def is_injured_or_suspended(player_id, gameweek):
    """
    Query the API for 'chance of playing next round', and if this
    is <=50%, see if we can find a return date.
    """
    ## check if a player is injured or suspended
    pdata = fetcher.get_player_summary_data()[player_id]
    if (
            "chance_of_playing_next_round" in pdata.keys() \
            and pdata["chance_of_playing_next_round"] is not None
            and pdata["chance_of_playing_next_round"] <= 0.5
    ):
        ## check if we have a return date
        return_gameweek = get_return_gameweek_for_player(player_id)
        if return_gameweek is None or return_gameweek > gameweek:
            return True
    return False


def get_fitted_models():
    """
    Retrieve match and player models, and fit player model to the playerscore data.
    """
    df_team = get_result_df()
    df_X = get_ratings_df()
    model_team = get_team_model(df_team, df_X)
    model_player = get_player_model()
    print("Generating player history dataframe - slow")
    df_player, fits, reals = fit_all_data(model_player)
    return model_team, df_player


def calc_all_predicted_points(weeks_ahead):
    """
    Do the full prediction.
    """
    model_team, df_player = get_fitted_models()
    all_predictions = {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        for player in list_players(position=pos):
            all_predictions[player] = get_predicted_points(
                player, model_team, df_player, weeks_ahead
            )
    return all_predictions


def fill_table(prediction_dict, tag):
    """
    fill the table with the contents of prediction_dict
    """
    for player, player_prediction in prediction_dict.items():
        for gw in player_prediction.keys():
            pp = PlayerPrediction()
            pp.player_id = player
            pp.gameweek = gw
            pp.predicted_points = player_prediction[gw]
            pp.method = tag
            session.add(pp)
    session.commit()
