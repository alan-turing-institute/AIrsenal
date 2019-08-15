"""
Use the BPL models to predict scores for upcoming fixtures.
"""

import os
import sys



from collections import defaultdict

from .mappings import (
    alternative_team_names,
    alternative_player_names,
    positions,
)

from scipy.stats import multinomial

from sqlalchemy import create_engine, and_, or_
from sqlalchemy.orm import sessionmaker

from .schema import Player, PlayerPrediction, Fixture, Base, engine

from .utils import (
    get_next_gameweek,
    get_fixtures_for_player,
    estimate_minutes_from_prev_season,
    get_recent_minutes_for_player,
    get_return_gameweek_for_player,
    get_player_name,
    list_players,
    CURRENT_SEASON
)
from .bpl_interface import (
    get_player_model,
    get_team_model,
    get_result_df,
    get_ratings_df,
    fit_all_data,
    list_players,
    fetcher
)

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


def calc_predicted_points(
        player, model_team, df_player, season, tag, session,
        gw_range=None, fixures_behind=3
):
    """
    Use the team-level model to get the probs of scoring or conceding
    N goals, and player-level model to get the chance of player scoring
    or assisting given that their team scores.
    """

    print("Getting points prediction for player {}".format(player.name))
    if not gw_range:
        # by default, go for next three matches
        next_gw = get_next_gameweek(season, session)
        gw_range = list(range(next_gw, min(next_gw+3,38))) # don't go beyond gw 38!
    team = player.team(season)
    position = player.position(season)
    fixtures = get_fixtures_for_player(player,
                                       season,
                                       gw_range=gw_range,
                                       dbsession=session)
    expected_points = defaultdict(float)  # default value is 0.0

    for fid in fixtures:
        fixture = session.query(Fixture)\
                         .filter_by(season=season)\
                         .filter_by(fixture_id=fid).first()
        gameweek = fixture.gameweek
        is_home = fixture.home_team == team
        opponent = fixture.away_team if is_home else fixture.home_team
        print("gameweek: {} vs {} home? {}".format(gameweek, opponent, is_home))
        recent_minutes = get_recent_minutes_for_player(
            player, num_match_to_use=fixures_behind, season=season, last_gw=gameweek-1,
            dbsession=session
        )
        if len(recent_minutes) == 0:
            # e.g. for gameweek 1 - try temporary hack
            recent_minutes = estimate_minutes_from_prev_season(
                player, season=season, dbsession=session
            )
        points = 0.
        expected_points[gameweek] = points
        # points for fixture will be zero if suspended or injured
        if is_injured_or_suspended(player.player_id, gameweek, season, session):
            points = 0.
        else:
        # now loop over recent minutes and average
            points = sum(
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
            ) / len(recent_minutes)
        # write the prediction for this fixture to the db
        fill_prediction(player, fixture, points, tag, session)
        expected_points[gameweek] += points
        # and return the per-gameweek predictions as a dict
        print("Expected points: {:.2f}".format(points))

    return expected_points


def fill_prediction(player, fixture, points, tag, session):
    """
    fill one row in the player_prediction table
    """
    pp = PlayerPrediction()
    pp.predicted_points = points
    pp.tag = tag
    pp.player = player
    pp.fixture = fixture
    session.add(pp)


def get_fitted_team_model(season, session):
    """
    get the fitted team model using the past results and the FIFA rankings
    """
    df_team = get_result_df(session)
    df_X = get_ratings_df(session)
    model_team = get_team_model(df_team, df_X)
    return model_team

def get_fitted_models(season, session):
    """
    Retrieve match and player models, and fit player model to the playerscore data.
    """
    model_team = get_fited_team_model(season, session)
    model_player = get_player_model()
    print("Generating player history dataframe - slow")
    df_player, fits, reals = fit_all_data(model_player, season, session)
    return model_team, df_player



def is_injured_or_suspended(player_id, gameweek, season, session):
    """
    Query the API for 'chance of playing next round', and if this
    is <=50%, see if we can find a return date.
    """
    if season != CURRENT_SEASON: # no API info for past seasons
        return False
    ## check if a player is injured or suspended
    pdata = fetcher.get_player_summary_data()[player_id]
    if (
            "chance_of_playing_next_round" in pdata.keys() \
            and pdata["chance_of_playing_next_round"] is not None
            and pdata["chance_of_playing_next_round"] <= 0.75
    ):
        ## check if we have a return date
        return_gameweek = get_return_gameweek_for_player(player_id, session)
        if return_gameweek is None or return_gameweek > gameweek:
            return True
    return False


def fill_ep(csv_filename):
    """
    fill the database with FPLs ep_next prediction, and also
    write output to a csv.
    """
    if not os.path.exists(csv_filename):
        outfile = open(csv_filename,"w")
        outfile.write("player_id,gameweek,EP\n")
    else:
        outfile = open(csv_filename,"a")

    summary_data = fetcher.get_player_summary_data()
    gameweek = get_next_gameweek()
    for k,v in summary_data.items():
        outfile.write("{},{},{}\n".format(k,gameweek,v['ep_next']))
        pp = PlayerPrediction()
        pp.player_id = k
        pp.gameweek = gameweek
        pp.predicted_points = v['ep_next']
        pp.method="EP"
        session.add(pp)
    session.commit()
    outfile.close()
