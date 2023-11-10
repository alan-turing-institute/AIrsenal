"""
test the score-calculating functions
"""
import numpy as np
import pandas as pd
import pytest
from bpl import ExtendedDixonColesMatchPredictor, NeutralDixonColesMatchPredictor

from airsenal.conftest import past_data_session_scope
from airsenal.framework.bpl_interface import (
    fixture_probabilities,
    get_fitted_team_model,
    get_ratings_dict,
    get_result_dict,
)
from airsenal.framework.FPL_scoring_rules import get_appearance_points
from airsenal.framework.player_model import (
    ConjugatePlayerModel,
    NumpyroPlayerModel,
    scale_goals_by_minutes,
)
from airsenal.framework.prediction_utils import (
    fit_bonus_points,
    fit_card_points,
    fit_player_data,
    fit_save_points,
    get_attacking_points,
    get_bonus_points,
    get_card_points,
    get_defending_points,
    get_player_history_df,
    get_player_scores,
    get_save_points,
    mean_group_min_count,
)
from airsenal.framework.schema import Fixture, Result


def generate_player_df(prob_score, prob_assist):
    """
    output a dataframe with custom player-level probabilities.
    """
    df = pd.DataFrame(columns=["pr_score", "pr_assist"])
    df.loc[0] = [prob_score, prob_assist]
    return df


def test_appearance_points():
    """
    points for just being on the pitch
    """
    assert get_appearance_points(0) == 0
    assert get_appearance_points(45) == 1
    assert get_appearance_points(60) == 2
    assert get_appearance_points(90) == 2


def test_defending_points_0_conceded():
    """
    for 0-0 draw, defenders and keepers should get clean sheet bonus
    if they were on the pitch for >= 60 mins.
    """
    # set chance of conceding n goals as {0: 1.0} .
    assert get_defending_points("FWD", 90, {0: 1.0}) == 0
    assert get_defending_points("MID", 90, {0: 1.0}) == 1
    assert get_defending_points("DEF", 90, {0: 1.0}) == 4
    assert get_defending_points("GK", 90, {0: 1.0}) == 4
    for pos in ["FWD", "MID", "DEF", "GK"]:
        assert get_defending_points(pos, 59, {0: 1.0}) == 0


def test_defending_points_2_conceded():
    """
    for 2 conceded, defenders and keepers should get lose 1 point
    """
    concede_probs = {0: 0.0, 1: 0.0, 2: 1.0}
    # set chance of conceding n goals as {2: 1.0} .
    assert get_defending_points("FWD", 90, concede_probs) == 0
    assert get_defending_points("MID", 90, concede_probs) == 0
    assert get_defending_points("DEF", 90, concede_probs) == -1
    assert get_defending_points("GK", 90, concede_probs) == -1
    for pos in ["DEF", "GK"]:
        assert get_defending_points(pos, 60, concede_probs) == -2 / 3


def test_defending_points_4_conceded():
    """
    for 4 conceded, defenders and keepers should get lose 2 points
    """
    # set chance of conceding n goals as {4: 1.0} .
    concede_probs = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 1.0}
    assert get_defending_points("FWD", 90, concede_probs) == 0
    assert get_defending_points("MID", 90, concede_probs) == 0
    assert get_defending_points("DEF", 90, concede_probs) == -2
    assert get_defending_points("GK", 90, concede_probs) == -2
    for pos in ["DEF", "GK"]:
        assert get_defending_points(pos, 60, concede_probs) == -4 / 3


def test_attacking_points_0_0():
    """
    For 0-0 no-one should get any attacking points.
    """

    team_score_prob = {0: 1.0}
    player_probs = {"prob_score": 1.0, "prob_assist": 0.0, "prob_neither": 0.0}
    assert get_attacking_points("FWD", 90, team_score_prob, player_probs) == 0
    assert get_attacking_points("MID", 90, team_score_prob, player_probs) == 0
    assert get_attacking_points("DEF", 90, team_score_prob, player_probs) == 0
    assert get_attacking_points("GK", 90, team_score_prob, player_probs) == 0


def test_attacking_points_1_0_top_scorer():
    """
    If team scores, and pr_score is 1, should get 4 points for FWD,
    5 for MID, 6 for DEF.  We don't consider possibility of GK scoring.
    """
    team_score_prob = {0: 0.0, 1: 1.0}
    player_probs = {"prob_score": 1.0, "prob_assist": 0.0, "prob_neither": 0.0}
    assert get_attacking_points("FWD", 90, team_score_prob, player_probs) == 4
    assert get_attacking_points("MID", 90, team_score_prob, player_probs) == 5
    assert get_attacking_points("DEF", 90, team_score_prob, player_probs) == 6
    assert get_attacking_points("GK", 90, team_score_prob, player_probs) == 0

    # play 45 mins - 50% chance that goal was scored while they were playing
    assert get_attacking_points("FWD", 45, team_score_prob, player_probs) == 2
    assert get_attacking_points("MID", 45, team_score_prob, player_probs) == 2.5
    assert get_attacking_points("DEF", 45, team_score_prob, player_probs) == 3
    assert get_attacking_points("GK", 45, team_score_prob, player_probs) == 0


def test_attacking_points_1_0_top_assister():
    """
    FWD, MID, DEF all get 3 points for an assist.
    """
    team_score_prob = {0: 0.0, 1: 1.0}
    player_probs = {"prob_score": 0.0, "prob_assist": 1.0, "prob_neither": 0.0}
    assert get_attacking_points("FWD", 90, team_score_prob, player_probs) == 3
    assert get_attacking_points("MID", 90, team_score_prob, player_probs) == 3
    assert get_attacking_points("DEF", 90, team_score_prob, player_probs) == 3
    assert get_attacking_points("GK", 90, team_score_prob, player_probs) == 0

    # play 45 mins - 50% chance that goal was scored while they were playing
    assert get_attacking_points("FWD", 45, team_score_prob, player_probs) == 1.5
    assert get_attacking_points("MID", 45, team_score_prob, player_probs) == 1.5
    assert get_attacking_points("DEF", 45, team_score_prob, player_probs) == 1.5
    assert get_attacking_points("GK", 45, team_score_prob, player_probs) == 0


def test_get_bonus_points():
    """Test correct bonus points returned for players from fitted (average) bonus"""
    df_90 = pd.Series({1: 1, 2: 2})
    df_60 = pd.Series({1: 0.5, 2: 0.25})
    df_bonus = (df_90, df_60)

    # 90 mins - use df_90 value
    assert get_bonus_points(1, 90, df_bonus) == 1
    assert get_bonus_points(2, 90, df_bonus) == 2
    # 45 mins - use df_60 value
    assert get_bonus_points(1, 45, df_bonus) == 0.5
    assert get_bonus_points(2, 45, df_bonus) == 0.25
    # <30 mins - zero
    assert get_bonus_points(1, 20, df_bonus) == 0
    assert get_bonus_points(1, 0, df_bonus) == 0
    # player not present in df_bonus (no bonus points history)
    assert get_bonus_points(3, 90, df_bonus) == 0


def test_get_save_points():
    """Test correct szve points returned for players from fitted
    (average) save points"""
    df_saves = pd.Series({1: 1, 2: 2})

    # >60 mins - return df value
    assert get_save_points("GK", 1, 90, df_saves) == 1
    assert get_save_points("GK", 2, 90, df_saves) == 2
    # <60 mins - zero
    assert get_save_points("GK", 1, 50, df_saves) == 0
    # player not present in df_saves (no history)
    assert get_save_points("GK", 3, 90, df_saves) == 0
    # not a goalkeeper - zero
    assert get_save_points("DEF", 1, 90, df_saves) == 0


def test_get_card_points():
    """Test correct card points returned for players from fitted
    (average) card points"""
    df_cards = pd.Series({1: -1, 2: -2})
    # >30 mins - return df value
    assert get_card_points(1, 90, df_cards) == -1
    assert get_card_points(2, 45, df_cards) == -2
    # 360 mins - zero
    assert get_card_points(1, 20, df_cards) == 0
    # player not present in df_saves (no history)
    assert get_card_points(3, 90, df_cards) == 0


def test_get_player_history_df():
    """
    test that we only consider gameweeks up to the specified gameweek
    (gw 12 in 1819 season).
    """
    with past_data_session_scope() as ts:
        df = get_player_history_df(season="1819", gameweek=12, dbsession=ts)
        assert len(df) > 0
        result_ids = df.match_id.unique()
        for result_id in result_ids:
            if result_id == 0:
                continue
            fixture_id = (
                ts.query(Result).filter_by(result_id=int(result_id)).first().fixture_id
            )
            fixture = ts.query(Fixture).filter_by(fixture_id=fixture_id).first()
            assert fixture.season in ["1718", "1819"]
            if fixture.season == "1819":
                assert fixture.gameweek < 12


def test_scale_goals_by_minutes():
    """Test scaling goal involvements by minutes played works as expected. Neither
    goals should be reduced by fraction of minutes played."""
    goals = np.zeros((2, 2, 3))
    goals[0, :, :] = np.array([[0, 0, 0], [1, 2, 3]])
    goals[1, :, :] = np.array([[0, 1, 2], [1, 0, 2]])
    minutes = np.array([[90, 90], [45, 45]])
    scaled_goals = scale_goals_by_minutes(goals, minutes)
    assert (scaled_goals == np.array([[1, 2, 3], [1, 1, 1]])).all()


def test_get_conjugate_prior():
    pm = ConjugatePlayerModel()
    goals = np.zeros((2, 2, 3))
    goals[0, :, :] = np.array([[0, 0, 0], [2, 2, 5]])
    goals[1, :, :] = np.array([[0, 1, 2], [1, 0, 2]])

    minutes = np.array([[90, 90], [90, 90]])
    scaled_goals = scale_goals_by_minutes(goals, minutes)
    assert (pm.get_prior(scaled_goals, n_goals_prior=15) == np.array([3, 3, 9])).all()

    minutes = np.array([[90, 90], [45, 45]])
    scaled_goals = scale_goals_by_minutes(goals, minutes)
    assert (pm.get_prior(scaled_goals, n_goals_prior=4) == np.array([1, 1, 2])).all()


def test_fit_conjugate_player_model():
    """Test results of fitting ConjugatePlayerModel"""
    pm = ConjugatePlayerModel()
    y = np.zeros((2, 2, 3))
    y[0, :, :] = np.array([[0, 0, 0], [1, 2, 3]])  # all y add to 4
    y[1, :, :] = np.array([[1, 2, 1], [2, 0, 0]])
    data = {
        "y": y,
        "player_ids": [0, 1],
        "minutes": 90 * np.ones((2, 2)),
    }

    pm = pm.fit(data, n_goals_prior=0)
    assert (pm.posterior == np.array([[1, 2, 3], [3, 2, 1]])).all()

    pm = pm.fit(data, n_goals_prior=3)
    assert (pm.posterior == np.array([[2, 3, 4], [4, 3, 2]])).all()


@pytest.mark.xfail(
    reason=(
        "NumpyroPlayerModel is broken after numpyro updates. "
        "See https://github.com/alan-turing-institute/AIrsenal/issues/611"
    )
)
def test_get_fitted_player_model_numpyro():
    pm = NumpyroPlayerModel()
    assert isinstance(pm, NumpyroPlayerModel)
    with past_data_session_scope() as ts:
        fpm = fit_player_data("FWD", "1819", 12, model=pm, dbsession=ts)
        assert isinstance(fpm, pd.DataFrame)
        assert len(fpm) > 0


def test_get_fitted_player_model_conjugate():
    cpm = ConjugatePlayerModel()
    assert isinstance(cpm, ConjugatePlayerModel)
    with past_data_session_scope() as ts:
        fcpm = fit_player_data("FWD", "1819", 12, model=cpm, dbsession=ts)
        assert isinstance(fcpm, pd.DataFrame)
        assert len(fcpm) > 0


def test_get_result_dict():
    with past_data_session_scope() as ts:
        d = get_result_dict("1819", 10, ts)
        assert isinstance(d, dict)
        assert len(d) > 0


def test_get_ratings_dict():
    with past_data_session_scope() as ts:
        rd = get_result_dict("1819", 10, ts)
        teams = set(rd["home_team"]) | set(rd["away_team"])
        d = get_ratings_dict("1819", teams, ts)
        assert isinstance(d, dict)
        assert len(d) >= 20


def test_get_fitted_team_model():
    # extended model
    with past_data_session_scope() as ts:
        extended = ExtendedDixonColesMatchPredictor()
        model_team = get_fitted_team_model("1819", 10, ts, model=extended)
        assert isinstance(model_team, ExtendedDixonColesMatchPredictor)
    # extended model with epsilon = 0.0 by default
    with past_data_session_scope() as ts:
        model_team = get_fitted_team_model("1819", 10, ts)
        assert isinstance(model_team, ExtendedDixonColesMatchPredictor)
        assert model_team.epsilon is None
    # extended model with epsilon = 0.5
    with past_data_session_scope() as ts:
        extended = ExtendedDixonColesMatchPredictor()
        model_team = get_fitted_team_model("1819", 10, ts, model=extended, epsilon=0.5)
        assert isinstance(model_team, ExtendedDixonColesMatchPredictor)
        assert model_team.epsilon == 0.5
    # neutral model with epsilon = 0.5
    with past_data_session_scope() as ts:
        neutral = NeutralDixonColesMatchPredictor()
        model_team = get_fitted_team_model("1819", 10, ts, model=neutral, epsilon=0.5)
        assert isinstance(model_team, NeutralDixonColesMatchPredictor)
        assert model_team.epsilon == 0.5
    # neutral model with no epsilon passed
    with past_data_session_scope() as ts:
        neutral = NeutralDixonColesMatchPredictor()
        model_team = get_fitted_team_model("1819", 10, ts, model=neutral)
        assert isinstance(model_team, NeutralDixonColesMatchPredictor)
        assert model_team.epsilon is None


def test_fixture_probabilities():
    with past_data_session_scope() as ts:
        df = fixture_probabilities(20, "1819", dbsession=ts)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10


def test_get_player_scores():
    """Test utility function used by fit bonus, save and card points to get player
    scores rows filtered by season, gameweek and minutese played values"""
    with past_data_session_scope() as ts:
        df = get_player_scores(season="1819", gameweek=12, dbsession=ts)
        # check type and columns
        assert len(df) > 0
        assert isinstance(df, pd.DataFrame)
        req_cols = [
            "player_id",
            "minutes",
            "saves",
            "bonus",
            "yellow_cards",
            "red_cards",
        ]
        for col in req_cols:
            assert col in df.columns
        # test player scores correctly filtered by gameweek and season
        for _, row in df.iterrows():
            assert row["season"] in ["1718", "1819"]
            if row["season"] == "1819":
                assert row["gameweek"] < 12
        # test filtering on min minutes
        df = get_player_scores(season="1819", gameweek=12, min_minutes=10, dbsession=ts)
        assert len(df) > 0
        assert all(df["minutes"] >= 10)
        # test filtering on max minutes
        df = get_player_scores(season="1819", gameweek=12, max_minutes=10, dbsession=ts)
        assert len(df) > 0
        assert all(df["minutes"] <= 10)


def test_mean_group_min_count():
    """Test mean for groups in df, normalising by a minimum valuee"""
    df = pd.DataFrame({"idx": [1, 1, 1, 1, 2, 2], "value": [1, 1, 1, 1, 2, 2]})

    mean_1 = mean_group_min_count(df, "idx", "value", min_count=1)
    assert mean_1.loc[1] == 1
    assert mean_1.loc[2] == 2

    mean_4 = mean_group_min_count(df, "idx", "value", min_count=4)
    assert mean_4.loc[1] == 1
    assert mean_4.loc[2] == 1


def test_fit_bonus():
    with past_data_session_scope() as ts:
        df_bonus = fit_bonus_points(gameweek=1, season="1819", dbsession=ts)
        assert len(df_bonus) == 2
        for df in df_bonus:
            assert isinstance(df, pd.Series)
            assert len(df) > 0
            assert all(df <= 3)
            assert all(df >= 0)


def test_fit_saves():
    with past_data_session_scope() as ts:
        df_saves = fit_save_points(gameweek=1, season="1819", dbsession=ts)
        assert isinstance(df_saves, pd.Series)
        assert len(df_saves) > 0
        assert all(df_saves >= 0)


def test_fit_cards():
    with past_data_session_scope() as ts:
        df_cards = fit_card_points(gameweek=1, season="1819", dbsession=ts)
        assert isinstance(df_cards, pd.Series)
        assert len(df_cards) > 0
        assert all(df_cards <= 0)
        assert all(df_cards >= -3)
