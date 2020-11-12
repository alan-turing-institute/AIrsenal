"""
test the score-calculating functions
"""

import pandas as pd
import pystan

import bpl

from airsenal.framework.FPL_scoring_rules import (
    get_appearance_points,
)
from airsenal.framework.prediction_utils import (
    get_defending_points,
    get_attacking_points,
    get_player_history_df,
    get_player_model,
    get_fitted_player_model,
    get_bonus_points,
    get_save_points,
    get_card_points,
    fit_bonus_points,
    fit_card_points,
    fit_save_points,
    get_player_scores,
    mean_group_min_count,
)
from airsenal.framework.bpl_interface import (
    get_result_df,
    get_ratings_df,
    get_fitted_team_model,
    fixture_probabilities,
)

from airsenal.conftest import test_past_data_session_scope
from airsenal.framework.schema import PlayerScore, Result, Fixture


class DummyTeamModel(object):
    """
    output an object that quotes custom
    probabilities for certain scorelines.
    """

    def __init__(self, score_prob_dict):
        self.score_prob_dict = score_prob_dict

    def score_n_probability(self, n, team, opp, is_home):
        total_prob = 0.0
        for score, prob in self.score_prob_dict.items():
            s = score[0] if is_home else score[1]
            if s == n:
                total_prob += prob
        return total_prob

    def concede_n_probability(self, n, team, opp, is_home):
        total_prob = 0.0
        for score, prob in self.score_prob_dict.items():
            s = score[1] if is_home else score[0]
            if s == n:
                total_prob += prob
        return total_prob


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


def test_defending_points_0_0():
    """
    for 0-0 draw, defenders and keepers should get clean sheet bonus
    if they were on the pitch for >= 60 mins.
    """
    tm = DummyTeamModel({(0, 0): 1.0})
    ## home or away doesn't matter
    assert get_defending_points("FWD", "dummy", "dummy", True, 90, tm) == 0
    assert get_defending_points("MID", "dummy", "dummy", True, 90, tm) == 1
    assert get_defending_points("DEF", "dummy", "dummy", True, 90, tm) == 4
    assert get_defending_points("GK", "dummy", "dummy", True, 90, tm) == 4
    for pos in ["FWD", "MID", "DEF", "GK"]:
        assert get_defending_points(pos, "dummy", "dummy", True, 59, tm) == 0


def test_defending_points_1_0():
    """
    test that out home/away logic works.
    """
    tm = DummyTeamModel({(1, 0): 1.0})
    ## home
    assert get_defending_points("FWD", "dummy", "dummy", True, 90, tm) == 0
    assert get_defending_points("MID", "dummy", "dummy", True, 90, tm) == 1
    assert get_defending_points("DEF", "dummy", "dummy", True, 90, tm) == 4
    assert get_defending_points("GK", "dummy", "dummy", True, 90, tm) == 4
    for pos in ["FWD", "MID", "DEF", "GK"]:
        assert get_defending_points(pos, "dummy", "dummy", True, 59, tm) == 0
    ## away
    assert get_defending_points("FWD", "dummy", "dummy", False, 90, tm) == 0
    assert get_defending_points("MID", "dummy", "dummy", False, 90, tm) == 0
    assert get_defending_points("DEF", "dummy", "dummy", False, 90, tm) == 0
    assert get_defending_points("GK", "dummy", "dummy", False, 90, tm) == 0
    for pos in ["FWD", "MID", "DEF", "GK"]:
        assert get_defending_points(pos, "dummy", "dummy", False, 59, tm) == 0


def test_defending_points_2_2():
    """
    defenders and keepers lose 1 point per 2 goals conceded.
    """
    tm = DummyTeamModel({(2, 2): 1.0})
    assert get_defending_points("FWD", "dummy", "dummy", True, 90, tm) == 0
    assert get_defending_points("MID", "dummy", "dummy", True, 90, tm) == 0
    assert get_defending_points("DEF", "dummy", "dummy", True, 90, tm) == -1
    assert get_defending_points("GK", "dummy", "dummy", True, 90, tm) == -1
    for pos in ["DEF", "GK"]:
        assert get_defending_points(pos, "dummy", "dummy", True, 60, tm) == (-2 / 3)


def test_defending_points_4_4():
    """
    defenders and keepers lose 1 point per 2 goals conceded.
    """
    tm = DummyTeamModel({(4, 4): 1.0})
    assert get_defending_points("FWD", "dummy", "dummy", True, 90, tm) == 0
    assert get_defending_points("MID", "dummy", "dummy", True, 90, tm) == 0
    assert get_defending_points("DEF", "dummy", "dummy", True, 90, tm) == -2
    assert get_defending_points("GK", "dummy", "dummy", True, 90, tm) == -2
    for pos in ["DEF", "GK"]:
        assert get_defending_points(pos, "dummy", "dummy", True, 60, tm) == (-4 / 3)


def test_defending_points_concede_0_to_4():
    """
    (Slightly) more realistic probalities - team concedes between 0 and 4
    goals with equal prob.
    """
    tm = DummyTeamModel(
        {(0, 0): 0.2, (0, 1): 0.2, (0, 2): 0.2, (0, 3): 0.2, (0, 4): 0.2}
    )
    assert get_defending_points("FWD", "dummy", "dummy", True, 90, tm) == 0
    assert get_defending_points("MID", "dummy", "dummy", True, 90, tm) == 0.2
    assert round(get_defending_points("DEF", "dummy", "dummy", True, 90, tm), 2) == 0.0
    assert round(get_defending_points("GK", "dummy", "dummy", True, 90, tm), 2) == 0.0
    ## TODO - how many points do we expect for < 90 mins?
    ## Current calculation only awards clean sheet points for team conceding 0,
    ## but we should allow for possibility that team concedes first goal after
    ## the player has been subbed.


def test_attacking_points_0_0():
    """
    For 0-0 no-one should get any attacking points.
    """
    tm = DummyTeamModel({(0, 0): 1.0})
    pm = generate_player_df(1.0, 0.0)
    assert get_attacking_points(0, "FWD", "dummy", "dummy", True, 90, tm, pm) == 0
    assert get_attacking_points(0, "MID", "dummy", "dummy", True, 90, tm, pm) == 0
    assert get_attacking_points(0, "DEF", "dummy", "dummy", True, 90, tm, pm) == 0
    assert get_attacking_points(0, "GK", "dummy", "dummy", True, 90, tm, pm) == 0


def test_attacking_points_1_0_top_scorer():
    """
    If team scores, and pr_score is 1, should get 4 points for FWD,
    5 for MID, 6 for DEF.  We don't consider possibility of GK scoring.
    """
    tm = DummyTeamModel({(1, 0): 1.0})
    pm = generate_player_df(1.0, 0.0)  # certain to score if team scores
    assert get_attacking_points(0, "FWD", "dummy", "dummy", True, 90, tm, pm) == 4
    assert get_attacking_points(0, "MID", "dummy", "dummy", True, 90, tm, pm) == 5
    assert get_attacking_points(0, "DEF", "dummy", "dummy", True, 90, tm, pm) == 6
    assert get_attacking_points(0, "GK", "dummy", "dummy", True, 90, tm, pm) == 0
    ## play 45 mins - 50% chance that goal was scored while they were playing
    assert get_attacking_points(0, "FWD", "dummy", "dummy", True, 45, tm, pm) == 2
    assert get_attacking_points(0, "MID", "dummy", "dummy", True, 45, tm, pm) == 2.5
    assert get_attacking_points(0, "DEF", "dummy", "dummy", True, 45, tm, pm) == 3


def test_attacking_points_1_0_top_assister():
    """
    FWD, MID, DEF all get 3 points for an assist.
    """
    tm = DummyTeamModel({(1, 0): 1.0})
    pm = generate_player_df(0.0, 1.0)  # certain to assist if team scores
    assert get_attacking_points(0, "FWD", "dummy", "dummy", True, 90, tm, pm) == 3
    assert get_attacking_points(0, "MID", "dummy", "dummy", True, 90, tm, pm) == 3
    assert get_attacking_points(0, "DEF", "dummy", "dummy", True, 90, tm, pm) == 3
    assert get_attacking_points(0, "GK", "dummy", "dummy", True, 90, tm, pm) == 0
    ## play 45 mins - 50% chance that goal was scored while they were playing
    assert get_attacking_points(0, "FWD", "dummy", "dummy", True, 45, tm, pm) == 1.5
    assert get_attacking_points(0, "MID", "dummy", "dummy", True, 45, tm, pm) == 1.5
    assert get_attacking_points(0, "DEF", "dummy", "dummy", True, 45, tm, pm) == 1.5


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
    """Test correct szve points returned for players from fitted (average) save points"""
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
    """Test correct card points returned for players from fitted (average) card points"""
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
    with test_past_data_session_scope() as ts:
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
            assert fixture.season == "1718" or fixture.season == "1819"
            if fixture.season == "1819":
                assert fixture.gameweek < 12


def test_get_fitted_player_model():
    pm = get_player_model()
    assert isinstance(pm, pystan.model.StanModel)
    with test_past_data_session_scope() as ts:
        fpm = get_fitted_player_model(pm, "FWD", "1819", 12, ts)
        assert isinstance(fpm, pd.DataFrame)
        assert len(fpm) > 0


def test_get_result_df():
    with test_past_data_session_scope() as ts:
        df = get_result_df("1819", 10, ts)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


def test_get_ratings_df():
    with test_past_data_session_scope() as ts:
        df = get_ratings_df("1819", ts)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 20


def test_get_fitted_team_model():
    with test_past_data_session_scope() as ts:
        model_team = get_fitted_team_model("1819", 10, ts)
        assert isinstance(model_team, bpl.models.BPLModel)


def test_fixture_probabilities():
    with test_past_data_session_scope() as ts:
        df = fixture_probabilities(20, "1819", ts)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10


def test_get_player_scores():
    """Test utility function used by fit bonus, save and card points to get player
    scores rows filtered by season, gameweek and minutese played values"""
    with test_past_data_session_scope() as ts:
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
            assert row["season"] == "1718" or row["season"] == "1819"
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
    with test_past_data_session_scope() as ts:
        df_bonus = fit_bonus_points(gameweek=1, season="1819", dbsession=ts)
        assert len(df_bonus) == 2
        for df in df_bonus:
            assert isinstance(df, pd.Series)
            assert len(df) > 0
            assert all(df <= 3)
            assert all(df >= 0)


def test_fit_saves():
    with test_past_data_session_scope() as ts:
        df_saves = fit_save_points(gameweek=1, season="1819", dbsession=ts)
        assert isinstance(df_saves, pd.Series)
        assert len(df_saves) > 0
        assert all(df_saves >= 0)


def test_fit_cards():
    with test_past_data_session_scope() as ts:
        df_cards = fit_card_points(gameweek=1, season="1819", dbsession=ts)
        assert isinstance(df_cards, pd.Series)
        assert len(df_cards) > 0
        assert all(df_cards <= 0)
        assert all(df_cards >= -3)
