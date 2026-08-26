"""The score-calculating functions, from single events up to a fitted model."""

import numpy as np
import pandas as pd
import pytest
from bpl import ExtendedDixonColesMatchPredictor, NeutralDixonColesMatchPredictor
from sqlalchemy import select

from airsenal.db.models import Fixture, Result
from airsenal.db.queries.scores import get_player_scores_df
from airsenal.game.scoring import get_appearance_points
from airsenal.prediction.features import (
    fit_bonus_points,
    fit_card_points,
    fit_save_points,
    get_player_history_df,
    mean_group_prior,
)
from airsenal.prediction.player_models import (
    ConjugatePlayerConfig,
    ConjugatePlayerModel,
    NumpyroPlayerModel,
)
from airsenal.prediction.player_models.fitting import fit_player_data
from airsenal.prediction.player_models.scaling import scale_goals_by_minutes
from airsenal.prediction.points import (
    get_attacking_points,
    get_bonus_points,
    get_card_points,
    get_defending_points,
    get_save_points,
)
from airsenal.prediction.team_models.dixon_coles import (
    DEFAULT_TEAM_EPSILON,
    DixonColesTeamModel,
)
from airsenal.prediction.team_models.fitting import (
    fixture_probabilities,
    get_fitted_team_model,
    get_ratings_dict,
    get_result_dict,
)
from tests.conftest import past_data_session_scope


def generate_player_df(prob_score, prob_assist):
    """A data frame with custom player-level probabilities."""
    df = pd.DataFrame(columns=["pr_score", "pr_assist"])
    df.loc[0] = [prob_score, prob_assist]
    return df


def test_appearance_points():
    """Points for appearances alone."""
    assert get_appearance_points(0) == 0
    assert get_appearance_points(45) == 1
    assert get_appearance_points(60) == 2
    assert get_appearance_points(90) == 2


def test_defending_points_0_conceded():
    """Defenders and keepers get the clean-sheet bonus for a 0-0, if they played 60."""
    # set chance of conceding n goals as {0: 1.0} .
    assert get_defending_points("FWD", 90, {0: 1.0}) == 0
    assert get_defending_points("MID", 90, {0: 1.0}) == 1
    assert get_defending_points("DEF", 90, {0: 1.0}) == 4
    assert get_defending_points("GK", 90, {0: 1.0}) == 4
    for pos in ["FWD", "MID", "DEF", "GK"]:
        assert get_defending_points(pos, 59, {0: 1.0}) == 0


def test_defending_points_2_conceded():
    """Defenders and keepers lose a point for two goals conceded."""
    concede_probs = {0: 0.0, 1: 0.0, 2: 1.0}
    # set chance of conceding n goals as {2: 1.0} .
    assert get_defending_points("FWD", 90, concede_probs) == 0
    assert get_defending_points("MID", 90, concede_probs) == 0
    assert get_defending_points("DEF", 90, concede_probs) == -1
    assert get_defending_points("GK", 90, concede_probs) == -1
    for pos in ["DEF", "GK"]:
        assert get_defending_points(pos, 60, concede_probs) == -2 / 3


def test_defending_points_4_conceded():
    """Defenders and keepers lose two points for four goals conceded."""
    # set chance of conceding n goals as {4: 1.0} .
    concede_probs = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 1.0}
    assert get_defending_points("FWD", 90, concede_probs) == 0
    assert get_defending_points("MID", 90, concede_probs) == 0
    assert get_defending_points("DEF", 90, concede_probs) == -2
    assert get_defending_points("GK", 90, concede_probs) == -2
    for pos in ["DEF", "GK"]:
        assert get_defending_points(pos, 60, concede_probs) == -4 / 3


def test_attacking_points_0_0():
    """A 0-0 gives nobody attacking points."""
    team_score_prob = {0: 1.0}
    player_probs = {"prob_score": 1.0, "prob_assist": 0.0, "prob_neither": 0.0}
    assert get_attacking_points("FWD", 90, team_score_prob, player_probs) == 0
    assert get_attacking_points("MID", 90, team_score_prob, player_probs) == 0
    assert get_attacking_points("DEF", 90, team_score_prob, player_probs) == 0
    assert get_attacking_points("GK", 90, team_score_prob, player_probs) == 0


def test_attacking_points_1_0_top_scorer():
    """
    A certain goalscorer gets their position's points for it.

    Four for a forward, five for a midfielder, six for a defender, ten for a
    goalkeeper.
    """
    team_score_prob = {0: 0.0, 1: 1.0}
    player_probs = {"prob_score": 1.0, "prob_assist": 0.0, "prob_neither": 0.0}
    assert get_attacking_points("FWD", 90, team_score_prob, player_probs) == 4
    assert get_attacking_points("MID", 90, team_score_prob, player_probs) == 5
    assert get_attacking_points("DEF", 90, team_score_prob, player_probs) == 6
    assert get_attacking_points("GK", 90, team_score_prob, player_probs) == 10

    # play 45 mins - 50% chance that goal was scored while they were playing
    assert get_attacking_points("FWD", 45, team_score_prob, player_probs) == 2
    assert get_attacking_points("MID", 45, team_score_prob, player_probs) == 2.5
    assert get_attacking_points("DEF", 45, team_score_prob, player_probs) == 3
    assert get_attacking_points("GK", 45, team_score_prob, player_probs) == 5


def test_attacking_points_1_0_top_assister():
    """Every position gets 3 points for an assist."""
    team_score_prob = {0: 0.0, 1: 1.0}
    player_probs = {"prob_score": 0.0, "prob_assist": 1.0, "prob_neither": 0.0}
    assert get_attacking_points("FWD", 90, team_score_prob, player_probs) == 3
    assert get_attacking_points("MID", 90, team_score_prob, player_probs) == 3
    assert get_attacking_points("DEF", 90, team_score_prob, player_probs) == 3
    assert get_attacking_points("GK", 90, team_score_prob, player_probs) == 3

    # play 45 mins - 50% chance that goal was scored while they were playing
    assert get_attacking_points("FWD", 45, team_score_prob, player_probs) == 1.5
    assert get_attacking_points("MID", 45, team_score_prob, player_probs) == 1.5
    assert get_attacking_points("DEF", 45, team_score_prob, player_probs) == 1.5
    assert get_attacking_points("GK", 45, team_score_prob, player_probs) == 1.5


def test_get_bonus_points():
    """Bonus points come back from the fitted average."""
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
    """Save points come back from the fitted average."""
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
    """Card points come back from the fitted average."""
    df_cards = pd.Series({1: -1, 2: -2})
    # >30 mins - return df value
    assert get_card_points(1, 90, df_cards) == -1
    assert get_card_points(2, 45, df_cards) == -2
    # 360 mins - zero
    assert get_card_points(1, 20, df_cards) == 0
    # player not present in df_saves (no history)
    assert get_card_points(3, 90, df_cards) == 0


def test_get_player_history_df():
    """Only gameweeks up to the one asked for are considered."""
    with past_data_session_scope() as ts:
        df = get_player_history_df(season="1819", gameweek=12, dbsession=ts)
        assert len(df) > 0
        result_ids = df.match_id.unique()
        for result_id in result_ids:
            if result_id == 0:
                continue
            result = ts.scalars(
                select(Result).where(Result.result_id == int(result_id)).limit(1)
            )
            result_row = result.first()
            assert result_row is not None
            fixture_id = result_row.fixture_id
            fixture = ts.scalars(
                select(Fixture).where(Fixture.fixture_id == fixture_id).limit(1)
            ).first()
            assert fixture is not None
            assert fixture.season in ["1718", "1819"]
            if fixture.season == "1819":
                assert fixture.gameweek is not None
                assert fixture.gameweek < 12


def test_scale_goals_by_minutes():
    """
    Scaling goal involvements by minutes played shrinks the "neither" count.

    It falls by the fraction of minutes the player was not on the pitch for.
    """
    goals = np.zeros((2, 2, 3))
    goals[0, :, :] = np.array([[0, 0, 0], [1, 2, 3]])
    goals[1, :, :] = np.array([[0, 1, 2], [1, 0, 2]])
    minutes = np.array([[90, 90], [45, 45]])
    scaled_goals = scale_goals_by_minutes(goals, minutes)
    assert (scaled_goals == np.array([[1, 2, 3], [1, 1, 1]])).all()


def test_get_conjugate_prior():
    pm = ConjugatePlayerModel(ConjugatePlayerConfig(n_goals_prior=0, epsilon=None))
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
    """Fitting ConjugatePlayerModel gives the expected posterior."""
    pm = ConjugatePlayerModel(ConjugatePlayerConfig(n_goals_prior=0, epsilon=None))
    y = np.zeros((2, 2, 3))
    y[0, :, :] = np.array([[0, 0, 0], [1, 2, 3]])  # all y add to 4
    y[1, :, :] = np.array([[1, 2, 1], [2, 0, 0]])
    data = {
        "y": y,
        "player_ids": [0, 1],
        "minutes": 90 * np.ones((2, 2)),
    }

    pm = pm.fit(data)
    assert (pm.posterior == np.array([[1, 2, 3], [3, 2, 1]])).all()

    # A different prior means a different model object, not a different fit call.
    pm = ConjugatePlayerModel(ConjugatePlayerConfig(n_goals_prior=3, epsilon=None))
    pm = pm.fit(data)
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


@pytest.mark.slow
def test_get_fitted_team_model():
    """
    Fit every team model against two full seasons.

    22 seconds, and almost all of it is jax. The shape and coverage assertions
    are worth having on every run, so they are duplicated against the small e2e
    database in tests/e2e/test_team_models.py; this stays as the "does it still
    work on real data" check for the nightly job.
    """
    # extended model
    with past_data_session_scope() as ts:
        model_team = get_fitted_team_model("1819", 10, ts, model=DixonColesTeamModel())
        assert isinstance(model_team.model, ExtendedDixonColesMatchPredictor)
    # extended model with default epsilon
    with past_data_session_scope() as ts:
        model_team = get_fitted_team_model("1819", 10, ts)
        assert isinstance(model_team.model, ExtendedDixonColesMatchPredictor)
        assert model_team.epsilon == DEFAULT_TEAM_EPSILON
    # extended model with epsilon = 0.5
    with past_data_session_scope() as ts:
        model_team = get_fitted_team_model(
            "1819", 10, ts, model=DixonColesTeamModel(epsilon=0.5)
        )
        assert isinstance(model_team.model, ExtendedDixonColesMatchPredictor)
        assert model_team.epsilon == 0.5
    # neutral model with epsilon = 0.5
    with past_data_session_scope() as ts:
        model_team = get_fitted_team_model(
            "1819", 10, ts, model=DixonColesTeamModel(neutral=True, epsilon=0.5)
        )
        assert isinstance(model_team.model, NeutralDixonColesMatchPredictor)
        assert model_team.epsilon == 0.5
    # neutral model with no epsilon passed
    with past_data_session_scope() as ts:
        model_team = get_fitted_team_model(
            "1819", 10, ts, model=DixonColesTeamModel(neutral=True)
        )
        assert isinstance(model_team.model, NeutralDixonColesMatchPredictor)
        assert model_team.epsilon == DEFAULT_TEAM_EPSILON


@pytest.mark.slow
def test_fixture_probabilities():
    with past_data_session_scope() as ts:
        df = fixture_probabilities(20, "1819", dbsession=ts)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10


def test_get_player_scores_df():
    """The row filter the bonus, save and card fits share: season, gameweek, minutes."""
    with past_data_session_scope() as ts:
        df = get_player_scores_df(season="1819", gameweek=12, dbsession=ts)
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
        df = get_player_scores_df(
            season="1819", gameweek=12, min_minutes=10, dbsession=ts
        )
        assert len(df) > 0
        assert all(df["minutes"] >= 10)
        # test filtering on max minutes
        df = get_player_scores_df(
            season="1819", gameweek=12, max_minutes=10, dbsession=ts
        )
        assert len(df) > 0
        assert all(df["minutes"] <= 10)


def test_mean_group_prior():
    """The empirical Bayes mean, with the prior weighted in."""
    df = pd.DataFrame(
        {
            "player_id": [1, 1, 1, 1, 2, 2],
            "bonus": [1, 1, 1, 1, 2, 2],
            "season": ["2526"] * 6,
            "gameweek": [1] * 6,
            "position": ["MID"] * 4 + ["FWD"] * 2,
        }
    )

    mean_1 = mean_group_prior(df, "player_id", "bonus", n_prior=0)
    assert mean_1.loc[1] == 1
    assert mean_1.loc[2] == 2

    n_prior = 6
    prior = 8 / 6
    mean_1_exp = (1 * 4 + n_prior * prior) / (4 + n_prior)
    mean_2_exp = (2 * 2 + n_prior * prior) / (2 + n_prior)
    mean_actual = mean_group_prior(df, "player_id", "bonus", n_prior=n_prior)
    assert mean_actual.loc[1] == mean_1_exp
    assert mean_actual.loc[2] == mean_2_exp

    mean_pos = mean_group_prior(
        df, "player_id", "bonus", n_prior=n_prior, prior_by_position=True
    )
    assert mean_pos.loc[1] == 1
    assert mean_pos.loc[2] == 2


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
