"""
The typed involvement a player model returns, and the error form of scoring it.

`predict_involvement` replaced a `dict[str, np.ndarray]` whose keys and
invariants lived only in a docstring. These check the invariants are now the
type's, and that the shares can be scored by error as well as by log
probability - which is what a model that is not probabilistic needs.
"""

import numpy as np
import pandas as pd
import pytest

from airsenal.prediction.evaluation import (
    InvolvementScore,
    score_involvement_error,
)
from airsenal.prediction.player_models import ConstantPlayerModel
from airsenal.prediction.protocols import PlayerInvolvement


def involvement(prob_score=0.2, prob_assist=0.3, player_ids=(1,)):
    n = len(player_ids)
    return PlayerInvolvement(
        player_ids=np.array(player_ids),
        prob_score=np.full(n, prob_score),
        prob_assist=np.full(n, prob_assist),
        prob_neither=np.full(n, 1.0 - prob_score - prob_assist),
    )


def test_the_shares_have_to_sum_to_one():
    """The invariant the old docstring only claimed."""
    with pytest.raises(ValueError, match="sum to"):
        PlayerInvolvement(
            player_ids=np.array([7]),
            prob_score=np.array([0.5]),
            prob_assist=np.array([0.5]),
            prob_neither=np.array([0.5]),
        )


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="Mismatched lengths"):
        PlayerInvolvement(
            player_ids=np.array([1, 2]),
            prob_score=np.array([0.1]),
            prob_assist=np.array([0.1]),
            prob_neither=np.array([0.8]),
        )


def test_an_empty_involvement_is_allowed():
    """A position with no players is not an error, just nothing to say."""
    empty = PlayerInvolvement(
        player_ids=np.array([]),
        prob_score=np.array([]),
        prob_assist=np.array([]),
        prob_neither=np.array([]),
    )
    assert len(empty.as_dict()["player_id"]) == 0


def test_a_fitted_model_returns_the_typed_object():
    model = ConstantPlayerModel().fit(
        {
            "player_ids": np.array([1, 2, 3]),
            "nplayer": 3,
            "nmatch": 1,
            "minutes": np.full((3, 1), 90),
            "y": np.zeros((3, 1, 3), dtype=int),
            "alpha": np.array([1.0, 1.0, 1.0]),
            "time_diff": np.zeros((3, 1)),
        }
    )
    result = model.predict_involvement()
    assert isinstance(result, PlayerInvolvement)
    assert set(result.as_dict()) == {
        "player_id",
        "prob_score",
        "prob_assist",
        "prob_neither",
    }


def frame(prob_score, prob_assist, player_id=1):
    return pd.DataFrame(
        {"prob_score": [prob_score], "prob_assist": [prob_assist]},
        index=pd.Index([player_id], name="player_id"),
    )


class Performance:
    """The parts of a PlayerScore the involvement scorers read."""

    def __init__(self, goals, assists, minutes, team_goals, player_id=1):
        self.player_id = player_id
        self.goals = goals
        self.assists = assists
        self.minutes = minutes
        self.opponent = "AWAY"
        self.fixture = type("F", (), {"home_team": "AWAY", "away_team": "HOME"})()
        self.result = type("R", (), {"away_score": team_goals, "home_score": 0})()


def test_a_perfect_share_has_no_error():
    """A player who took a third of a hat-trick, predicted as a third."""
    score = score_involvement_error(
        frame(prob_score=1 / 3, prob_assist=0.0),
        [Performance(goals=1, assists=0, minutes=90, team_goals=3)],
    )
    assert score.n_observations == 1
    assert score.mean_absolute_error_goals == pytest.approx(0.0)


def test_the_error_is_conditioned_on_the_minutes_actually_played():
    """
    Half a match means half the share, so the minutes model is not on trial.

    A player predicted to take every goal who played 45 minutes of a two-goal
    win is expected to have scored one, not two.
    """
    score = score_involvement_error(
        frame(prob_score=1.0, prob_assist=0.0),
        [Performance(goals=1, assists=0, minutes=45, team_goals=2)],
    )
    assert score.mean_absolute_error_goals == pytest.approx(0.0)


def test_assists_are_scored_separately_from_goals():
    score = score_involvement_error(
        frame(prob_score=0.0, prob_assist=0.5),
        [Performance(goals=0, assists=0, minutes=90, team_goals=2)],
    )
    assert score.mean_absolute_error_goals == pytest.approx(0.0)
    assert score.mean_absolute_error_assists == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("goals", "assists", "minutes", "team_goals"),
    [(0, 0, 90, 0), (0, 0, 0, 2)],
)
def test_a_performance_the_model_says_nothing_about_is_skipped(
    goals, assists, minutes, team_goals
):
    """No goals to share out, or no minutes to share them over."""
    score = score_involvement_error(
        frame(prob_score=0.5, prob_assist=0.0),
        [Performance(goals, assists, minutes, team_goals)],
    )
    assert score.n_observations == 0
    assert score.n_skipped == 1


def test_a_player_the_model_was_not_fitted_for_is_skipped():
    score = score_involvement_error(
        frame(prob_score=0.5, prob_assist=0.0, player_id=1),
        [Performance(goals=1, assists=0, minutes=90, team_goals=1, player_id=999)],
    )
    assert score.n_observations == 0
    assert score.n_skipped == 1


def test_scores_add():
    total = InvolvementScore(1.0, 2.0, 1, 0) + InvolvementScore(3.0, 4.0, 3, 2)
    assert total.n_observations == 4
    assert total.n_skipped == 2
    assert total.mean_absolute_error_goals == pytest.approx(1.0)
    assert total.mean_absolute_error_assists == pytest.approx(1.5)


def test_an_empty_score_is_a_number_not_a_crash():
    empty = InvolvementScore()
    assert empty.mean_absolute_error_goals == 0.0
    assert empty.mean_absolute_error_assists == 0.0
