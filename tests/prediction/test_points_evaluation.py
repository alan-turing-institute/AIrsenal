"""
Scoring predicted points against what players actually scored.

The arithmetic of `PointsScore`, on predictions built by hand rather than
fitted: the point of this scorer is that it asks nothing of a model beyond a
number per player per fixture, so nothing here constructs a model at all.
`tests/e2e/test_evaluation.py` runs it against real predictions.
"""

import math

import pytest

from airsenal.db.models import PlayerPrediction, PlayerScore
from airsenal.prediction.evaluation import PointsScore, score_points_predictions


def prediction(player_id: int, fixture_id: int, points: float) -> PlayerPrediction:
    return PlayerPrediction(
        player_id=player_id, fixture_id=fixture_id, predicted_points=points, tag="test"
    )


def performance(
    player_id: int, fixture_id: int, points: int, minutes: int = 90
) -> PlayerScore:
    return PlayerScore(
        player_id=player_id, fixture_id=fixture_id, points=points, minutes=minutes
    )


def test_an_empty_score_is_a_number_not_a_crash():
    empty = PointsScore()
    assert empty.mean_absolute_error == 0.0
    assert empty.mean_absolute_error_played == 0.0
    assert empty.root_mean_squared_error == 0.0
    assert empty.mean_rank_correlation == 0.0


def test_scores_add():
    """Every total adds and every count adds, so a backtest can accumulate."""
    first = PointsScore(
        total_absolute_error=6.0,
        total_squared_error=12.0,
        n_observations=2,
        total_absolute_error_played=5.0,
        n_played=1,
        n_skipped=1,
        total_rank_correlation=0.5,
        n_ranked=1,
    )
    second = PointsScore(
        total_absolute_error=4.0,
        total_squared_error=8.0,
        n_observations=2,
        total_absolute_error_played=4.0,
        n_played=1,
        n_skipped=0,
        total_rank_correlation=1.0,
        n_ranked=1,
    )
    total = first + second
    assert total.n_observations == 4
    assert total.n_skipped == 1
    assert total.mean_absolute_error == pytest.approx(10.0 / 4)
    assert total.mean_absolute_error_played == pytest.approx(9.0 / 2)
    assert total.mean_rank_correlation == pytest.approx(0.75)


def test_a_perfect_prediction_has_no_error():
    predictions = [prediction(i, 1, float(i)) for i in range(1, 4)]
    performances = [performance(i, 1, i) for i in range(1, 4)]
    score = score_points_predictions(predictions, performances)
    assert score.n_observations == 3
    assert score.total_absolute_error == 0.0
    assert score.root_mean_squared_error == 0.0
    assert score.mean_rank_correlation == pytest.approx(1.0)


def test_the_error_is_the_gap_between_prediction_and_outcome():
    """Predicted 5, 2, 8 against 3, 6, 9 - so the misses are 2, -4 and -1."""
    predictions = [prediction(1, 1, 5.0), prediction(2, 1, 2.0), prediction(3, 1, 8.0)]
    performances = [performance(1, 1, 3), performance(2, 1, 6), performance(3, 1, 9)]
    score = score_points_predictions(predictions, performances)
    assert score.n_observations == 3
    assert score.total_absolute_error == pytest.approx(7.0)
    assert score.mean_absolute_error == pytest.approx(7.0 / 3)
    assert score.total_squared_error == pytest.approx(21.0)
    assert score.root_mean_squared_error == pytest.approx(math.sqrt(7.0))


def test_a_double_gameweek_is_two_observations():
    """One player, two fixtures - matched per fixture, not summed into one."""
    predictions = [prediction(1, 1, 4.0), prediction(1, 2, 4.0)]
    performances = [performance(1, 1, 2), performance(1, 2, 10)]
    score = score_points_predictions(predictions, performances)
    assert score.n_observations == 2
    assert score.total_absolute_error == pytest.approx(8.0)


def test_a_prediction_with_no_performance_is_skipped_not_guessed():
    """The fixture may not have been played; that is not the model being wrong."""
    predictions = [prediction(1, 1, 5.0), prediction(2, 1, 6.0)]
    performances = [performance(1, 1, 5)]
    score = score_points_predictions(predictions, performances)
    assert score.n_observations == 1
    assert score.n_skipped == 1
    assert score.total_absolute_error == 0.0


def test_a_performance_nothing_predicted_is_skipped_too():
    """A run that covers barely any of the players who played has to say so."""
    predictions = [prediction(1, 1, 5.0)]
    performances = [performance(1, 1, 5), performance(2, 1, 9)]
    score = score_points_predictions(predictions, performances)
    assert score.n_observations == 1
    assert score.n_skipped == 1


def test_a_prediction_for_the_wrong_fixture_is_not_scored():
    """Matching on the player alone would score a prediction against any match."""
    score = score_points_predictions([prediction(1, 1, 5.0)], [performance(1, 2, 5)])
    assert score.n_observations == 0
    assert score.n_skipped == 2


@pytest.mark.parametrize(
    ("predicted", "expected"),
    [([1.0, 2.0, 3.0], 1.0), ([3.0, 2.0, 1.0], -1.0), ([2.0, 1.0, 3.0], 0.5)],
)
def test_the_rank_correlation_measures_the_order_not_the_level(predicted, expected):
    """Ten times the points in the right order still ranks perfectly."""
    predictions = [prediction(i, 1, p * 10) for i, p in enumerate(predicted, 1)]
    performances = [performance(i, 1, i) for i in range(1, 4)]
    score = score_points_predictions(predictions, performances)
    assert score.n_ranked == 1
    assert score.mean_rank_correlation == pytest.approx(expected)


def test_a_run_that_predicts_the_same_for_everyone_is_not_ranked():
    """There is no order to be right or wrong about, so no correlation is claimed."""
    predictions = [prediction(i, 1, 4.0) for i in range(1, 4)]
    performances = [performance(i, 1, i) for i in range(1, 4)]
    score = score_points_predictions(predictions, performances)
    assert score.n_observations == 3
    assert score.n_ranked == 0
    assert score.mean_rank_correlation == 0.0


def test_too_few_players_to_rank_are_still_scored_for_error():
    """Two players rank right or wrong by luck; their error is still real."""
    score = score_points_predictions(
        [prediction(1, 1, 5.0), prediction(2, 1, 1.0)],
        [performance(1, 1, 4), performance(2, 1, 2)],
    )
    assert score.n_observations == 2
    assert score.total_absolute_error == pytest.approx(2.0)
    assert score.n_ranked == 0


def test_scoring_needs_only_a_number_per_player_and_fixture():
    """
    Scoring asks nothing of a model beyond its predicted points.

    Whatever produced these predictions need not be probabilistic, need not
    predict a share of team goals, and need not decompose a score into
    components. See docs/adding-a-model.md.
    """
    predictions = [prediction(i, 1, float(4 - i)) for i in range(1, 4)]
    performances = [performance(i, 1, i) for i in range(1, 4)]
    score = score_points_predictions(predictions, performances)
    assert score.n_observations == 3
    assert math.isfinite(score.mean_absolute_error)
    assert score.mean_rank_correlation == pytest.approx(-1.0)


def test_a_player_who_did_not_play_and_was_predicted_zero_costs_nothing():
    """
    Which is most non-appearances: the points calculation predicts them at zero.

    They are still observations - a model that predicted points for them would
    be wrong - but they contribute no error, so they pull `mean_absolute_error`
    below the error on the players a squad is picked from.
    """
    predictions = [prediction(1, 1, 0.0), prediction(2, 1, 6.0)]
    performances = [performance(1, 1, 0, minutes=0), performance(2, 1, 2)]
    score = score_points_predictions(predictions, performances)
    assert score.n_observations == 2
    assert score.n_played == 1
    assert score.total_absolute_error == pytest.approx(4.0)
    # the whole error came from the player who played
    assert score.mean_absolute_error == pytest.approx(2.0)
    assert score.mean_absolute_error_played == pytest.approx(4.0)


def test_a_benched_player_who_was_predicted_points_is_an_error():
    """The non-appearances that do cost something: predicted to start, benched."""
    score = score_points_predictions(
        [prediction(1, 1, 3.0)], [performance(1, 1, 0, minutes=0)]
    )
    assert score.n_observations == 1
    assert score.n_played == 0
    assert score.mean_absolute_error == pytest.approx(3.0)
    # nothing appeared, so there is no played-only error to report
    assert score.mean_absolute_error_played == 0.0
