"""
A player model fitted to expected goals rather than to goals.

What it has to get right is that a share is a share of the team's expected
goals, and that a chance counts whether or not it went in.
`tests/e2e/test_player_models.py` fits it against a real database as one of the
table entries.
"""

import numpy as np
import pytest

from airsenal.game.enums import Position
from airsenal.prediction.player_models import build_player_model
from airsenal.prediction.player_models.xg import (
    DEFAULT_XG_GOAL_WEIGHT,
    DEFAULT_XG_N_GOALS_PRIOR,
    DEFAULT_XG_PLAYER_EPSILON,
    XGPlayerConfig,
    XGPlayerModel,
)

# Nothing between the data and the fitted shares: no prior, no time weighting,
# no calibration and no realised goals mixed in, so a test that asserts an
# arithmetic result is asserting only what it says.
NO_TIME_WEIGHTING = XGPlayerConfig(
    epsilon=None, n_goals_prior=0, calibrate=False, goal_weight=0.0
)


def fit_data(
    *,
    expected_goals,
    expected_assists,
    team_expected_goals,
    goals=None,
    assists=None,
    minutes=None,
    position="MID",
    drop_expected=(),
):
    """
    A training frame for `n_players` over `n_matches`, as a player model gets one.

    `goals` and `assists` default to the expected values rounded down, so a
    caller that only cares about expected goals gets a consistent `y` for free.
    """
    expected_goals = np.asarray(expected_goals, dtype=float)
    expected_assists = np.asarray(expected_assists, dtype=float)
    team_expected_goals = np.asarray(team_expected_goals, dtype=float)
    n_players, n_matches = expected_goals.shape
    goals = np.floor(expected_goals) if goals is None else np.asarray(goals)
    assists = np.floor(expected_assists) if assists is None else np.asarray(assists)
    team_goals = np.floor(team_expected_goals)
    y = np.stack([goals, assists, team_goals - goals - assists], axis=2)
    data = {
        "position": position,
        "player_ids": np.arange(n_players),
        "nplayer": n_players,
        "nmatch": n_matches,
        "minutes": 90 * np.ones((n_players, n_matches))
        if minutes is None
        else np.asarray(minutes),
        "y": y,
        "alpha": np.ones(3),
        "time_diff": np.zeros((n_players, n_matches)),
        "expected_goals": expected_goals,
        "expected_assists": expected_assists,
        "team_expected_goals": team_expected_goals,
    }
    for key in drop_expected:
        del data[key]
    return data


def shares(model):
    """(prob_score, prob_assist) per player, which is what the fit is for."""
    involvement = model.predict_involvement()
    return involvement.prob_score, involvement.prob_assist


def test_a_share_is_of_the_teams_expected_goals():
    """
    Two players, one expected to take a quarter of his team's chances.

    With no prior and no scaling to argue about, the share is the arithmetic.
    """
    data = fit_data(
        expected_goals=np.full((2, 4), 0.5),
        expected_assists=np.zeros((2, 4)),
        team_expected_goals=np.full((2, 4), 2.0),
    )
    prob_score, _ = shares(XGPlayerModel(NO_TIME_WEIGHTING).fit(data))
    assert prob_score == pytest.approx([0.25, 0.25])


def test_a_goalless_match_is_still_evidence():
    """
    The whole point of the model.

    Both players create the same over four matches, but one team scored in every
    one and the other in none. The goals model has nothing to say about the
    second player; this one rates them the same.
    """
    data = fit_data(
        expected_goals=np.full((2, 4), 0.5),
        expected_assists=np.zeros((2, 4)),
        team_expected_goals=np.full((2, 4), 2.0),
        # the second player's team never actually scored
        goals=np.array([[1.0] * 4, [0.0] * 4]),
        assists=np.zeros((2, 4)),
    )
    data["y"][1] = 0
    prob_score, _ = shares(XGPlayerModel(NO_TIME_WEIGHTING).fit(data))
    assert prob_score[0] == pytest.approx(prob_score[1])


def test_a_bigger_share_of_the_same_chances_rates_higher():
    data = fit_data(
        expected_goals=np.array([[0.8] * 4, [0.4] * 4, [0.1] * 4]),
        expected_assists=np.zeros((3, 4)),
        team_expected_goals=np.full((3, 4), 1.6),
    )
    prob_score, _ = shares(XGPlayerModel().fit(data))
    assert list(prob_score) == sorted(prob_score, reverse=True)


def test_a_player_who_played_half_the_match_is_rated_per_ninety():
    """
    The share is what the player would take over a full match.

    Same expected goals in half the minutes is twice the player, which is
    `scale_goals_by_minutes` doing the same job it does for the goals model.
    """
    data = fit_data(
        expected_goals=np.full((2, 4), 0.4),
        expected_assists=np.zeros((2, 4)),
        team_expected_goals=np.full((2, 4), 2.0),
        minutes=np.array([[90.0] * 4, [45.0] * 4]),
    )
    prob_score, _ = shares(XGPlayerModel(NO_TIME_WEIGHTING).fit(data))
    assert prob_score[1] == pytest.approx(2 * prob_score[0])


def test_calibration_is_what_the_window_converted_its_chances_into():
    """
    Expected assists undershoot the assists FPL awards, and it is measured.

    Here the players took 6 expected assists and were credited with 12, so an
    expected assist is worth two and the fitted share says so.
    """
    data = fit_data(
        expected_goals=np.zeros((2, 4)),
        expected_assists=np.full((2, 4), 0.75),
        team_expected_goals=np.full((2, 4), 3.0),
        goals=np.zeros((2, 4)),
        assists=np.full((2, 4), 1.5),
    )
    model = XGPlayerModel(XGPlayerConfig(epsilon=None, n_goals_prior=0)).fit(data)
    assert model.creation == pytest.approx(2.0)
    _, prob_assist = shares(model)
    assert prob_assist == pytest.approx([0.5, 0.5])


def test_calibration_can_be_turned_off():
    data = fit_data(
        expected_goals=np.zeros((2, 4)),
        expected_assists=np.full((2, 4), 0.75),
        team_expected_goals=np.full((2, 4), 3.0),
        goals=np.zeros((2, 4)),
        assists=np.full((2, 4), 1.5),
    )
    model = XGPlayerModel(NO_TIME_WEIGHTING).fit(data)
    assert model.creation == 1.0
    _, prob_assist = shares(model)
    assert prob_assist == pytest.approx([0.25, 0.25])


def test_a_position_that_neither_scores_nor_is_expected_to_is_not_a_division_by_zero():
    """Goalkeepers, whose expected goals over a window can be exactly nothing."""
    data = fit_data(
        expected_goals=np.zeros((2, 4)),
        expected_assists=np.zeros((2, 4)),
        team_expected_goals=np.full((2, 4), 2.0),
    )
    model = XGPlayerModel().fit(data)
    assert (model.finishing, model.creation) == (1.0, 1.0)
    prob_score, prob_assist = shares(model)
    assert prob_score == pytest.approx([0.0, 0.0])
    assert prob_assist == pytest.approx([0.0, 0.0])


def test_blending_all_the_way_to_goals_is_fitting_to_goals():
    """
    The ends of the mixing weight are the two models it mixes.

    At one the expected goals are not consulted at all, so this is the conjugate
    model's count reached the long way round.
    """
    data = fit_data(
        expected_goals=np.full((2, 4), 0.5),
        expected_assists=np.zeros((2, 4)),
        team_expected_goals=np.full((2, 4), 2.0),
        goals=np.array([[2.0] * 4, [0.0] * 4]),
        assists=np.zeros((2, 4)),
    )
    config = XGPlayerConfig(epsilon=None, n_goals_prior=0, goal_weight=1.0)
    prob_score, _ = shares(XGPlayerModel(config).fit(data))
    assert prob_score == pytest.approx([1.0, 0.0])


def test_the_prior_is_the_one_for_the_position_being_fitted():
    """
    Every position wants a different shrinkage, so the fit data says which it is.

    Two identical frames, fitted with a prior of nothing for one position and a
    heavy one for the other: the heavy prior pulls the fitted share towards the
    pooled squad and the empty one leaves it where the data put it.
    """
    config = XGPlayerConfig(
        epsilon=None,
        calibrate=False,
        goal_weight=0.0,
        n_goals_prior={"MID": 0, "FWD": 400},
    )
    counts = {
        "expected_goals": np.array([[1.2] * 4, [0.1] * 4]),
        "expected_assists": np.zeros((2, 4)),
        "team_expected_goals": np.full((2, 4), 2.0),
    }
    light = shares(XGPlayerModel(config).fit(fit_data(position="MID", **counts)))[0]
    heavy = shares(XGPlayerModel(config).fit(fit_data(position="FWD", **counts)))[0]
    assert light[0] > heavy[0]
    assert light[1] < heavy[1]


def test_a_single_prior_is_still_a_single_prior():
    """Which is what a sweep over one number needs, and what the tools pass."""
    config = XGPlayerConfig(n_goals_prior=12)
    assert config.prior_for("MID") == 12
    assert config.prior_for(None) == 12


def test_a_per_position_prior_needs_the_data_to_say_which_position():
    """
    Rather than quietly fitting goalkeepers with a midfielder's shrinkage.

    Training data a caller assembled themselves has no position in it, and the
    default configuration is per-position, so this is the error they get.
    """
    data = fit_data(
        expected_goals=np.full((2, 4), 0.5),
        expected_assists=np.zeros((2, 4)),
        team_expected_goals=np.full((2, 4), 2.0),
    )
    del data["position"]
    with pytest.raises(ValueError, match="which position"):
        XGPlayerModel().fit(data)


def test_a_position_the_prior_has_never_heard_of_is_refused():
    data = fit_data(
        expected_goals=np.full((2, 4), 0.5),
        expected_assists=np.zeros((2, 4)),
        team_expected_goals=np.full((2, 4), 2.0),
        position="MNG",
    )
    with pytest.raises(ValueError, match="No n_goals_prior for position 'MNG'"):
        XGPlayerModel().fit(data)


def test_a_mixing_weight_outside_zero_and_one_is_refused():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        XGPlayerConfig(goal_weight=1.5)


def test_time_weighting_prefers_the_recent_chances():
    """A player who has stopped creating is not the player he was two years ago."""
    data = fit_data(
        expected_goals=np.array([[1.0, 0.0], [0.0, 1.0]]),
        expected_assists=np.zeros((2, 2)),
        team_expected_goals=np.full((2, 2), 2.0),
    )
    # the first match of the two is the old one
    data["time_diff"] = np.array([[2.0, 0.0], [2.0, 0.0]])
    prob_score, _ = shares(
        XGPlayerModel(XGPlayerConfig(n_goals_prior=0, calibrate=False)).fit(data)
    )
    assert prob_score[1] > prob_score[0]


@pytest.mark.parametrize(
    "dropped", ["expected_goals", "expected_assists", "team_expected_goals"]
)
def test_training_data_without_expected_goals_is_refused(dropped):
    """Rather than quietly fitting to goals, which is a different model."""
    data = fit_data(
        expected_goals=np.full((2, 4), 0.5),
        expected_assists=np.zeros((2, 4)),
        team_expected_goals=np.full((2, 4), 2.0),
        drop_expected=[dropped],
    )
    with pytest.raises(ValueError, match=dropped):
        XGPlayerModel().fit(data)


def test_training_data_where_nothing_was_recorded_is_refused():
    """
    And the refusal names the way out.

    Expected goals only exist from season 2223, so anyone fitting an earlier one
    hits this and needs to be told which model to ask for instead.
    """
    data = fit_data(
        expected_goals=np.full((2, 4), np.nan),
        expected_assists=np.full((2, 4), np.nan),
        team_expected_goals=np.full((2, 4), np.nan),
    )
    with pytest.raises(ValueError, match=r"No match") as excinfo:
        XGPlayerModel().fit(data)
    assert "2223" in str(excinfo.value)
    assert "conjugate" in str(excinfo.value)


def test_matches_with_no_expected_goals_are_dropped_not_zeroed():
    """A match nobody recorded is not a chanceless one."""
    recorded = fit_data(
        expected_goals=np.full((2, 4), 0.5),
        expected_assists=np.zeros((2, 4)),
        team_expected_goals=np.full((2, 4), 2.0),
    )
    partial = fit_data(
        expected_goals=np.full((2, 4), 0.5),
        expected_assists=np.zeros((2, 4)),
        team_expected_goals=np.full((2, 4), 2.0),
    )
    partial["expected_goals"][:, 0] = np.nan
    assert shares(XGPlayerModel().fit(partial))[0] == pytest.approx(
        shares(XGPlayerModel().fit(recorded))[0]
    )


def test_an_unfitted_model_says_so():
    with pytest.raises(RuntimeError, match="not been fitted"):
        XGPlayerModel().predict_involvement()


def test_the_table_entry_builds_the_default_configuration():
    model = build_player_model("xg")
    assert isinstance(model, XGPlayerModel)
    assert model.config == XGPlayerConfig()
    assert model.config.epsilon == DEFAULT_XG_PLAYER_EPSILON
    assert model.config.goal_weight == DEFAULT_XG_GOAL_WEIGHT
    assert set(DEFAULT_XG_N_GOALS_PRIOR) == {str(p) for p in Position}
