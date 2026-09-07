"""
A team model fitted to expected goals rather than to goals.

What it has to get right is separating a good attack from an easy schedule,
which is what the alternating fit is for. `tests/e2e/test_team_models.py` fits
it against a real database as one of the table entries.
"""

import numpy as np
import pytest

from airsenal.prediction.team_models import build_team_model
from airsenal.prediction.team_models.scorelines import (
    DEFAULT_GOAL_DISPERSION,
    ConwayMaxwellScorelines,
)
from airsenal.prediction.team_models.xg import (
    DEFAULT_XG_EPSILON,
    XGTeamConfig,
    XGTeamModel,
)

TEAMS = ["AAA", "BBB", "CCC", "DDD"]


def training_data(matches, *, with_expected_goals=True):
    """(home, away, home xG, away xG) tuples, as a team model is fitted to."""
    home, away, home_xg, away_xg = zip(*matches, strict=True)
    data = {
        "home_team": np.array(home),
        "away_team": np.array(away),
        "home_goals": np.round(home_xg).astype(int),
        "away_goals": np.round(away_xg).astype(int),
        "time_diff": np.zeros(len(matches)),
        "neutral_venue": np.zeros(len(matches)),
        "game_weights": np.ones(len(matches)),
    }
    if with_expected_goals:
        data["home_expected_goals"] = np.array(home_xg, dtype=float)
        data["away_expected_goals"] = np.array(away_xg, dtype=float)
    return data


def round_robin(strength):
    """Every team plays every other home and away, creating `strength[team]` xG."""
    return [
        (home, away, strength[home] * 1.2, strength[away])
        for home in strength
        for away in strength
        if home != away
    ]


def test_the_ratings_average_one():
    """
    They are only identified up to a constant, so the level lives in the means.

    Without pinning them the fit could drift, and two fitted models would not be
    comparable team by team.
    """
    model = XGTeamModel().fit(
        training_data(round_robin({t: 1.0 + i for i, t in enumerate(TEAMS)}))
    )
    assert np.mean(list(model.attack.values())) == pytest.approx(1.0)
    assert np.mean(list(model.defence.values())) == pytest.approx(1.0)


def test_a_team_that_creates_more_has_a_better_attack():
    strength = {"AAA": 3.0, "BBB": 2.0, "CCC": 1.0, "DDD": 0.5}
    model = XGTeamModel().fit(training_data(round_robin(strength)))
    ratings = [model.attack[team] for team in strength]
    assert ratings == sorted(ratings, reverse=True)


def test_home_advantage_is_the_league_average_not_a_team_rating():
    """Everyone creates more at home here, so it belongs to the venue."""
    model = XGTeamModel().fit(training_data(round_robin(dict.fromkeys(TEAMS, 1.5))))
    assert model.home_mean > model.away_mean
    assert model.predict_expected_goals("AAA", "BBB", home=True) > (
        model.predict_expected_goals("AAA", "BBB", home=False)
    )


def test_an_easy_schedule_is_not_a_good_attack():
    """
    The point of fitting attack and defence together.

    AAA and BBB both create exactly 2.0 a match. AAA does it against CCC, which
    concedes 1.2 a match to everyone; BBB does it against DDD, which concedes
    2.5. Creating the same against a better defence is the better attack, and a
    model that only counted expected goals would call them equal.
    """
    matches = [
        *[("AAA", "CCC", 2.0, 0.5) for _ in range(6)],
        *[("CCC", "AAA", 3.0, 2.0) for _ in range(6)],
        *[("BBB", "DDD", 2.0, 0.5) for _ in range(6)],
        *[("DDD", "BBB", 0.3, 2.0) for _ in range(6)],
        # CCC and DDD play each other too, so their defences are comparable
        *[("CCC", "DDD", 3.0, 0.5) for _ in range(6)],
        *[("DDD", "CCC", 0.3, 3.0) for _ in range(6)],
    ]
    model = XGTeamModel().fit(training_data(matches))
    assert model.attack["AAA"] > model.attack["BBB"]
    # a defence rating multiplies what opponents create, so higher is worse
    assert model.defence["DDD"] > model.defence["CCC"]


def test_a_prediction_is_the_venue_the_attack_and_the_defence():
    model = XGTeamModel().fit(training_data(round_robin(dict.fromkeys(TEAMS, 1.5))))
    expected = model.home_mean * model.attack["AAA"] * model.defence["BBB"]
    assert model.predict_expected_goals("AAA", "BBB") == pytest.approx(expected)


def test_a_new_team_is_the_league_average_until_it_plays():
    model = XGTeamModel().fit(training_data(round_robin(dict.fromkeys(TEAMS, 1.5))))
    model.add_new_team("NEW")
    assert "NEW" in model.teams
    assert model.attack["NEW"] == 1.0
    assert model.predict_expected_goals("NEW", "AAA") == pytest.approx(
        model.home_mean * model.defence["AAA"]
    )


def test_shrinkage_pulls_a_short_record_towards_the_average():
    """One freakish match should not make a team the best in the league."""
    matches = [
        ("AAA", "BBB", 6.0, 0.1),
        *[("CCC", "DDD", 1.5, 1.5) for _ in range(20)],
        *[("DDD", "CCC", 1.5, 1.5) for _ in range(20)],
    ]
    heavy = XGTeamModel(XGTeamConfig(prior_matches=20.0)).fit(training_data(matches))
    light = XGTeamModel(XGTeamConfig(prior_matches=0.5)).fit(training_data(matches))
    assert light.attack["AAA"] > heavy.attack["AAA"]


def test_training_data_without_expected_goals_is_refused():
    """Rather than quietly fitting to goals, which is a different model."""
    with pytest.raises(ValueError, match="expected goals"):
        XGTeamModel().fit(
            training_data(
                round_robin(dict.fromkeys(TEAMS, 1.5)), with_expected_goals=False
            )
        )


def test_training_data_where_nothing_was_recorded_is_refused():
    """
    And the refusal names the way out, because this is now the default model.

    Expected goals only exist from season 2223, so anyone fitting an earlier one
    hits this and needs to be told which model to ask for instead.
    """
    data = training_data(round_robin(dict.fromkeys(TEAMS, 1.5)))
    data["home_expected_goals"] = np.full(len(data["home_team"]), np.nan)
    with pytest.raises(ValueError, match=r"No match") as excinfo:
        XGTeamModel().fit(data)
    assert "2223" in str(excinfo.value)
    assert "extended" in str(excinfo.value)


def test_matches_with_no_expected_goals_are_dropped_not_zeroed():
    """A match nobody recorded is not a goalless one."""
    matches = round_robin(dict.fromkeys(TEAMS, 1.5))
    data = training_data(matches)
    data["home_expected_goals"][0] = np.nan
    model = XGTeamModel().fit(data)
    assert model.home_mean == pytest.approx(1.8)


def test_an_unfitted_model_says_so():
    with pytest.raises(RuntimeError, match="not been fitted"):
        XGTeamModel().predict_expected_goals("AAA", "BBB")


def test_a_team_nobody_has_heard_of_is_predicted_as_a_promoted_one():
    """The same assumption, for a name that never reached `add_new_team`."""
    model = XGTeamModel(XGTeamConfig(promoted_like_bottom=2)).fit(
        training_data(round_robin({t: 1.0 + i for i, t in enumerate(TEAMS)}))
    )
    assert model.predict_expected_goals("NEVER_SEEN", "AAA") == pytest.approx(
        model.home_mean * model.promoted_attack * model.defence["AAA"]
    )


def test_by_default_a_team_with_no_record_is_an_average_one():
    """
    Which is not the likely story, but is the better prediction here.

    Assuming a promoted team resembles the teams it replaced measured worse than
    assuming it is average, across the two seasons that have promoted teams in
    this database - so the plausible assumption is available and off. See
    docs/prediction-seams-plan.md.
    """
    model = XGTeamModel().fit(
        training_data(round_robin({t: 1.0 + i for i, t in enumerate(TEAMS)}))
    )
    assert model.config.promoted_like_bottom is None
    assert model.promoted_attack == 1.0
    assert model.promoted_defence == 1.0
    model.add_new_team("NEW")
    assert model.attack["NEW"] == 1.0


def test_the_table_entry_wraps_it_so_it_has_scorelines():
    """
    It predicts a mean; the points calculation needs a distribution.

    Conway-Maxwell rather than Poisson, so the spread of that distribution is a
    swept number instead of an assumption - see `DEFAULT_GOAL_DISPERSION`.
    """
    built = build_team_model("xg")
    assert isinstance(built, ConwayMaxwellScorelines)
    assert built.dispersion == DEFAULT_GOAL_DISPERSION
    assert isinstance(built.model, XGTeamModel)


def test_the_table_entry_passes_epsilon_to_the_model():
    """`--epsilon` is time weighting, and this model does time-weight."""
    assert build_team_model("xg").model.config.epsilon == DEFAULT_XG_EPSILON
    assert build_team_model("xg", 0.25).model.config.epsilon == 0.25


def test_time_weighting_prefers_what_happened_recently():
    """
    A team that has got worse is rated on the matches it has just played.

    AAA was the best attack in the league a year ago and is ordinary now, while
    everyone else has stayed the same. With no decay the model averages the two
    halves; with decay it mostly believes the recent one.

    Every team plays home and away, which matters: a team that only ever played
    at home would have its attack confounded with the league's home advantage,
    and the venue mean would absorb the whole change.
    """
    steady = {"BBB": 1.5, "CCC": 1.5, "DDD": 1.5}
    old_matches = round_robin({"AAA": 3.0, **steady})
    new_matches = round_robin({"AAA": 0.5, **steady})
    data = training_data([*old_matches, *new_matches])
    data["time_diff"] = np.array([1.0] * len(old_matches) + [0.0] * len(new_matches))

    flat = XGTeamModel(XGTeamConfig(epsilon=0.0)).fit(data)
    weighted = XGTeamModel(XGTeamConfig(epsilon=2.0)).fit(data)
    assert weighted.attack["AAA"] < flat.attack["AAA"]
    # and the teams that did not change are rated much the same either way
    for team in steady:
        assert weighted.attack[team] > flat.attack[team]


def test_a_promoted_team_is_rated_like_the_teams_it_replaced():
    """
    Not like the league average, which is what it is not.

    A side with no record in the window has just come up, so it is assumed to
    resemble the worst teams that do have one.
    """
    strength = {"AAA": 3.0, "BBB": 2.0, "CCC": 1.0, "DDD": 0.5}
    model = XGTeamModel(XGTeamConfig(promoted_like_bottom=2)).fit(
        training_data(round_robin(strength))
    )
    model.add_new_team("NEW")
    assert model.attack["NEW"] < 1.0
    assert model.attack["NEW"] == pytest.approx(
        (model.attack["CCC"] + model.attack["DDD"]) / 2
    )
    # and it is expected to create less than an average side would
    assert (
        model.predict_expected_goals("NEW", "AAA")
        < model.home_mean * model.defence["AAA"]
    )


def test_the_ratings_do_not_depend_on_how_far_ahead_the_target_is():
    """
    `time_diff` is measured back from the gameweek being predicted.

    Aiming further ahead multiplies every weight by the same factor, and the
    shrinkage prior is in absolute units, so without rescaling the weights every
    rating would creep towards the league average the further ahead you asked -
    on identical training data. Two fits over the same matches, one aimed a
    season further ahead, must agree.
    """
    strength = {"AAA": 3.0, "BBB": 2.0, "CCC": 1.0, "DDD": 0.5}
    matches = round_robin(strength)
    near = training_data(matches)
    near["time_diff"] = np.linspace(0.0, 1.0, len(matches))
    far = training_data(matches)
    far["time_diff"] = near["time_diff"] + 1.0

    fitted_near = XGTeamModel().fit(near)
    fitted_far = XGTeamModel().fit(far)
    for team in strength:
        assert fitted_far.attack[team] == pytest.approx(fitted_near.attack[team])
        assert fitted_far.defence[team] == pytest.approx(fitted_near.defence[team])


def test_the_most_recent_match_in_the_window_counts_as_one():
    """Which is what makes `prior_matches` a number of matches."""
    matches = round_robin(dict.fromkeys(TEAMS, 1.5))
    data = training_data(matches)
    data["time_diff"] = np.linspace(2.0, 3.0, len(matches))
    model = XGTeamModel()
    played = np.ones(len(matches), dtype=bool)
    weights = model._weights(data, played)
    assert weights.max() == pytest.approx(1.0)
    # and older matches still count for less, which is the point of weighting
    assert weights.min() < 1.0
