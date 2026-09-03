"""
What a predicted score is made of, event by event.

Each function here turns one kind of event - a clean sheet, a goal, a card -
into points, given probabilities rather than a result. The fitted averages the
bonus, save and card ones read are built in test_point_components.py.
"""

import pandas as pd

from airsenal.prediction.points import (
    get_attacking_points,
    get_bonus_points,
    get_card_points,
    get_defending_points,
    get_save_points,
)


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
    assert get_save_points(1, "GK", 90, df_saves) == 1
    assert get_save_points(2, "GK", 90, df_saves) == 2
    # <60 mins - zero
    assert get_save_points(1, "GK", 50, df_saves) == 0
    # player not present in df_saves (no history)
    assert get_save_points(3, "GK", 90, df_saves) == 0
    # not a goalkeeper - zero
    assert get_save_points(1, "DEF", 90, df_saves) == 0


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
