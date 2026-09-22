"""
What a predicted score is made of, component by component.

Each component turns one kind of event - a clean sheet, a goal, a card - into
points, given probabilities rather than a result. The fitted averages the
bonus, save and card ones read are built in test_point_components.py; here they
are set directly, so what is under test is what a component does with them.
"""

import pandas as pd
import pytest

from airsenal.prediction.point_components import (
    BonusComponent,
    CardComponent,
    SaveComponent,
    get_attacking_points,
    get_defending_points,
)
from airsenal.prediction.protocols import ComponentRequest


def request(player_id=1, position="MID", minutes=90):
    """A request carrying only what the fitted components read."""
    return ComponentRequest(
        player_id=player_id,
        position=position,
        minutes=minutes,
        team_score_probability={0: 1.0},
        team_concede_probability={0: 1.0},
        prob_score=0.0,
        prob_assist=0.0,
    )


def test_defending_points_0_conceded():
    """Defenders and keepers get the clean-sheet bonus for a 0-0, if they played 60."""
    assert get_defending_points("FWD", 90, {0: 1.0}) == 0
    assert get_defending_points("MID", 90, {0: 1.0}) == 1
    assert get_defending_points("DEF", 90, {0: 1.0}) == 4
    assert get_defending_points("GK", 90, {0: 1.0}) == 4
    for pos in ["FWD", "MID", "DEF", "GK"]:
        assert get_defending_points(pos, 59, {0: 1.0}) == 0


def test_defending_points_2_conceded():
    """Defenders and keepers lose a point for two goals conceded."""
    concede_probs = {0: 0.0, 1: 0.0, 2: 1.0}
    assert get_defending_points("FWD", 90, concede_probs) == 0
    assert get_defending_points("MID", 90, concede_probs) == 0
    assert get_defending_points("DEF", 90, concede_probs) == -1
    assert get_defending_points("GK", 90, concede_probs) == -1
    for pos in ["DEF", "GK"]:
        assert get_defending_points(pos, 60, concede_probs) == -2 / 3


def test_defending_points_4_conceded():
    """Defenders and keepers lose two points for four goals conceded."""
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


def test_bonus_component():
    """Bonus points come back from the fitted average."""
    component = BonusComponent()
    component.fitted = (pd.Series({1: 1, 2: 2}), pd.Series({1: 0.5, 2: 0.25}))

    # 90 mins - use the full-match average
    assert component.expected_points(request(1, minutes=90)) == 1
    assert component.expected_points(request(2, minutes=90)) == 2
    # 45 mins - use the short-appearance average
    assert component.expected_points(request(1, minutes=45)) == 0.5
    assert component.expected_points(request(2, minutes=45)) == 0.25
    # under 30 mins - zero
    assert component.expected_points(request(1, minutes=20)) == 0
    assert component.expected_points(request(1, minutes=0)) == 0
    # a player with no bonus history
    assert component.expected_points(request(3, minutes=90)) == 0


def test_save_component():
    """Save points come back from the fitted average."""
    component = SaveComponent()
    component.fitted = pd.Series({1: 1, 2: 2})

    # over 60 mins - return the fitted value
    assert component.expected_points(request(1, "GK", 90)) == 1
    assert component.expected_points(request(2, "GK", 90)) == 2
    # under 60 mins - zero
    assert component.expected_points(request(1, "GK", 50)) == 0
    # a keeper with no history
    assert component.expected_points(request(3, "GK", 90)) == 0
    # not a goalkeeper - zero
    assert component.expected_points(request(1, "DEF", 90)) == 0


def test_card_component():
    """Card points come back from the fitted average."""
    component = CardComponent()
    component.fitted = pd.Series({1: -1, 2: -2})

    # over 30 mins - return the fitted value
    assert component.expected_points(request(1, minutes=90)) == -1
    assert component.expected_points(request(2, minutes=45)) == -2
    # under 30 mins - zero
    assert component.expected_points(request(1, minutes=20)) == 0
    # a player with no card history
    assert component.expected_points(request(3, minutes=90)) == 0


def test_a_component_that_has_not_been_fitted_says_so():
    """
    Rather than returning zero, which would look like a clean prediction.

    A run configured without a component does not include it, so a component
    that is present but unfitted is a bug.
    """
    for component in (BonusComponent(), CardComponent(), SaveComponent()):
        with pytest.raises(RuntimeError, match="not been fitted"):
            component.expected_points(request(1, "GK", 90))
