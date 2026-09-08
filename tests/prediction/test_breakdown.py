"""
Scoring whatever part of a prediction a model is willing to report.

The answer to "was it the minutes or the shares I got wrong?" - and still an
answer for a model that decomposes into nothing at all, which is the property
the whole seam exists to allow.
"""

import pytest

from airsenal.prediction.evaluation import (
    BreakdownScore,
    ErrorScore,
    score_prediction_breakdown,
)
from airsenal.prediction.protocols import (
    InvolvementShare,
    PointsPrediction,
    PointsRequest,
)

SEASON = "2526"


class Fixture:
    """The parts of a fixture the scorer reads."""

    def __init__(self, gameweek=1):
        self.gameweek = gameweek
        self.fixture_id = 1
        self.home_team = "AWAY"
        self.away_team = "HOME"


class Player:
    def __init__(self, position="MID"):
        self.player_id = 1
        self._position = position

    def position(self, season):
        del season
        return self._position


class Performance:
    """
    The parts of a `PlayerScore` the breakdown scorer and the decomposition read.

    Not the ORM class: its `player` and `fixture` are real relationships, and
    SQLAlchemy will not have them pointed at stubs.
    """

    def __init__(
        self, points=2, minutes=90, goals=0, assists=0, team_goals=0, **kwargs
    ):
        self.player_id = 1
        self.points = points
        self.minutes = minutes
        self.goals = goals
        self.assists = assists
        self.bonus = kwargs.get("bonus", 0)
        self.conceded = kwargs.get("conceded", 0)
        self.clean_sheets = kwargs.get("clean_sheets", 0)
        self.saves = 0
        self.yellow_cards = 0
        self.red_cards = 0
        self.own_goals = 0
        self.penalties_saved = 0
        self.penalties_missed = 0
        self.defensive_contribution = kwargs.get("defensive_contribution")
        self.opponent = "AWAY"
        self.fixture = Fixture()
        self.player = Player(kwargs.get("position", "MID"))
        self.result = type("R", (), {"away_score": team_goals, "home_score": 0})()


performance = Performance


class StubModel:
    """A points model that returns whatever prediction the test hands it."""

    def __init__(self, prediction):
        self.prediction = prediction
        self.asked = []

    def fit(self, request):
        del request
        return self

    def predict(self, request: PointsRequest) -> PointsPrediction:
        self.asked.append(request)
        return self.prediction


def score(prediction, performances):
    return score_prediction_breakdown(
        StubModel(prediction),
        performances,
        root_gameweek=1,
        season=SEASON,
        dbsession=None,
    )


def test_a_model_that_reports_only_a_total_is_scored_only_on_it():
    """
    The property the seam exists for.

    An end-to-end regressor has no minutes, no shares and no components, and is
    not penalised for the parts it does not claim to have.
    """
    result = score(PointsPrediction(expected_points=5.0), [performance(points=2)])
    assert result.points.n_observations == 1
    assert result.points.mean_absolute_error == pytest.approx(3.0)
    assert result.minutes is None
    assert result.involvement is None
    assert result.components is None


def test_reported_minutes_are_scored_against_what_was_played():
    result = score(
        PointsPrediction(expected_points=2.0, expected_minutes=70.0),
        [performance(minutes=45)],
    )
    assert result.minutes is not None
    assert result.minutes.mean_absolute_error == pytest.approx(25.0)


def test_reported_shares_are_scored_at_the_minutes_actually_played():
    """
    A rate, not a number of points, so the minutes error is not counted twice.

    Predicted to take every goal, played half the match, team scored two: one
    goal expected, one scored, no error - whatever the model thought about how
    long they would be on for.
    """
    result = score(
        PointsPrediction(
            expected_points=2.0,
            expected_minutes=90.0,
            involvement=InvolvementShare(prob_score=1.0, prob_assist=0.0),
        ),
        [performance(minutes=45, goals=1, team_goals=2)],
    )
    assert result.involvement is not None
    assert result.involvement.mean_absolute_error_goals == pytest.approx(0.0)


def test_each_reported_component_is_scored_against_its_own_outcome():
    """Two appearance points earned, three predicted: the appearance error is one."""
    result = score(
        PointsPrediction(
            expected_points=5.0,
            components={"appearance": 3.0, "attacking": 2.0},
        ),
        [performance(points=2, goals=0, conceded=1)],
    )
    assert result.components is not None
    assert result.components["appearance"].mean_absolute_error == pytest.approx(1.0)
    # nothing was scored, so the attacking component was two points wrong
    assert result.components["attacking"].mean_absolute_error == pytest.approx(2.0)


def test_a_component_the_outcome_has_no_view_on_is_left_alone():
    """`def_con` was not recorded before 25/26, so there is nothing to compare."""
    result = score(
        PointsPrediction(expected_points=2.0, components={"def_con": 1.0}),
        [performance(points=2, conceded=1)],
    )
    assert result.components == {}


def test_scores_add_across_performances():
    result = score(
        PointsPrediction(expected_points=4.0, expected_minutes=90.0),
        [performance(points=2, conceded=1), performance(points=6, conceded=1)],
    )
    assert result.points.n_observations == 2
    assert result.points.mean_absolute_error == pytest.approx(2.0)
    assert result.minutes is not None
    assert result.minutes.n_observations == 2


def test_adding_a_reported_part_to_an_unreported_one_keeps_it():
    """Two models, one of which reports minutes, still add up."""
    reported = BreakdownScore(minutes=ErrorScore(10.0, 1))
    unreported = BreakdownScore()
    assert (reported + unreported).minutes == ErrorScore(10.0, 1)
    assert (unreported + reported).minutes == ErrorScore(10.0, 1)
    assert (unreported + unreported).minutes is None


def test_an_empty_error_score_is_a_number_not_a_crash():
    assert ErrorScore().mean_absolute_error == 0.0


def test_a_fixture_with_no_gameweek_is_skipped():
    scores = [performance()]
    scores[0].fixture.gameweek = None
    result = score(PointsPrediction(expected_points=1.0), scores)
    assert result.points.n_observations == 0
    assert result.points.n_skipped == 1


def test_a_manager_is_skipped_rather_than_predicted():
    """
    Managers have performances and points in the database like anyone else.

    Nothing here models one - no involvement is fitted for the position, and no
    squad can contain one - so a manager is not an observation, and asking the
    model about one at all is the bug this guards against.
    """
    model = StubModel(PointsPrediction(expected_points=5.0))
    result = score_prediction_breakdown(
        model,
        [performance(points=9, position="MNG"), performance(points=2)],
        root_gameweek=1,
        season=SEASON,
        dbsession=None,
    )
    assert [request.player.position(SEASON) for request in model.asked] == ["MID"]
    assert result.points.n_observations == 1
    assert result.points.n_skipped == 1
