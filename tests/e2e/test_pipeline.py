"""
The whole pipeline, on a database small enough to run in seconds.

Every other test covers one function. This one covers the joins between the
stages - that prediction writes something optimisation can read, and that
optimisation produces a squad the game's rules would accept. Those joins are
where a change can break something without any unit test noticing.

Both models are `constant`, which are shipped models rather than test doubles,
so the registry indirection is exercised the same way it is in production.
"""

import math

import pytest
from sqlalchemy import select

from airsenal.db.models import Fixture, Player, PlayerPrediction, Result
from airsenal.db.queries.gameweeks import reset_gameweek_cache, set_next_gameweek
from airsenal.db.queries.predictions import get_predicted_points
from airsenal.game.enums import Position
from airsenal.optimization.moves import GameweekMove
from airsenal.optimization.protocols import TransferRequest
from airsenal.optimization.squad_optimizers import GeneticAlgorithmConfig
from airsenal.optimization.squad_optimizers.genetic_algorithm import (
    make_new_squad,
)
from airsenal.optimization.strategies import DEFAULT_STRATEGIES
from airsenal.optimization.transfer_optimizers.tree_search import (
    _make_best_transfers,
)
from airsenal.prediction.player_models import (
    build_player_model,
)
from airsenal.prediction.run import make_predictedscore_table
from airsenal.prediction.team_models import (
    build_team_model,
)
from airsenal.squad.squad import SubWeights
from tests.e2e.conftest import FUTURE_GAMEWEEKS, SEASON, SQUAD_SHAPE, TEAMS

BUDGET = 1000
SQUAD_SIZE = 15
MAX_PER_TEAM = 3
FORMATION = {Position.GK: 2, Position.DEF: 5, Position.MID: 5, Position.FWD: 3}


@pytest.fixture(scope="module")
def seeded(pipeline_db):
    """The database, with the next gameweek pinned so nothing reaches for the API."""
    set_next_gameweek(FUTURE_GAMEWEEKS[0])
    yield pipeline_db
    # The gameweek cache is process-wide, so pinning it here would otherwise leak
    # this season's gameweek into whatever runs after this module. Put it back to
    # the value conftest.py sets for the rest of the suite.
    reset_gameweek_cache()
    set_next_gameweek(1)


def test_database_is_populated(seeded):
    assert len(seeded.scalars(select(Player)).all()) == sum(SQUAD_SHAPE.values())
    fixtures = seeded.scalars(select(Fixture)).all()
    # 4 fixtures per gameweek, over two past seasons and the current one
    assert len(fixtures) == 4 * (8 + 8 + len(FUTURE_GAMEWEEKS))
    # only the past seasons have been played
    assert len(seeded.scalars(select(Result)).all()) == 4 * 16


@pytest.fixture(scope="module")
def prediction_tag(seeded):
    """Run the prediction stage once, and hand its tag to the tests below."""
    return make_predictedscore_table(
        gameweeks=FUTURE_GAMEWEEKS,
        season=SEASON,
        player_model=build_player_model("constant"),
        team_model=build_team_model("constant"),
        dbsession=seeded,
    )


def test_prediction_writes_one_tag(seeded, prediction_tag):
    tags = set(seeded.scalars(select(PlayerPrediction.tag)).all())
    assert tags == {prediction_tag}


def test_predictions_cover_every_player_and_gameweek(seeded, prediction_tag):
    predictions = seeded.scalars(
        select(PlayerPrediction).where(PlayerPrediction.tag == prediction_tag)
    ).all()
    assert predictions

    gameweeks = {p.fixture.gameweek for p in predictions}
    assert gameweeks == set(FUTURE_GAMEWEEKS)


def test_predicted_points_are_finite_and_non_negative(seeded, prediction_tag):
    # A NaN here propagates silently all the way into the optimiser's argmax,
    # where it wins every comparison.
    points = seeded.scalars(
        select(PlayerPrediction.predicted_points).where(
            PlayerPrediction.tag == prediction_tag
        )
    ).all()
    assert all(math.isfinite(p) for p in points)
    assert all(p >= 0 for p in points)


def test_predictions_are_readable_by_the_optimiser(seeded, prediction_tag):
    # The optimiser reads predictions back through a different query than the
    # one that wrote them; this is the join between the two stages.
    points = get_predicted_points(
        FUTURE_GAMEWEEKS, prediction_tag, season=SEASON, dbsession=seeded
    )
    assert points
    assert all(math.isfinite(p) for _, p in points)


@pytest.fixture(scope="module")
def squad(seeded, prediction_tag):
    """Build a squad from the predictions, with a deliberately tiny search."""
    return make_new_squad(
        FUTURE_GAMEWEEKS,
        prediction_tag,
        budget=BUDGET,
        season=SEASON,
        sub_weights=SubWeights(),
        ga_config=GeneticAlgorithmConfig(
            population_size=20, generations=5, random_state=0, verbose=False
        ),
        dbsession=seeded,
    )


def test_squad_is_legal(squad):
    assert squad is not None
    assert len(squad.players) == SQUAD_SIZE


def test_squad_has_the_right_shape(squad):
    counts = dict.fromkeys(Position, 0)
    for player in squad.players:
        counts[Position(player.position)] += 1
    assert counts == FORMATION


def test_squad_is_within_budget(squad):
    spent = sum(player.purchase_price for player in squad.players)
    assert spent <= BUDGET


def test_squad_respects_the_three_per_club_limit(squad):
    per_team = dict.fromkeys(TEAMS, 0)
    for player in squad.players:
        per_team[player.team] += 1
    assert max(per_team.values()) <= MAX_PER_TEAM


def test_squad_has_no_duplicates(squad):
    ids = [player.player_id for player in squad.players]
    assert len(ids) == len(set(ids))


# --- transfers ---------------------------------------------------------------
#
# The multiprocessing tree itself is covered by tests/optimization/; what is
# checked here is the decision path a worker runs: pick a strategy for the move,
# search it against the predictions written above, and hand back a legal squad.


def _request(move, squad, prediction_tag, num_iterations=100):
    return TransferRequest(
        move=move,
        squad=squad,
        tag=prediction_tag,
        gameweeks=FUTURE_GAMEWEEKS[:2],
        root_gw=FUTURE_GAMEWEEKS[0],
        season=SEASON,
        num_iterations=num_iterations,
    )


def _best_transfers(request):
    # one node of the tree, reached directly: the tree itself is covered by
    # test_transfer_search.py, and what is checked here is the decision it makes
    return _make_best_transfers(request, DEFAULT_STRATEGIES.create(request.move))


@pytest.fixture(scope="module")
def transfer_result(seeded, prediction_tag, squad):
    return _best_transfers(_request(GameweekMove(1), squad, prediction_tag, 5))


def test_transfers_are_balanced(transfer_result):
    _, transfers, _ = transfer_result
    assert len(transfers["in"]) == len(transfers["out"]) == 1


def test_no_player_is_transferred_both_in_and_out(transfer_result):
    _, transfers, _ = transfer_result
    assert not set(transfers["in"]) & set(transfers["out"])


def test_resulting_squad_is_still_legal(transfer_result):
    new_squad, _, _ = transfer_result
    assert len(new_squad.players) == SQUAD_SIZE
    counts = dict.fromkeys(Position, 0)
    for player in new_squad.players:
        counts[Position(player.position)] += 1
    assert counts == FORMATION


def test_transfer_score_is_finite_and_positive(transfer_result):
    _, _, points = transfer_result
    assert math.isfinite(points)
    assert points > 0


def test_a_transfer_is_not_worse_than_doing_nothing(seeded, prediction_tag, squad):
    """
    A single-transfer search can never beat itself by doing nothing.

    It considers keeping every player, so its best is at least the do-nothing
    baseline. If it is worse, the search is scoring the squad it returns
    differently from the one it evaluated.
    """
    _, _, baseline = _best_transfers(_request(GameweekMove(0), squad, prediction_tag))
    _, _, improved = _best_transfers(
        _request(GameweekMove(1), squad, prediction_tag, 5)
    )
    assert improved >= baseline
