"""
Swapping a component, which is what the pipeline object exists to allow.

These run `AIrsenalPipeline.run()` itself - nothing else does - with the
optimizers replaced by recorders that satisfy the protocols but are unknown to
any registry. If a component can only be one of the things the registry ships,
these tests cannot be written.

`refresh_database=False` and `new_squad=True` are what let `run()` execute with
no network call at all; they are the reason those two settings exist.
"""

import pytest
from sqlalchemy import select

from airsenal.db.models import PlayerPrediction
from airsenal.db.queries.gameweeks import reset_gameweek_cache, set_next_gameweek
from airsenal.db.queries.predictions import get_predicted_points
from airsenal.optimization.squad_optimizers import (
    SQUAD_OPTIMIZERS,
    GeneticAlgorithmConfig,
    GeneticSquadOptimizer,
)
from airsenal.pipeline import AIrsenalPipeline, PipelineSettings
from airsenal.prediction.player_models import (
    build_player_model,
)
from airsenal.prediction.team_models import (
    build_team_model,
)
from tests.e2e.conftest import FUTURE_GAMEWEEKS, SEASON

TEAM_ID = -1


@pytest.fixture(scope="module")
def seeded(pipeline_db):
    set_next_gameweek(FUTURE_GAMEWEEKS[0])
    yield pipeline_db
    reset_gameweek_cache()
    set_next_gameweek(1)


class RecordingSquadOptimizer:
    """Satisfies SquadOptimizer, records what it was asked, builds a real squad."""

    def __init__(self):
        self.requests = []
        self._real = GeneticSquadOptimizer(
            GeneticAlgorithmConfig(population_size=20, generations=5, random_state=0)
        )

    def optimize(self, request):
        self.requests.append(request)
        return self._real.optimize(request)


class RecordingTransferOptimizer:
    """Satisfies TransferOptimizer. Never expected to be called in these tests."""

    def __init__(self):
        self.requests = []

    def search(self, request):
        self.requests.append(request)
        msg = "the transfer optimizer should not have been reached"
        raise AssertionError(msg)


def _pipeline(team_model="constant", player_model="constant", **settings):
    return AIrsenalPipeline(
        team_model=build_team_model(team_model),
        player_model=build_player_model(player_model),
        squad_optimizer=RecordingSquadOptimizer(),
        transfer_optimizer=RecordingTransferOptimizer(),
        settings=PipelineSettings(
            fpl_team_id=TEAM_ID,
            season=SEASON,
            n_gameweeks=len(FUTURE_GAMEWEEKS),
            gameweek_start=FUTURE_GAMEWEEKS[0],
            new_squad=True,
            refresh_database=False,
            apply_transfers=False,
            save_absences=False,
            **settings,
        ),
    )


@pytest.fixture(scope="module")
def completed(seeded):
    pipeline = _pipeline()
    pipeline.run()
    return pipeline


def test_run_reaches_the_squad_optimizer_it_was_given(completed):
    assert len(completed.squad_optimizer.requests) == 1


def test_run_does_not_reach_the_transfer_optimizer_for_a_new_squad(completed):
    assert completed.transfer_optimizer.requests == []


def test_the_optimizer_is_asked_for_the_gameweeks_the_run_covers(completed):
    assert completed.squad_optimizer.requests[0].gameweeks == FUTURE_GAMEWEEKS


def test_the_optimizer_is_handed_the_tag_prediction_actually_wrote(completed, seeded):
    """
    The tag is threaded through rather than looked up again afterwards.

    Checked by membership, not equality: the database is shared with the other
    e2e modules, so it holds their tags too.
    """
    tag = completed.squad_optimizer.requests[0].tag
    written = set(seeded.scalars(select(PlayerPrediction.tag)).all())
    assert tag in written
    predictions = seeded.scalars(
        select(PlayerPrediction).where(PlayerPrediction.tag == tag)
    ).all()
    assert predictions


def test_the_optimizer_is_told_which_season(completed):
    assert completed.squad_optimizer.requests[0].season == SEASON


def test_swapping_the_team_model_changes_the_predictions(seeded):
    """
    Proves the component object is used, not a name resolved somewhere downstream.

    Two runs differing only in their team model must not produce identical
    predicted points.
    """
    constant = _pipeline(team_model="constant")
    constant.run()
    random_model = _pipeline(team_model="random")
    random_model.run()

    tags = [
        constant.squad_optimizer.requests[0].tag,
        random_model.squad_optimizer.requests[0].tag,
    ]
    assert tags[0] != tags[1]

    points = [
        {
            player.player_id: score
            for player, score in get_predicted_points(
                FUTURE_GAMEWEEKS, tag, season=SEASON, dbsession=seeded
            )
        }
        for tag in tags
    ]
    assert points[0]
    assert points[1]
    assert points[0] != points[1]


def test_a_pipeline_can_be_rebuilt_with_different_settings():
    pipeline = _pipeline()
    changed = pipeline.with_settings(n_gameweeks=9)

    assert changed.settings.n_gameweeks == 9
    assert pipeline.settings.n_gameweeks == len(FUTURE_GAMEWEEKS)
    # the components come along unchanged
    assert changed.squad_optimizer is pipeline.squad_optimizer


def test_a_component_the_tables_do_not_know_about_still_works():
    """
    The point of the whole seam: a squad optimizer defined here, registered
    nowhere, is a first-class component of the pipeline.
    """
    optimizer = RecordingSquadOptimizer()
    pipeline = AIrsenalPipeline(squad_optimizer=optimizer)

    assert pipeline.squad_optimizer is optimizer
    assert not any(optimizer is entry for entry in SQUAD_OPTIMIZERS.values())
