"""
Replaying a past season, which nothing tested until now.

`airsenal replay` is the only command that scores a configuration over real
results, so it is what an experiment is judged with - and it had no test at all.
These run it over two gameweeks of the seeded database, with a tiny squad search
and a stub transfer search, so the whole thing runs in seconds and forks nothing.
"""

import json
from typing import ClassVar

import pytest

from airsenal.db.queries.gameweeks import reset_gameweek_cache, set_next_gameweek
from airsenal.game.enums import Chip
from airsenal.optimization.moves import GameweekMove
from airsenal.optimization.plan import GameweekOutcome, Plan, TransferSearchResult
from airsenal.optimization.squad_optimizers import (
    GeneticAlgorithmConfig,
    GeneticSquadOptimizer,
)
from airsenal.pipeline import (
    AIrsenalPipeline,
    PipelineSettings,
    ReplayResult,
    ReplaySettings,
    replay_season,
    run_replays,
)
from airsenal.pipeline.replay import _gameweek_outcome
from airsenal.prediction.player_models import build_player_model
from airsenal.prediction.team_models import build_team_model
from tests.e2e.conftest import GAMEWEEKS_PER_PAST_SEASON, PAST_SEASONS

REPLAY_SEASON = PAST_SEASONS[-1]
TEAM_ID = -1
FIRST_GAMEWEEK = 1
LAST_GAMEWEEK = 2


class NoTransferOptimizer:
    """
    Satisfies TransferOptimizer by proposing nothing.

    A real search forks, and the replay driver is what these tests are about,
    not the search. Keeping the starting squad makes every gameweek after the
    first a plan with no moves, which is still a plan.
    """

    def search(self, request):
        plan = Plan(
            root_gameweek=request.gameweeks[0],
            outcomes=tuple(
                GameweekOutcome(
                    gameweek=gw,
                    move=GameweekMove(),
                    points=0.0,
                    discount_factor=1.0,
                    points_hit=0,
                    free_transfers=request.num_free_transfers,
                )
                for gw in request.gameweeks
            ),
        )
        return TransferSearchResult(best=plan, baseline=plan)


@pytest.fixture(scope="module")
def seeded(pipeline_db):
    set_next_gameweek(FIRST_GAMEWEEK)
    yield pipeline_db
    reset_gameweek_cache()
    set_next_gameweek(1)


def _pipeline(**settings):
    return AIrsenalPipeline(
        team_model=build_team_model("constant"),
        player_model=build_player_model("constant"),
        squad_optimizer=GeneticSquadOptimizer(
            GeneticAlgorithmConfig(population_size=20, generations=5, random_state=0)
        ),
        transfer_optimizer=NoTransferOptimizer(),
        settings=PipelineSettings(
            **{
                "fpl_team_id": TEAM_ID,
                "season": REPLAY_SEASON,
                "n_gameweeks": 2,
                "refresh_database": False,
                "apply_transfers": False,
                "save_absences": False,
                **settings,
            }
        ),
    )


@pytest.fixture(scope="module")
def replayed(seeded, tmp_path_factory):
    out = tmp_path_factory.mktemp("replay_out")
    result = replay_season(
        _pipeline(),
        ReplaySettings(
            gameweek_start=FIRST_GAMEWEEK,
            gameweek_end=LAST_GAMEWEEK,
            tag_prefix="test_replay",
            output_dir=out,
        ),
    )
    return result, out


def test_replay_returns_a_result(replayed):
    result, _ = replayed
    assert result.season == REPLAY_SEASON
    assert [gw.gameweek for gw in result.gameweeks] == [FIRST_GAMEWEEK, LAST_GAMEWEEK]


def test_every_gameweek_picks_a_full_team(replayed):
    result, _ = replayed
    for gw in result.gameweeks:
        assert len(gw.starting_11) == 11
        assert len(gw.subs) == 4
        assert gw.captain is not None
        assert gw.vice_captain != gw.captain


def test_totals_agree_with_the_gameweeks(replayed):
    result, _ = replayed
    assert result.total_points == sum(gw.actual_points for gw in result.gameweeks)
    assert result.total_points_hit == sum(gw.points_hit for gw in result.gameweeks)
    assert result.mean_absolute_error >= 0


def test_the_result_says_what_produced_it(replayed):
    """Two replays are only comparable if each records its own configuration."""
    result, _ = replayed
    assert result.config["team_model"] == "ConstantTeamModel"
    assert result.config["player_model"] == "ConstantPlayerModel"
    assert result.config["transfer_optimizer"] == "NoTransferOptimizer"


def test_it_writes_where_it_was_told(replayed):
    result, out = replayed
    path = out / "test_replay.json"
    assert path.exists()
    written = json.loads(path.read_text())
    assert written["total_points"] == result.total_points
    assert written["config"] == result.config
    # the per-gameweek keys the replay plotting notebook reads
    assert set(written["gameweeks"][0]) >= {
        "gameweek",
        "starting_11",
        "subs",
        "captain",
    }


def test_mean_absolute_error_is_zero_for_no_gameweeks():
    """The summary of an empty replay is a number, not a ZeroDivisionError."""
    empty = ReplayResult(tag="t", season=REPLAY_SEASON, n_gameweeks=None, config={})
    assert empty.mean_absolute_error == 0.0
    assert empty.total_points == 0


def test_run_replays_returns_one_result_per_loop(seeded, tmp_path_factory):
    out = tmp_path_factory.mktemp("replay_loop")
    results = run_replays(
        _pipeline(),
        ReplaySettings(
            gameweek_start=GAMEWEEKS_PER_PAST_SEASON,
            gameweek_end=GAMEWEEKS_PER_PAST_SEASON,
            output_dir=out,
            loop=2,
        ),
    )
    assert len(results) == 2
    assert len(list(out.glob("*.json"))) == len(results)


# ----------------------------------------------------------------- chips ---


class ChipRecordingSquad:
    """Records the chip flags it was scored with, and nothing else."""

    players: ClassVar[list] = []

    def __init__(self):
        self.expected_calls = []
        self.actual_calls = []

    def get_expected_points(
        self, _tag, _gameweek, bench_boost=False, triple_captain=False
    ):
        self.expected_calls.append((bench_boost, triple_captain))
        return 0.0

    def get_actual_points(
        self, _gameweek, _season, triple_captain=False, bench_boost=False
    ):
        self.actual_calls.append((bench_boost, triple_captain))
        return 0


def _plan_playing(chip):
    return Plan(
        root_gameweek=1,
        outcomes=(
            GameweekOutcome(
                gameweek=1,
                move=GameweekMove(chip=chip),
                points=0.0,
                discount_factor=1.0,
                points_hit=0,
                free_transfers=1,
            ),
        ),
    )


@pytest.mark.parametrize(
    ("chip", "expected_flags"),
    [
        (Chip.BENCH_BOOST, (True, False)),
        (Chip.TRIPLE_CAPTAIN, (False, True)),
        (Chip.WILDCARD, (False, False)),
        (None, (False, False)),
    ],
)
def test_a_replayed_gameweek_is_scored_with_the_chip_it_played(chip, expected_flags):
    """
    Both point totals are scored with the chip, not without it.

    A bench boost scores the bench and a triple captain trebles, so scoring a
    chip week as though no chip were played understates exactly the weeks the
    chip was meant to win - and `total_points` is what two replays are compared on.
    """
    squad = ChipRecordingSquad()
    row = _gameweek_outcome("tag", 1, squad, _plan_playing(chip), "2526")

    assert squad.expected_calls == [expected_flags]
    assert squad.actual_calls == [expected_flags]
    assert row.chip_played == (str(chip) if chip else None)


def test_a_gameweek_with_no_plan_is_scored_without_chips():
    """A squad built from scratch has no plan to read a chip off."""
    squad = ChipRecordingSquad()
    row = _gameweek_outcome("tag", 1, squad, None, "2526")

    assert squad.actual_calls == [(False, False)]
    assert row.chip_played is None
