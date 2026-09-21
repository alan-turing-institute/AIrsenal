"""
Tests for how the optimiser handles points hits, discounting and bonus fixtures.
"""

import json

import pytest

from airsenal.framework.optimization_utils import get_discount_factor
from airsenal.framework.prediction_utils import (
    AVERAGE_TEAM_GOALS,
    bonus_fixture_factor,
    expected_goals_from_probs,
)
from airsenal.scripts import fill_transfersuggestion_table as fts


def write_strategy(output_dir, tag, sid, total_score, points_hit):
    strat = {
        "total_score": total_score,
        "points_hit": points_hit,
        "points_per_gw": dict.fromkeys(points_hit, 0),
    }
    (output_dir / f"strategy_{tag}_{sid}.json").write_text(json.dumps(strat))


@pytest.fixture
def output_dir(tmp_path, monkeypatch):
    out = tmp_path / "airsopt"
    out.mkdir()
    monkeypatch.setattr(fts, "OUTPUT_DIR", str(out))
    return out


def test_hit_taken_when_gain_is_large(output_dir):
    write_strategy(output_dir, "t", "0", total_score=50.0, points_hit={"1": 0})
    write_strategy(output_dir, "t", "2", total_score=56.0, points_hit={"1": 4})

    best = fts.find_best_strat_from_json("t", min_hit_gain=2.0)
    assert best["total_score"] == 56.0, "a 6 point gain clears the threshold"


def test_hit_rejected_when_gain_is_marginal(output_dir):
    """A hit that wins by 0.3pts is not evidence it is better than not taking it."""
    write_strategy(output_dir, "t", "0", total_score=50.0, points_hit={"1": 0})
    write_strategy(output_dir, "t", "2", total_score=50.3, points_hit={"1": 4})

    best = fts.find_best_strat_from_json("t", min_hit_gain=2.0)
    assert best["total_score"] == 50.0, "should fall back to the no-hit strategy"


def test_threshold_of_zero_keeps_old_behaviour(output_dir):
    write_strategy(output_dir, "t", "0", total_score=50.0, points_hit={"1": 0})
    write_strategy(output_dir, "t", "2", total_score=50.3, points_hit={"1": 4})

    best = fts.find_best_strat_from_json("t", min_hit_gain=0.0)
    assert best["total_score"] == 50.3


def test_threshold_does_not_affect_free_transfers(output_dir):
    """Free transfers cost nothing, so even a tiny gain is worth taking."""
    write_strategy(output_dir, "t", "0", total_score=50.0, points_hit={"1": 0})
    write_strategy(output_dir, "t", "1", total_score=50.1, points_hit={"1": 0})

    best = fts.find_best_strat_from_json("t", min_hit_gain=5.0)
    assert best["total_score"] == 50.1


def test_no_alternative_to_a_hit(output_dir):
    """If every strategy takes a hit there is nothing to fall back to."""
    write_strategy(output_dir, "t", "2", total_score=50.3, points_hit={"1": 4})
    best = fts.find_best_strat_from_json("t", min_hit_gain=5.0)
    assert best["total_score"] == 50.3


@pytest.mark.parametrize(
    ("discount", "n_ahead", "expected"),
    [
        (1.0, 3, 1.0),  # no discounting: every gameweek in the window counts equally
        (0.5, 1, 0.5),
        (0.5, 2, 0.25),
    ],
)
def test_discount_factor_is_configurable(discount, n_ahead, expected):
    assert get_discount_factor(1, 1 + n_ahead, discount=discount) == pytest.approx(
        expected
    )


def test_expected_goals_from_probs():
    assert expected_goals_from_probs({0: 0.5, 1: 0.3, 2: 0.2}) == pytest.approx(0.7)


def test_bonus_factor_favours_easier_fixtures():
    """A team expected to score more than average should earn more bonus."""
    easy = bonus_fixture_factor({0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4})
    hard = bonus_fixture_factor({0: 0.6, 1: 0.3, 2: 0.1, 3: 0.0})
    assert easy > 1.0
    assert hard < 1.0
    assert easy > hard


def test_bonus_factor_is_one_for_an_average_fixture():
    factor = bonus_fixture_factor({0: 0.0, 1: 0.0, AVERAGE_TEAM_GOALS: 1.0})
    assert factor == pytest.approx(1.0)


def test_bonus_factor_is_clipped():
    """The proxy is rough, so it must not swing the estimate wildly."""
    assert bonus_fixture_factor({0: 0.0, 10: 1.0}) == 1.5
    assert bonus_fixture_factor({0: 1.0}) == 0.5
