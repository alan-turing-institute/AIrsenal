"""
The squad builder and the transfer optimiser have to value bench points the same
way, or the squad you start the season with is optimised for a season you then
never play.
"""

import sys
from unittest.mock import patch

from airsenal.framework.optimization_utils import DEFAULT_SUB_WEIGHTS
from airsenal.scripts import squad_builder


def run_main(argv, **fakes):
    """Run squad_builder.main far enough to capture the arguments it would pass."""
    captured = {}

    def fake_fill_initial_squad(**kwargs):
        captured.update(kwargs)

    with (
        patch.object(sys, "argv", ["airsenal_make_squad", *argv]),
        patch.object(squad_builder, "fill_initial_squad", fake_fill_initial_squad),
        patch.object(squad_builder, "get_latest_prediction_tag", lambda *a, **k: "tag"),
        patch.object(squad_builder, "check_tag_valid", lambda *a, **k: True),
        patch.object(squad_builder, "get_max_gameweek", lambda *a, **k: 38),
        patch.object(squad_builder, "NEXT_GAMEWEEK", 1),
        patch.object(squad_builder.fetcher, "FPL_TEAM_ID", 1234),
    ):
        squad_builder.main()
    return captured | fakes


def test_uses_the_shared_sub_weights():
    """These were hardcoded to a different, lower set than DEFAULT_SUB_WEIGHTS, so
    airsenal_make_squad and the transfer optimiser were maximising different
    things - and the squad the command stored was not the best one under either."""
    captured = run_main(["--season", "2627"])
    assert captured["sub_weights"] == DEFAULT_SUB_WEIGHTS


def test_no_subs_ignores_the_bench_entirely():
    captured = run_main(["--season", "2627", "--no_subs"])
    assert captured["sub_weights"] == {"GK": 0, "Outfield": (0, 0, 0)}


def test_seed_is_passed_through():
    captured = run_main(["--season", "2627", "--seed", "99"])
    assert captured["random_state"] == 99


def test_no_seed_means_no_random_state():
    captured = run_main(["--season", "2627"])
    assert captured["random_state"] is None
