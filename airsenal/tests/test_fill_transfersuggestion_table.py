"""
Tests for airsenal.scripts.fill_transfersuggestion_table.
"""

from unittest.mock import patch

import pytest

from airsenal.scripts.fill_transfersuggestion_table import (
    _detect_available_chip_gameweeks,
)


def test_detect_available_chip_gameweeks_replay_assumes_fresh_allocation():
    # no live API access during replay - assume every chip is unused so far, and
    # rely on next_week_transfers' half-aware reuse check to cap actual usage.
    with patch(
        "airsenal.scripts.fill_transfersuggestion_table.fetcher.get_available_chips"
    ) as mock_get_available_chips:
        result = _detect_available_chip_gameweeks(fpl_team_id=123, use_api=False)

    mock_get_available_chips.assert_not_called()
    assert result == {
        "wildcard": 0,
        "free_hit": 0,
        "bench_boost": 0,
        "triple_captain": 0,
    }


def test_detect_available_chip_gameweeks_live_maps_api_names():
    with patch(
        "airsenal.scripts.fill_transfersuggestion_table.fetcher.get_available_chips",
        return_value=["wildcard", "bboost"],
    ) as mock_get_available_chips:
        result = _detect_available_chip_gameweeks(fpl_team_id=123, use_api=True)

    mock_get_available_chips.assert_called_once_with(123)
    assert result == {
        "wildcard": 0,
        "free_hit": -1,
        "bench_boost": 0,
        "triple_captain": -1,
    }


def test_detect_available_chip_gameweeks_unrecognised_api_name_raises():
    with (
        patch(
            "airsenal.scripts.fill_transfersuggestion_table.fetcher.get_available_chips",
            return_value=["mystery_chip"],
        ),
        pytest.raises(RuntimeError, match="mystery_chip"),
    ):
        _detect_available_chip_gameweeks(fpl_team_id=123, use_api=True)
