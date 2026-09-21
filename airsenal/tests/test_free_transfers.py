"""
Tests for estimating how many free transfers a team has.

Since 2024/25 up to 5 free transfers can be saved up; the estimates used when the
logged-in API call fails previously capped out at 2.
"""

import pytest
from curl_cffi import requests

from airsenal.framework.FPL_scoring_rules import MAX_FREE_TRANSFERS
from airsenal.framework.utils import get_free_transfers


class DummyFetcher:
    """Stands in for FPLDataFetcher: the logged-in call fails, so we fall back to
    estimating from public transfer history."""

    FPL_TEAM_ID = 1

    def __init__(self, transfers_per_gw):
        self.transfers_per_gw = transfers_per_gw

    def get_num_free_transfers(self, *args, **kwargs):
        msg = "not logged in"
        raise requests.exceptions.RequestException(msg)

    def get_fpl_team_history_data(self, *args, **kwargs):
        return {
            "current": [
                {"event": gw, "event_transfers": n}
                for gw, n in self.transfers_per_gw.items()
            ]
        }


@pytest.fixture(autouse=True)
def _patch_start_gameweek(monkeypatch):
    """The team started in gameweek 1, so every gameweek in the history counts."""
    monkeypatch.setattr(
        "airsenal.framework.utils.get_entry_start_gameweek", lambda *a, **k: 1
    )


@pytest.mark.parametrize(
    ("transfers_per_gw", "gameweek", "expected"),
    [
        # no transfers made: one earned per week, accumulating
        ({2: 0, 3: 0}, 4, 3),
        # five quiet gameweeks would earn 6, but the cap is 5
        ({2: 0, 3: 0, 4: 0, 5: 0, 6: 0}, 7, MAX_FREE_TRANSFERS),
        # bank three, spend one, keep the rest
        ({2: 0, 3: 0, 4: 1}, 5, 3),
        # spending everything leaves the minimum of 1
        ({2: 0, 3: 0, 4: 4}, 5, 1),
        # a hit (more transfers than available) still leaves 1
        ({2: 3}, 3, 1),
    ],
)
def test_free_transfers_accumulate_to_the_cap(transfers_per_gw, gameweek, expected):
    with pytest.warns(UserWarning, match="Failed to get actual free transfers"):
        num = get_free_transfers(
            fpl_team_id=1,
            gameweek=gameweek,
            apifetcher=DummyFetcher(transfers_per_gw),
        )
    assert num == expected


def test_free_transfers_never_exceeds_max():
    """Whatever the history, we can never have more than the game allows."""
    quiet_season = dict.fromkeys(range(2, 20), 0)
    with pytest.warns(UserWarning, match="Failed to get actual free transfers"):
        num = get_free_transfers(
            fpl_team_id=1,
            gameweek=20,
            apifetcher=DummyFetcher(quiet_season),
        )
    assert num == MAX_FREE_TRANSFERS
