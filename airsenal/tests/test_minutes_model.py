"""
Tests for the expected-minutes model (airsenal.framework.minutes_model).
"""

import numpy as np
import pandas as pd

from airsenal.conftest import TEST_PAST_SEASON, session_scope
from airsenal.framework.minutes_model import (
    MinutesModel,
    count_absent_teammates,
    get_position_teammates,
)
from airsenal.framework.schema import Absence, Player, PlayerAttributes


def _make_player(ts, player_id, name, team, position, gameweek=1):
    """Create a Player + PlayerAttributes row and return the plain int player_id
    (not the ORM object - these get expired/detached across the commits below, and
    the id is all the tests need).
    """
    p = Player()
    p.player_id = player_id
    p.fpl_api_id = player_id
    p.name = name
    ts.add(p)
    pa = PlayerAttributes()
    pa.player_id = player_id
    pa.season = TEST_PAST_SEASON
    pa.gameweek = gameweek
    pa.price = 50
    pa.team = team
    pa.position = position
    pa.player = p
    ts.add(pa)
    return player_id


def test_get_position_teammates_excludes_other_teams_and_positions(fill_players):
    with session_scope() as ts:
        target_id = _make_player(ts, 90001, "Backup Striker", "ARS", "FWD")
        same_team_same_pos_id = _make_player(ts, 90002, "Rival Striker", "ARS", "FWD")
        same_team_diff_pos_id = _make_player(ts, 90003, "Team Defender", "ARS", "DEF")
        diff_team_same_pos_id = _make_player(ts, 90004, "Other Striker", "CHE", "FWD")
        ts.commit()

        target = ts.get(Player, target_id)
        teammates = get_position_teammates(target, TEST_PAST_SEASON, 1, dbsession=ts)
        teammate_ids = {p.player_id for p in teammates}

        assert same_team_same_pos_id in teammate_ids
        assert target_id not in teammate_ids
        assert same_team_diff_pos_id not in teammate_ids
        assert diff_team_same_pos_id not in teammate_ids


def test_count_absent_teammates_counts_only_absent_ones(fill_players):
    with session_scope() as ts:
        target_id = _make_player(ts, 90101, "Backup Striker", "MIN", "FWD")
        injured_rival_id = _make_player(ts, 90102, "Injured Rival", "MIN", "FWD")
        fit_rival_id = _make_player(ts, 90103, "Fit Rival", "MIN", "FWD")
        ts.commit()

        absence = Absence()
        absence.player_id = injured_rival_id
        absence.season = TEST_PAST_SEASON
        absence.reason = "injury"
        absence.date_from = "2021-08-01"
        absence.date_until = "2021-09-01"
        absence.gw_from = 0
        absence.gw_until = 3
        absence.timestamp = "2021-08-01"
        ts.add(absence)
        ts.commit()

        target = ts.get(Player, target_id)
        teammates = get_position_teammates(target, TEST_PAST_SEASON, 1, dbsession=ts)
        n_absent = count_absent_teammates(
            teammates, TEST_PAST_SEASON, current_gw=1, fixture_gw=1, dbsession=ts
        )

        assert {p.player_id for p in teammates} == {injured_rival_id, fit_rival_id}
        assert n_absent == 1


def test_minutes_model_predicts_more_when_competitor_absent():
    """Fit on synthetic data where teammate absence clearly increases minutes, and
    check the model actually picks up that direction - not just that it runs.
    """
    rng = np.random.default_rng(42)
    n = 500
    own_recent_minutes = rng.uniform(0, 90, size=n)
    n_teammates_absent = rng.integers(0, 3, size=n)
    positions = rng.choice(["GK", "DEF", "MID", "FWD"], size=n)
    noise = rng.normal(0, 5, size=n)
    minutes = np.clip(own_recent_minutes * 0.6 + n_teammates_absent * 20 + noise, 0, 90)
    df = pd.DataFrame(
        {
            "own_recent_minutes": own_recent_minutes,
            "n_teammates_absent": n_teammates_absent,
            "position": positions,
            "minutes": minutes,
        }
    )

    model = MinutesModel().fit(df)

    low = model.predict_one(
        own_recent_minutes=10.0, n_teammates_absent=0, position="MID"
    )
    high = model.predict_one(
        own_recent_minutes=10.0, n_teammates_absent=2, position="MID"
    )

    assert 0.0 <= low <= 90.0
    assert 0.0 <= high <= 90.0
    assert high > low
