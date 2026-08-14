"""
Tests for the expected-minutes model (airsenal.framework.minutes_model).
"""

import numpy as np
import pandas as pd

from airsenal.conftest import TEST_PAST_SEASON, session_scope
from airsenal.framework.minutes_model import (
    MinutesModel,
    get_position_teammates,
    get_teammate_typical_minutes,
    sum_absent_teammates_typical_minutes,
)
from airsenal.framework.schema import (
    Absence,
    Fixture,
    Player,
    PlayerAttributes,
    PlayerScore,
    Result,
)


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


def _make_appearance(ts, player_id, team, gameweek, minutes):
    """Create a Fixture + Result + PlayerScore for one gameweek's appearance."""
    fixture = Fixture()
    fixture.date = f"2021-08-{gameweek:02d}"
    fixture.gameweek = gameweek
    fixture.home_team = team
    fixture.away_team = "OPP"
    fixture.season = TEST_PAST_SEASON
    fixture.tag = "test"
    ts.add(fixture)
    ts.flush()

    result = Result()
    result.fixture_id = fixture.fixture_id
    result.home_score = 1
    result.away_score = 1
    ts.add(result)
    ts.flush()

    score = PlayerScore()
    score.player_id = player_id
    score.player_team = team
    score.opponent = "OPP"
    score.points = 0
    score.goals = 0
    score.assists = 0
    score.bonus = 0
    score.conceded = 0
    score.minutes = minutes
    score.result_id = result.result_id
    score.fixture_id = fixture.fixture_id
    ts.add(score)


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


def test_sum_absent_teammates_typical_minutes_weights_by_importance(fill_players):
    with session_scope() as ts:
        target_id = _make_player(ts, 90101, "Backup Striker", "MIN", "FWD", gameweek=6)
        starter_id = _make_player(
            ts, 90102, "Regular Starter", "MIN", "FWD", gameweek=6
        )
        bench_id = _make_player(ts, 90103, "Bench Player", "MIN", "FWD", gameweek=6)

        starter_minutes = [85, 80, 90, 88, 82]
        for gw, mins in enumerate(starter_minutes, start=1):
            _make_appearance(ts, starter_id, "MIN", gw, mins)
        bench_minutes = [5, 0, 10, 0, 3]
        for gw, mins in enumerate(bench_minutes, start=1):
            _make_appearance(ts, bench_id, "MIN", gw, mins)
        ts.commit()

        for player_id in (starter_id, bench_id):
            absence = Absence()
            absence.player_id = player_id
            absence.season = TEST_PAST_SEASON
            absence.reason = "injury"
            absence.date_from = "2021-09-01"
            absence.date_until = "2021-09-15"
            absence.gw_from = 5
            absence.gw_until = 8
            absence.timestamp = "2021-09-01"
            ts.add(absence)
        ts.commit()

        starter = ts.get(Player, starter_id)
        bench = ts.get(Player, bench_id)

        starter_importance = get_teammate_typical_minutes(
            starter, TEST_PAST_SEASON, current_gw=6, dbsession=ts
        )
        bench_importance = get_teammate_typical_minutes(
            bench, TEST_PAST_SEASON, current_gw=6, dbsession=ts
        )
        # bench player's zero-minute rows should be skipped, not averaged in
        assert bench_importance == np.mean([5, 10, 3])
        assert starter_importance == np.mean(starter_minutes)
        assert starter_importance > bench_importance

        target = ts.get(Player, target_id)
        teammates = get_position_teammates(target, TEST_PAST_SEASON, 6, dbsession=ts)
        total = sum_absent_teammates_typical_minutes(
            teammates, TEST_PAST_SEASON, current_gw=6, fixture_gw=6, dbsession=ts
        )
        assert total == starter_importance + bench_importance


def test_minutes_model_predicts_more_when_competitor_absent():
    """Fit on synthetic data where an absent competitor's own typical minutes
    clearly increases the target player's minutes, and check the model actually
    picks up that direction and magnitude - not just that it runs.
    """
    rng = np.random.default_rng(42)
    n = 1500
    own_recent_minutes = rng.uniform(0, 90, size=n)
    absent_teammates_typical_minutes = rng.uniform(0, 90, size=n)
    positions = rng.choice(["GK", "DEF", "MID", "FWD"], size=n)
    noise = rng.normal(0, 15, size=n)
    minutes = np.clip(
        own_recent_minutes * 0.5 + absent_teammates_typical_minutes * 0.3 + noise,
        0,
        90,
    )
    df = pd.DataFrame(
        {
            "own_recent_minutes": own_recent_minutes,
            "absent_teammates_typical_minutes": absent_teammates_typical_minutes,
            "position": positions,
            "minutes": minutes,
        }
    )

    model = MinutesModel().fit(df)

    low = model.predict_one(
        own_recent_minutes=40.0, absent_teammates_typical_minutes=5.0, position="MID"
    )
    high = model.predict_one(
        own_recent_minutes=40.0, absent_teammates_typical_minutes=80.0, position="MID"
    )

    assert 0.0 <= low <= 90.0
    assert 0.0 <= high <= 90.0
    assert high > low
