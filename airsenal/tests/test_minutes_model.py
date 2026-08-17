"""
Tests for the expected-minutes model (airsenal.framework.minutes_model).
"""

import numpy as np
import pandas as pd

from airsenal.conftest import TEST_PAST_SEASON, session_scope
from airsenal.framework.minutes_model import (
    MinutesModel,
    _compute_replacement_uplift_table,
    _replacement_uplift_lookup_dict,
    _weighted_uplift_before_season,
    get_absent_teammates,
    get_max_replacement_uplift,
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
        absent_teammates = get_absent_teammates(
            teammates, TEST_PAST_SEASON, current_gw=6, fixture_gw=6, dbsession=ts
        )
        assert {p.player_id for p in absent_teammates} == {starter_id, bench_id}

        total = sum_absent_teammates_typical_minutes(
            absent_teammates, TEST_PAST_SEASON, current_gw=6, dbsession=ts
        )
        assert total == starter_importance + bench_importance


def _history_row(player_id, season, gameweek, minutes, team, position, absent=False):
    return {
        "player_id": player_id,
        "season": season,
        "gameweek": gameweek,
        "minutes": minutes,
        "team": team,
        "position": position,
        "absence_reason": "injury" if absent else None,
    }


def test_replacement_uplift_favours_true_backup_over_unrelated_teammate():
    """_compute_replacement_uplift_table should identify B as A's replacement (B's
    minutes jump specifically when A is out) while C - who plays a steady amount
    regardless of A's status - should show ~no uplift, even though both are same-team,
    same-position teammates of A. Built directly from a synthetic history frame (no DB
    needed) since _compute_replacement_uplift_table is a pure dataframe transform.
    """
    a_id, b_id, c_id = 201, 202, 203
    team, position, season = "MIN", "FWD", TEST_PAST_SEASON
    a_absent_gws = {6, 7, 8}

    rows = []
    for gw in range(1, 11):
        a_absent = gw in a_absent_gws
        rows.append(
            _history_row(
                a_id, season, gw, 0 if a_absent else 90, team, position, a_absent
            )
        )
        rows.append(
            _history_row(b_id, season, gw, 90 if a_absent else 0, team, position)
        )
        rows.append(_history_row(c_id, season, gw, 45, team, position))
    history = pd.DataFrame(rows)

    table = _compute_replacement_uplift_table(history)
    a_rows = table[table["player_a"] == a_id].set_index("player_b")

    assert a_rows.loc[b_id, "uplift"] == 90.0
    assert a_rows.loc[c_id, "uplift"] == 0.0
    assert a_rows.loc[b_id, "n_absent_gws"] == 3

    lookup = _replacement_uplift_lookup_dict(table)
    # only strictly-prior-season evidence should count - same/future season excluded
    # to avoid leaking within-season future gameweeks into an earlier prediction.
    entries = lookup[(team, position, a_id, b_id)]
    assert _weighted_uplift_before_season(entries, "2122") == 90.0
    assert np.isnan(_weighted_uplift_before_season(entries, season))
    assert np.isnan(_weighted_uplift_before_season(entries, "1920"))


def test_get_max_replacement_uplift_prefers_true_backup(fill_players):
    with session_scope() as ts:
        target_id = _make_player(ts, 90201, "Direct Backup", "MIN", "FWD", gameweek=6)
        starter_id = _make_player(
            ts, 90202, "Regular Starter", "MIN", "FWD", gameweek=6
        )
        unrelated_id = _make_player(
            ts, 90203, "Unrelated Teammate", "MIN", "FWD", gameweek=6
        )
        ts.commit()

        starter = ts.get(Player, starter_id)
        target = ts.get(Player, target_id)

        # uplift lookup keyed on a season strictly before TEST_PAST_SEASON, so it's
        # usable when predicting for TEST_PAST_SEASON (see
        # _weighted_uplift_before_season's season < before_season restriction).
        prior_season = "1920"
        lookup = {
            (
                "MIN",
                "FWD",
                starter_id,
                target_id,
            ): [(prior_season, 65.0, 4)],
            (
                "MIN",
                "FWD",
                starter_id,
                unrelated_id,
            ): [(prior_season, 2.0, 4)],
        }

        uplift = get_max_replacement_uplift(
            target, [starter], lookup, "MIN", "FWD", TEST_PAST_SEASON
        )
        assert uplift == 65.0

        # no absent teammates -> no signal
        assert np.isnan(
            get_max_replacement_uplift(
                target, [], lookup, "MIN", "FWD", TEST_PAST_SEASON
            )
        )
        # unrelated player's own absence shouldn't matter to target's uplift lookup
        unrelated = ts.get(Player, unrelated_id)
        assert np.isnan(
            get_max_replacement_uplift(
                target, [unrelated], lookup, "MIN", "FWD", TEST_PAST_SEASON
            )
        )


def test_minutes_model_predicts_more_when_competitor_absent():
    """Fit on synthetic data where an absent competitor's own typical minutes and
    replacement uplift clearly increase the target player's minutes, and check the
    model actually picks up that direction and magnitude - not just that it runs.
    max_replacement_uplift is NaN for a chunk of rows (as it is for real data) to
    confirm the model handles the sparse feature without special-casing it.
    """
    rng = np.random.default_rng(42)
    n = 1500
    own_recent_minutes = rng.uniform(0, 90, size=n)
    absent_teammates_typical_minutes = rng.uniform(0, 90, size=n)
    max_replacement_uplift = rng.uniform(0, 60, size=n)
    has_uplift_signal = rng.random(n) > 0.5
    max_replacement_uplift = np.where(has_uplift_signal, max_replacement_uplift, np.nan)
    positions = rng.choice(["GK", "DEF", "MID", "FWD"], size=n)
    noise = rng.normal(0, 15, size=n)
    uplift_contribution = np.nan_to_num(max_replacement_uplift, nan=0.0)
    minutes = np.clip(
        own_recent_minutes * 0.4
        + absent_teammates_typical_minutes * 0.2
        + uplift_contribution * 0.4
        + noise,
        0,
        90,
    )
    df = pd.DataFrame(
        {
            "own_recent_minutes": own_recent_minutes,
            "absent_teammates_typical_minutes": absent_teammates_typical_minutes,
            "max_replacement_uplift": max_replacement_uplift,
            "position": positions,
            "minutes": minutes,
        }
    )

    model = MinutesModel().fit(df)

    low = model.predict_one(
        own_recent_minutes=40.0,
        absent_teammates_typical_minutes=5.0,
        max_replacement_uplift=5.0,
        position="MID",
    )
    high = model.predict_one(
        own_recent_minutes=40.0,
        absent_teammates_typical_minutes=5.0,
        max_replacement_uplift=55.0,
        position="MID",
    )
    no_signal = model.predict_one(
        own_recent_minutes=40.0,
        absent_teammates_typical_minutes=5.0,
        max_replacement_uplift=float("nan"),
        position="MID",
    )

    assert 0.0 <= low <= 90.0
    assert 0.0 <= high <= 90.0
    assert 0.0 <= no_signal <= 90.0
    assert high > low
