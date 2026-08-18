"""
Tests for the chip-timing heuristic (airsenal.framework.chip_heuristics).
"""

from types import SimpleNamespace

from airsenal.conftest import TEST_PAST_SEASON, session_scope
from airsenal.framework.chip_heuristics import (
    another_unusual_gameweek_within,
    available_squad_players,
    can_field_valid_xi,
    captain_is_doubled_with_home_fixture,
    count_doubled_players,
    fixture_count,
    is_biggest_fixture_pileup_before_boundary,
    simulate_chip_decisions,
    suggest_chip_gameweeks,
    weeks_until_chip_boundary,
)
from airsenal.framework.schema import (
    Absence,
    Fixture,
    Player,
    PlayerAttributes,
    PlayerPrediction,
)
from airsenal.framework.squad import Squad

SEASON = TEST_PAST_SEASON


def _make_player(ts, player_id, name, team, position, gameweek=1, price=40):
    p = Player()
    p.player_id = player_id
    p.fpl_api_id = player_id
    p.name = name
    ts.add(p)
    pa = PlayerAttributes()
    pa.player_id = player_id
    pa.season = SEASON
    pa.gameweek = gameweek
    pa.price = price
    pa.team = team
    pa.position = position
    pa.player = p
    ts.add(pa)
    return player_id


def _make_fixture(ts, home_team, away_team, gameweek, tag="test"):
    fixture = Fixture()
    fixture.date = f"2021-{gameweek:02d}-01"
    fixture.gameweek = gameweek
    fixture.home_team = home_team
    fixture.away_team = away_team
    fixture.season = SEASON
    fixture.tag = tag
    ts.add(fixture)
    ts.flush()
    return fixture


def _make_prediction(ts, player_id, fixture, tag, predicted_points):
    pred = PlayerPrediction()
    pred.player_id = player_id
    pred.fixture_id = fixture.fixture_id
    pred.tag = tag
    pred.predicted_points = predicted_points
    ts.add(pred)


def test_fixture_count(fill_players):
    with session_scope() as ts:
        _make_fixture(ts, "ARS", "CHE", 40)
        _make_fixture(ts, "MIN", "LIV", 40)
        ts.commit()

        assert fixture_count(40, SEASON, dbsession=ts) == 2
        assert fixture_count(41, SEASON, dbsession=ts) == 0


def test_weeks_until_chip_boundary():
    assert weeks_until_chip_boundary(15) == 4
    assert weeks_until_chip_boundary(19) == 0
    assert weeks_until_chip_boundary(25) == 13
    assert weeks_until_chip_boundary(38) == 0


def test_can_field_valid_xi():
    enough = (
        [SimpleNamespace(position="GK")]
        + [SimpleNamespace(position="DEF")] * 3
        + [SimpleNamespace(position="MID")] * 4
        + [SimpleNamespace(position="FWD")] * 3
    )
    assert can_field_valid_xi(enough) is True

    not_enough = (
        [SimpleNamespace(position="GK")]
        + [SimpleNamespace(position="DEF")] * 2
        + [SimpleNamespace(position="MID")] * 3
        + [SimpleNamespace(position="FWD")] * 2
    )
    assert can_field_valid_xi(not_enough) is False

    no_gk = (
        [SimpleNamespace(position="DEF")] * 5
        + [SimpleNamespace(position="MID")] * 5
        + [SimpleNamespace(position="FWD")] * 3
    )
    assert can_field_valid_xi(no_gk) is False


def test_available_squad_players_excludes_blanking_and_injured(fill_players):
    with session_scope() as ts:
        fit_id = _make_player(ts, 92101, "Fit Player", "MIN", "MID", gameweek=50)
        blank_id = _make_player(ts, 92102, "Blank Player", "AVL", "MID", gameweek=50)
        injured_id = _make_player(
            ts, 92103, "Injured Player", "MIN", "MID", gameweek=50
        )
        ts.commit()

        _make_fixture(ts, "MIN", "CHE", 50)
        # AVL has no fixture at gw50 -> blank_id is unavailable
        ts.commit()

        absence = Absence()
        absence.player_id = injured_id
        absence.season = SEASON
        absence.reason = "injury"
        absence.date_from = "2021-09-01"
        absence.date_until = "2021-09-15"
        absence.gw_from = 49
        absence.gw_until = 52
        absence.timestamp = "2021-09-01"
        ts.add(absence)
        ts.commit()

        squad = Squad(season=SEASON)
        for pid in (fit_id, blank_id, injured_id):
            assert squad.add_player(pid, price=40, gameweek=50, dbsession=ts)

        available = available_squad_players(squad, SEASON, 50, dbsession=ts)
        assert {p.player_id for p in available} == {fit_id}


def test_count_doubled_players(fill_players):
    with session_scope() as ts:
        doubled_id = _make_player(
            ts, 92201, "Doubled Player", "MIN", "FWD", gameweek=51
        )
        single_id = _make_player(ts, 92202, "Single Player", "AVL", "FWD", gameweek=51)
        ts.commit()

        _make_fixture(ts, "MIN", "CHE", 51)
        _make_fixture(ts, "LIV", "MIN", 51)
        _make_fixture(ts, "AVL", "TOT", 51)
        ts.commit()

        squad = Squad(season=SEASON)
        squad.add_player(doubled_id, price=40, gameweek=51, dbsession=ts)
        squad.add_player(single_id, price=40, gameweek=51, dbsession=ts)

        assert count_doubled_players(squad, SEASON, 51, dbsession=ts) == 1


def test_captain_is_doubled_with_home_fixture(fill_players):
    with session_scope() as ts:
        star_id = _make_player(ts, 92301, "Star Player", "MIN", "FWD", gameweek=52)
        mate_id = _make_player(ts, 92302, "Squad Mate", "AVL", "MID", gameweek=52)
        ts.commit()

        home_fx = _make_fixture(ts, "MIN", "CHE", 52)
        away_fx = _make_fixture(ts, "LIV", "MIN", 52)
        mate_fx = _make_fixture(ts, "AVL", "TOT", 52)
        ts.commit()

        tag = "test-captain-tag"
        _make_prediction(ts, star_id, home_fx, tag, 8.0)
        _make_prediction(ts, star_id, away_fx, tag, 6.0)
        _make_prediction(ts, mate_id, mate_fx, tag, 3.0)
        ts.commit()

        squad = Squad(season=SEASON)
        squad.add_player(star_id, price=40, gameweek=52, dbsession=ts)
        squad.add_player(mate_id, price=40, gameweek=52, dbsession=ts)

        assert (
            captain_is_doubled_with_home_fixture(squad, 52, tag, SEASON, dbsession=ts)
            is True
        )


def test_captain_is_doubled_without_home_fixture(fill_players):
    with session_scope() as ts:
        star_id = _make_player(ts, 92401, "Star Player", "MIN", "FWD", gameweek=53)
        mate_id = _make_player(ts, 92402, "Squad Mate", "AVL", "MID", gameweek=53)
        ts.commit()

        away_fx1 = _make_fixture(ts, "CHE", "MIN", 53)
        away_fx2 = _make_fixture(ts, "LIV", "MIN", 53)
        mate_fx = _make_fixture(ts, "AVL", "TOT", 53)
        ts.commit()

        tag = "test-captain-tag-2"
        _make_prediction(ts, star_id, away_fx1, tag, 8.0)
        _make_prediction(ts, star_id, away_fx2, tag, 6.0)
        _make_prediction(ts, mate_id, mate_fx, tag, 3.0)
        ts.commit()

        squad = Squad(season=SEASON)
        squad.add_player(star_id, price=40, gameweek=53, dbsession=ts)
        squad.add_player(mate_id, price=40, gameweek=53, dbsession=ts)

        assert (
            captain_is_doubled_with_home_fixture(squad, 53, tag, SEASON, dbsession=ts)
            is False
        )


def test_is_biggest_fixture_pileup_before_boundary(fill_players):
    with session_scope() as ts:
        for i in range(2):
            _make_fixture(ts, f"T{i}A", f"T{i}B", 13)
        for i in range(3):
            _make_fixture(ts, f"U{i}A", f"U{i}B", 14)
        _make_fixture(ts, "V0A", "V0B", 15)
        for i in range(2):
            _make_fixture(ts, f"W{i}A", f"W{i}B", 18)
        ts.commit()

        assert (
            is_biggest_fixture_pileup_before_boundary(13, SEASON, dbsession=ts) is False
        )
        assert (
            is_biggest_fixture_pileup_before_boundary(14, SEASON, dbsession=ts) is True
        )
        assert (
            is_biggest_fixture_pileup_before_boundary(15, SEASON, dbsession=ts) is False
        )


def test_another_unusual_gameweek_within(fill_players):
    with session_scope() as ts:
        for gw, n_fixtures in [(21, 10), (22, 8), (23, 10)]:
            for i in range(n_fixtures):
                _make_fixture(ts, f"G{gw}T{i}A", f"G{gw}T{i}B", gw)
        ts.commit()

        assert another_unusual_gameweek_within(20, 3, SEASON, dbsession=ts) is True
        assert another_unusual_gameweek_within(21, 1, SEASON, dbsession=ts) is True
        assert another_unusual_gameweek_within(22, 1, SEASON, dbsession=ts) is False


def test_simulate_chip_decisions_blank_triggers_free_hit(fill_players):
    with session_scope() as ts:
        squad_spec = [
            (92501, "GK", "T1"),
            (92502, "DEF", "T2"),
            (92503, "DEF", "T3"),
            (92504, "DEF", "T4"),
            (92505, "MID", "T5"),
            (92506, "MID", "T6"),
            (92507, "MID", "T7"),
            (92508, "MID", "T8"),
            (92509, "FWD", "T9"),
            (92510, "FWD", "T10"),
            (92511, "FWD", "T11"),
        ]
        for pid, pos, team in squad_spec:
            _make_player(ts, pid, f"Player {pid}", team, pos, gameweek=5)
        ts.commit()

        # gw5: normal (10 fixtures), every squad player's team is playing
        _make_fixture(ts, "T1", "T2", 5)
        _make_fixture(ts, "T3", "T4", 5)
        _make_fixture(ts, "T5", "T6", 5)
        _make_fixture(ts, "T7", "T8", 5)
        _make_fixture(ts, "T9", "T10", 5)
        _make_fixture(ts, "T11", "X1", 5)
        _make_fixture(ts, "X2", "X3", 5)
        _make_fixture(ts, "X4", "X5", 5)
        _make_fixture(ts, "X6", "X7", 5)
        _make_fixture(ts, "X8", "X9", 5)

        # gw6: blank (3 fixtures) - only the GK and two defenders' teams play,
        # leaving no fieldable midfielders or forwards.
        _make_fixture(ts, "T1", "T2", 6)
        _make_fixture(ts, "T3", "Y1", 6)
        _make_fixture(ts, "Y2", "Y3", 6)
        ts.commit()

        squad = Squad(season=SEASON)
        for pid, _, _ in squad_spec:
            assert squad.add_player(pid, price=40, gameweek=5, dbsession=ts)

        tag = "test-simulate-tag"
        trace = simulate_chip_decisions(squad, [5, 6], tag, SEASON, dbsession=ts)

        assert trace[0]["gameweek"] == 5
        assert trace[0]["chip_played"] is None

        assert trace[1]["gameweek"] == 6
        assert trace[1]["chip_played"] == "free_hit"

        chip_gameweeks = suggest_chip_gameweeks(
            squad, [5, 6], tag, SEASON, dbsession=ts
        )
        assert chip_gameweeks["free_hit"] == 6
        assert chip_gameweeks["wildcard"] == -1
        assert chip_gameweeks["bench_boost"] == -1
        assert chip_gameweeks["triple_captain"] == -1
