"""
Scoring a squad against what actually happened.

`Squad.get_actual_points` is what `airsenal replay` totals, so an error here
misreports every backtest rather than any live decision.
"""

from dataclasses import dataclass

import pytest

from airsenal.game.season import CURRENT_SEASON
from airsenal.squad import squad as squad_module
from airsenal.squad.squad import Squad
from tests.conftest import session_scope

GAMEWEEK = 1

# conftest gives player i position by `i % 15`: 0-1 GK, 2-6 DEF, 7-11 MID, 12-14 FWD
GKS = [0, 1]
DEFS = [2, 3, 4, 5, 6]
MIDS = [7, 8, 9, 10, 11]
FWDS = [12, 13, 14]


@dataclass
class FakeScore:
    minutes: int
    points: int


def build_squad(dbsession, starting, bench_order):
    """A squad with a chosen starting eleven and a chosen bench order."""
    squad = Squad(season=CURRENT_SEASON)
    for player_id in GKS + DEFS + MIDS + FWDS:
        squad.add_player(
            player_id, check_budget=False, check_team=False, dbsession=dbsession
        )
    for player in squad.players:
        player.is_starting = player.player_id in starting
        player.is_captain = False
        player.is_vice_captain = False
        player.sub_position = (
            bench_order.index(player.player_id)
            if player.player_id in bench_order
            else None
        )
    return squad


@pytest.fixture
def scores(monkeypatch):
    """Stub the only database read `get_actual_points` makes."""
    table: dict[int, list[FakeScore]] = {}

    def fake(player_id, gameweek, season):
        return table.get(player_id, [FakeScore(minutes=90, points=2)])

    monkeypatch.setattr(squad_module, "get_playerscores_for_player_gameweek", fake)
    return table


def test_a_substitution_cannot_leave_an_illegal_formation(fill_players, scores):
    """
    Each substitution is judged against the formation the last one left.

    Judging every one against the original lineup let a third substitution
    through that took a 5-4-1 down to two defenders - a lineup FPL would never
    have fielded, scored as though it had.
    """
    starting = [GKS[0], *DEFS, *MIDS[:4], FWDS[0]]  # 5-4-1
    # the reserve keeper sits on the bench too, and can never make a legal
    # outfield substitution, so he goes last
    bench = [MIDS[4], FWDS[1], FWDS[2], GKS[1]]

    with session_scope() as ts:
        squad = build_squad(ts, starting, bench)

    # three defenders blank; every substitute plays and scores 5
    for player_id in DEFS[:3]:
        scores[player_id] = [FakeScore(minutes=0, points=0)]
    for player_id in bench:
        scores[player_id] = [FakeScore(minutes=90, points=5)]

    points = squad.get_actual_points(GAMEWEEK, CURRENT_SEASON)

    # 8 players who played, at 2 each, plus at most two legal substitutions at 5
    assert points == 8 * 2 + 2 * 5


def test_the_vice_captain_is_doubled_across_a_double_gameweek(fill_players, scores):
    """
    FPL doubles the vice-captain's whole gameweek, not their last fixture.

    The running total used to be assigned rather than added to, so only the
    second fixture of a double was doubled.
    """
    starting = [GKS[0], *DEFS, *MIDS[:4], FWDS[0]]
    # the reserve keeper sits on the bench too, and can never make a legal
    # outfield substitution, so he goes last
    bench = [MIDS[4], FWDS[1], FWDS[2], GKS[1]]

    with session_scope() as ts:
        squad = build_squad(ts, starting, bench)

    captain = squad.get_player_from_id(MIDS[0])
    vice = squad.get_player_from_id(MIDS[1])
    captain.is_captain = True
    vice.is_vice_captain = True

    # the captain blanks, so the vice-captain is doubled instead
    scores[MIDS[0]] = [FakeScore(minutes=0, points=0)]
    scores[MIDS[1]] = [FakeScore(minutes=90, points=9), FakeScore(minutes=90, points=2)]
    # nobody comes off the bench for the captain: the sub also blanks
    for player_id in bench:
        scores[player_id] = [FakeScore(minutes=0, points=0)]

    points = squad.get_actual_points(GAMEWEEK, CURRENT_SEASON)

    # 9 who played at 2, the vice-captain's 11, and his 11 again for the armband
    assert points == 9 * 2 + 11 + 11
