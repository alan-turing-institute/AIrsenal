"""
test various methods of the Team class.
"""

import logging

import pytest
from rich.console import Console

from airsenal.game.season import CURRENT_SEASON
from airsenal.reporting.squad_view import formation_table
from airsenal.squad.lineup import FORMATION_SLOTS
from airsenal.squad.pricing import selling_price_from_api
from airsenal.squad.squad import Squad
from tests.conftest import session_scope

TEST_SEASON = CURRENT_SEASON


def test_formation_slots():
    assert FORMATION_SLOTS == {
        0: (),
        1: (2,),
        2: (1, 3),
        3: (1, 2, 3),
        4: (0, 1, 3, 4),
        5: (0, 1, 2, 3, 4),
    }


def test_add_player_by_id(fill_players):
    """
    Should be able to add a player with integer argument
    """
    with session_scope() as ts:
        t = Squad(season=TEST_SEASON)
        added_ok = t.add_player(50, dbsession=ts)
        assert added_ok


def test_add_player_by_name(fill_players):
    """
    Should be able to add a player with string argument
    """
    with session_scope() as ts:
        t = Squad(season=TEST_SEASON)
        added_ok = t.add_player("Alice", dbsession=ts)
        assert added_ok


def test_cant_add_same_player(fill_players):
    """
    can't add a player thats already on the squad.
    """
    with session_scope() as ts:
        t = Squad(season=TEST_SEASON)
        added_ok = t.add_player(1, dbsession=ts)
        assert added_ok
        added_ok = t.add_player(1, dbsession=ts)
        assert not added_ok


def test_cant_add_too_many_per_position(fill_players):
    """
    no more than two keepers, 5 defenders, 5 midfielders, 3 forwards.
    """
    with session_scope() as ts:
        t = Squad(season=TEST_SEASON)
        # keepers
        assert t.add_player("Alice", dbsession=ts)
        assert t.add_player("Bob", dbsession=ts)
        assert not t.add_player("Pedro", dbsession=ts)
        # defenders
        assert t.add_player("Carla", dbsession=ts)
        assert t.add_player("Donald", dbsession=ts)
        assert t.add_player("Erica", dbsession=ts)
        assert t.add_player("Frank", dbsession=ts)
        assert t.add_player("Gerry", dbsession=ts)
        assert not t.add_player("Stefan", dbsession=ts)


def test_cant_add_too_many_per_squad(fill_players):
    """
    no more than three from the same squad.
    """
    with session_scope() as ts:
        t = Squad(season=TEST_SEASON)
        assert t.add_player(1, dbsession=ts)
        assert t.add_player(21, dbsession=ts)
        assert t.add_player(41, dbsession=ts)
        assert not t.add_player(61, dbsession=ts)


def test_cant_exceed_budget():
    """
    try and make an expensive squad
    """
    with session_scope() as ts:
        t = Squad(season=TEST_SEASON)
        added_ok = True
        added_ok = added_ok and t.add_player(45, dbsession=ts)
        added_ok = added_ok and t.add_player(46, dbsession=ts)
        added_ok = added_ok and t.add_player(47, dbsession=ts)
        added_ok = added_ok and t.add_player(48, dbsession=ts)
        added_ok = added_ok and t.add_player(49, dbsession=ts)
        added_ok = added_ok and t.add_player(50, dbsession=ts)
        added_ok = added_ok and t.add_player(51, dbsession=ts)
        added_ok = added_ok and t.add_player(52, dbsession=ts)
        added_ok = added_ok and t.add_player(53, dbsession=ts)
        added_ok = added_ok and t.add_player(54, dbsession=ts)
        added_ok = added_ok and t.add_player(55, dbsession=ts)
        added_ok = added_ok and t.add_player(56, dbsession=ts)
        added_ok = added_ok and t.add_player(57, dbsession=ts)
        added_ok = added_ok and t.add_player(58, dbsession=ts)
        added_ok = added_ok and t.add_player(59, dbsession=ts)
        assert not added_ok


def test_remove_player(fill_players):
    """
    add a player then remove them.
    """
    with session_scope() as ts:
        t = Squad(season=TEST_SEASON)
        t.add_player(1, dbsession=ts)
        assert len(t.players) == 1
        assert t.num_position["GK"] == 1
        t.remove_player(1, use_api=False, dbsession=ts)
        assert len(t.players) == 0
        assert t.num_position["GK"] == 0
        assert t.budget == 1000


def test_empty_squad(fill_players):
    """
    shouldn't be able to estimate points with
    no players.
    """
    t = Squad()
    with pytest.raises(RuntimeError) as errmsg:
        t.get_expected_points(1, "dummy")
    assert str(errmsg.value) == "Squad is incomplete"


def test_order_substitutes():
    t = Squad()

    class MockPlayer:
        def __init__(self, points, is_starting, name, squad):
            self.predicted_points = {0: {0: points}}
            self.is_starting = is_starting
            self.name = name
            self.squad = squad
            self.sub_position = None

    players = [
        MockPlayer(10, False, "a", "A"),
        MockPlayer(9, False, "b", "B"),
        MockPlayer(8, False, "c", "C"),
        MockPlayer(11, True, "d", "D"),
    ]

    t.players = players
    t.order_substitutes(0, 0)

    expected_sub_positions = [0, 1, 2, None]
    for player, sub_position in zip(players, expected_sub_positions, strict=False):
        assert player.sub_position == sub_position

    # test the logic that's use in __repr__ as well
    subs = [p for p in t.players if not p.is_starting]
    subs.sort(key=lambda p: p.sub_position)
    expected_names = ["a", "b", "c"]
    for player, expected_name in zip(subs, expected_names, strict=False):
        assert player.name == expected_name


def test_formation_table():
    t = Squad()

    class MockPlayer:
        def __init__(
            self,
            name: str,
            position: str,
            is_starting: bool,
            sub_position: int | None = None,
        ):
            self.name = name
            self.team = "TEST"
            self.position = position
            self.is_starting = is_starting
            self.is_captain = name == "Captain"
            self.is_vice_captain = name == "Vice"
            self.sub_position = sub_position
            self.predicted_points = {"tag": {1: 5.0}}

        def __str__(self) -> str:
            return self.name

    t.players = [
        MockPlayer("Keeper", "GK", True),
        MockPlayer("Defender One", "DEF", True),
        MockPlayer("Defender Two", "DEF", True),
        MockPlayer("Defender Three", "DEF", True),
        MockPlayer("Midfielder One", "MID", True),
        MockPlayer("Midfielder Two", "MID", True),
        MockPlayer("Midfielder Three", "MID", True),
        MockPlayer("Midfielder Four", "MID", True),
        MockPlayer("Captain", "FWD", True),
        MockPlayer("Vice", "FWD", True),
        MockPlayer("Forward Three", "FWD", True),
        MockPlayer("Sub Keeper", "GK", False, 0),
        MockPlayer("Sub One", "DEF", False, 0),
        MockPlayer("Sub Two", "MID", False, 1),
        MockPlayer("Sub Three", "FWD", False, 2),
    ]
    scoring_calls = []

    def get_expected_points(gameweek, tag, bench_boost=False, triple_captain=False):
        scoring_calls.append((gameweek, tag, bench_boost, triple_captain))
        return 60.0 + 20.0 * bench_boost + 5.0 * triple_captain

    t.get_expected_points = get_expected_points
    console = Console(record=True, width=100)

    console.print(formation_table(t, "tag", 1))
    console.print(formation_table(t, "tag", 1, bench_boost=True))
    console.print(formation_table(t, "tag", 1, triple_captain=True))

    output = console.export_text()
    assert "Captain" in output
    assert "(C)" in output
    assert "Substitutes" in output
    assert "5.0 pts" in output
    assert "GAMEWEEK 1" in output
    assert "60.0pts" in output
    assert "80.0pts" in output
    assert "with bench boost" in output
    assert "65.0pts" in output
    assert "with triple captain" in output
    assert "(TC)" in output
    assert scoring_calls == [
        (1, "tag", False, False),
        (1, "tag", True, False),
        (1, "tag", False, True),
    ]


def test_get_expected_points():
    t = Squad()

    class MockPlayer:
        def __init__(
            self,
            name,
            squad,
            position,
            points,
            is_starting,
            is_captain,
            is_vice_captain,
        ):
            self.name = name
            self.squad = squad
            self.position = position
            self.predicted_points = {0: {0: points}}
            self.is_starting = is_starting
            self.sub_position = None
            self.is_captain = is_captain
            self.is_vice_captain = is_vice_captain

        def calc_predicted_points(self, tag):
            pass

    # 3 pts captain (x2 = 6pts, or x3 = 9pts for TC)
    # 2 pts starters
    # 1 pt subs
    players = [
        MockPlayer("a", "A", "GK", 2, True, False, False),
        MockPlayer("b", "B", "GK", 1, False, False, False),  # sub 1
        MockPlayer("c", "C", "DEF", 2, True, False, False),
        MockPlayer("d", "D", "DEF", 2, True, False, False),
        MockPlayer("e", "E", "DEF", 2, True, False, False),
        MockPlayer("f", "F", "DEF", 1, False, False, False),  # sub 2
        MockPlayer("g", "G", "DEF", 1, False, False, False),  # sub 3
        MockPlayer("h", "H", "MID", 2, True, False, False),
        MockPlayer("i", "I", "MID", 2, True, False, False),
        MockPlayer("j", "J", "MID", 2, True, False, False),
        MockPlayer("k", "K", "MID", 2, True, False, False),
        MockPlayer("l", "L", "MID", 1, False, False, False),  # sub 4
        MockPlayer("m", "M", "FWD", 3, True, True, False),  # captain
        MockPlayer("n", "N", "FWD", 2, True, False, True),  # vice-captain
        MockPlayer("o", "O", "FWD", 2, True, False, False),
    ]

    t.players = players
    t.num_position = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

    # no chips
    assert t.get_expected_points(0, 0) == 26
    # bench boost
    assert t.get_expected_points(0, 0, bench_boost=True) == 30
    # triple captain
    assert t.get_expected_points(0, 0, triple_captain=True) == 29


class _Recorder(logging.Handler):
    """Collect log records rather than printing them."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def squad_logs():
    """The records pricing.py logs, at every level."""
    logger = logging.getLogger("airsenal.squad.pricing")
    handler = _Recorder()
    original_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)


class FakeFetcher:
    """Just enough of FPLDataFetcher to answer for the current picks."""

    def __init__(self, picks=None, error=None):
        self.picks = picks if picks is not None else {}
        self.error = error

    def get_current_picks(self, fpl_team_id=None):  # noqa: ARG002
        if self.error is not None:
            raise self.error
        return self.picks


def test_selling_price_comes_from_the_picks_for_a_player_we_own(squad_logs):
    fetcher = FakeFetcher(picks={7: {"element": 7, "selling_price": 62}})

    assert selling_price_from_api(7, "Owned Player", fetcher=fetcher) == 62
    assert squad_logs == []


def test_a_player_we_do_not_own_has_no_selling_price_and_is_not_a_failure(squad_logs):
    """The optimizer prices squads that do not exist: a wildcard's, say.

    Those players are not in the entry's picks and never will be, so asking the
    API for a sale price they cannot have is an ordinary miss. It used to raise a
    KeyError inside a catch-all handler, which logged a warning blaming a failed
    login and printed a traceback, once per player per gameweek.
    """
    fetcher = FakeFetcher(picks={7: {"element": 7, "selling_price": 62}})

    assert selling_price_from_api(999, "Unowned Player", fetcher=fetcher) is None
    assert [record.levelno for record in squad_logs] == [logging.DEBUG]


def test_failing_to_reach_the_api_does_warn(squad_logs):
    """Not owning a player is routine; not being able to ask is not."""
    fetcher = FakeFetcher(error=RuntimeError("not logged in"))

    assert selling_price_from_api(7, "Owned Player", fetcher=fetcher) is None
    assert [record.levelno for record in squad_logs] == [logging.WARNING]


def test_a_pick_without_a_usable_price_warns_rather_than_raising(squad_logs):
    fetcher = FakeFetcher(picks={7: {"element": 7}})

    assert selling_price_from_api(7, "Owned Player", fetcher=fetcher) is None
    assert [record.levelno for record in squad_logs] == [logging.WARNING]
