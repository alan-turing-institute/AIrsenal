"""
The chip-timing rules, on made-up fixtures and a made-up squad.

Every club in the squad is its own, so a club's fixtures are one player's, and
a blank or a double can be aimed at exactly the players it should hit.
"""

from dataclasses import dataclass, field

import pytest

from airsenal.game.enums import Chip
from airsenal.optimization import chip_timing
from airsenal.optimization import run_transfers as rt
from airsenal.optimization.chip_timing import (
    can_field_eleven,
    chip_gameweeks,
    decide_chips,
)
from airsenal.optimization.moves import ChipGameweeks

POSITIONS = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
CLUBS = [f"C{i:02d}" for i in range(20)]


@dataclass
class _Player:
    player_id: int
    team: str
    position: str
    points: float = 2.0
    is_captain: bool = False
    is_vice_captain: bool = False
    predicted_points: dict = field(default_factory=dict)

    def calc_predicted_points(self, tag):
        self.predicted_points = {tag: dict.fromkeys(range(1, 39), self.points)}


@dataclass
class _Squad:
    players: list


@dataclass
class _Fixture:
    home_team: str
    away_team: str


def _squad(best: int = 0) -> _Squad:
    """Fifteen players, one per club C00-C14, with player `best` the captain."""
    players = [
        _Player(i, CLUBS[i], position, points=10.0 if i == best else 2.0)
        for i, position in enumerate(POSITIONS)
    ]
    return _Squad(players)


def _normal() -> list[_Fixture]:
    return [_Fixture(CLUBS[2 * i], CLUBS[2 * i + 1]) for i in range(10)]


def _blank(missing: set[str]) -> list[_Fixture]:
    """A gameweek in which every club in `missing` has no fixture."""
    return [f for f in _normal() if not {f.home_team, f.away_team} & missing]


def _double(clubs: list[str], home: bool = True) -> list[_Fixture]:
    """A normal gameweek plus a second fixture for each of `clubs`."""
    extra = [_Fixture(club, "C19") if home else _Fixture("C19", club) for club in clubs]
    return _normal() + extra


class _Fit:
    def is_injured_or_suspended(self, season, current_gameweek, fixture_gameweek):  # noqa: ARG002
        return False


@pytest.fixture
def fixtures(monkeypatch):
    """Set a gameweek's fixtures with `fixtures[gameweek] = [...]`; normal otherwise."""
    by_gameweek: dict[int, list[_Fixture]] = {}

    def get_fixtures_for_gameweeks(gameweeks, season=None, dbsession=None):
        (gameweek,) = gameweeks
        return by_gameweek.get(gameweek, _normal())

    monkeypatch.setattr(
        chip_timing, "get_fixtures_for_gameweeks", get_fixtures_for_gameweeks
    )
    monkeypatch.setattr(chip_timing, "get_max_gameweek", lambda *a, **k: 38)
    monkeypatch.setattr(chip_timing, "get_player", lambda *a, **k: _Fit())
    monkeypatch.setattr(chip_timing, "get_session", lambda *a, **k: None)
    return by_gameweek


ALL_CHIPS = frozenset(Chip)


def _chips(decisions):
    return [chip for _, chip, _ in decisions]


def test_normal_gameweeks_early_in_a_half_play_nothing(fixtures):
    decisions = decide_chips(_squad(), ALL_CHIPS, [3, 4, 5], "tag", season="2526")
    assert _chips(decisions) == [None, None, None]


def test_a_blank_that_breaks_the_eleven_is_a_free_hit(fixtures):
    # both goalkeepers' clubs blank
    fixtures[4] = _blank({CLUBS[0], CLUBS[1]})

    decisions = decide_chips(_squad(), ALL_CHIPS, [3, 4, 5], "tag", season="2526")

    assert _chips(decisions) == [None, Chip.FREE_HIT, None]


def test_a_blank_the_squad_can_still_field_eleven_for_is_left(fixtures):
    fixtures[4] = _blank({CLUBS[2]})

    decisions = decide_chips(_squad(), ALL_CHIPS, [4], "tag", season="2526")

    assert _chips(decisions) == [None]


def test_a_double_for_the_whole_squad_is_a_bench_boost(fixtures):
    fixtures[6] = _double(CLUBS[:15])

    decisions = decide_chips(_squad(), ALL_CHIPS, [6], "tag", season="2526")

    assert _chips(decisions) == [Chip.BENCH_BOOST]


def test_a_captain_with_a_home_double_is_a_triple_captain(fixtures):
    fixtures[6] = _double([CLUBS[9]])

    decisions = decide_chips(_squad(best=9), ALL_CHIPS, [6], "tag", season="2526")

    assert _chips(decisions) == [Chip.TRIPLE_CAPTAIN]


def test_a_captain_doubled_away_twice_is_not(fixtures):
    """
    Two away games for the captain: no triple captain.

    Both squad chips are left, and nothing else unusual is coming, so the
    double gets a free hit instead.
    """
    fixtures[6] = [
        f for f in _normal() if CLUBS[9] not in (f.home_team, f.away_team)
    ] + [_Fixture("C19", CLUBS[9]), _Fixture("C18", CLUBS[9])]

    decisions = decide_chips(_squad(best=9), ALL_CHIPS, [6], "tag", season="2526")

    assert _chips(decisions) == [Chip.FREE_HIT]


def test_a_double_with_another_close_behind_is_a_wildcard(fixtures):
    fixtures[6] = _double([CLUBS[3]])
    fixtures[8] = _blank({CLUBS[4]})

    decisions = decide_chips(_squad(), ALL_CHIPS, [6], "tag", season="2526")

    assert _chips(decisions) == [Chip.WILDCARD]


def test_more_chips_than_gameweeks_before_they_expire_forces_one(fixtures):
    """Four chips, and gameweeks 17, 18 and 19 to play them in: play one now."""
    decisions = decide_chips(_squad(), ALL_CHIPS, [17], "tag", season="2526")

    # in a normal gameweek the forced order starts with the triple captain
    assert _chips(decisions) == [Chip.TRIPLE_CAPTAIN]


def test_before_2025_26_only_the_wildcard_expires_at_the_split(fixtures):
    """The other chips can wait until the end of the season."""
    decisions = decide_chips(_squad(), ALL_CHIPS, [17, 18, 19], "tag", season="2425")

    assert _chips(decisions) == [None, None, Chip.WILDCARD]


def test_the_second_half_brings_the_chips_back(fixtures):
    """Nothing left in the first half, and a full set again from gameweek 20."""
    decisions = decide_chips(_squad(), frozenset(), [19, 20], "tag", season="2526")

    assert decisions[0][1] is None
    assert "no chips left" in decisions[0][2]
    assert "no chips left" not in decisions[1][2]


class _Injured:
    def is_injured_or_suspended(self, season, current_gameweek, fixture_gameweek):  # noqa: ARG002
        return True


@pytest.mark.parametrize("second_goalkeeper", [_Injured(), None])
def test_a_player_who_cannot_play_does_not_count_towards_the_eleven(
    fixtures, monkeypatch, second_goalkeeper
):
    """
    Only the first goalkeeper's club blanks, but the second is injured or unknown.

    With the second goalkeeper fit, the same blank is left alone.
    """
    fixtures[4] = [_Fixture(CLUBS[1], CLUBS[18]), *_normal()[1:9]]
    monkeypatch.setattr(
        chip_timing,
        "get_player",
        lambda player_id, **k: second_goalkeeper if player_id == 1 else _Fit(),
    )

    decisions = decide_chips(_squad(), ALL_CHIPS, [4], "tag", season="2526")

    assert _chips(decisions) == [Chip.FREE_HIT]


NO_FREE_HIT = frozenset({Chip.WILDCARD, Chip.TRIPLE_CAPTAIN, Chip.BENCH_BOOST})


def test_a_blank_leaving_nine_players_is_a_wildcard_without_a_free_hit(fixtures):
    fixtures[4] = _blank(set(CLUBS[:6]))

    decisions = decide_chips(_squad(), NO_FREE_HIT, [4], "tag", season="2526")

    assert _chips(decisions) == [Chip.WILDCARD]


@pytest.mark.parametrize(("gameweek", "chip"), [(17, Chip.WILDCARD), (4, None)])
def test_a_blank_without_a_free_hit_wildcards_only_if_the_wildcard_is_expiring(
    fixtures, gameweek, chip
):
    """
    Thirteen players left, but neither goalkeeper: bad, but not severe.

    In 2425 only the wildcard expires at gameweek 19, so none of the other chips
    is forced.
    """
    fixtures[gameweek] = _blank({CLUBS[0]})

    decisions = decide_chips(_squad(), NO_FREE_HIT, [gameweek], "tag", season="2425")

    assert _chips(decisions) == [chip]


@pytest.mark.parametrize(
    ("gameweek_fixtures", "chip"),
    [
        (_blank({CLUBS[2]}), Chip.FREE_HIT),
        (_double([CLUBS[3]]), Chip.WILDCARD),
    ],
)
def test_a_forced_chip_is_chosen_by_the_kind_of_gameweek(
    fixtures, gameweek_fixtures, chip
):
    fixtures[17] = gameweek_fixtures

    decisions = decide_chips(_squad(), ALL_CHIPS, [17], "tag", season="2526")

    assert _chips(decisions) == [chip]


BOOSTS = frozenset({Chip.BENCH_BOOST, Chip.TRIPLE_CAPTAIN})


@pytest.mark.parametrize(
    ("available", "chip"),
    [
        (BOOSTS, Chip.BENCH_BOOST),
        (frozenset({Chip.WILDCARD}), Chip.WILDCARD),
        (frozenset({Chip.FREE_HIT}), Chip.FREE_HIT),
    ],
)
def test_a_last_chip_goes_on_the_biggest_double_before_it_expires(
    fixtures, available, chip
):
    """One club doubled, not the captain's, and nothing bigger to come."""
    fixtures[6] = _double([CLUBS[3]])

    decisions = decide_chips(_squad(), available, [6], "tag", season="2526")

    assert _chips(decisions) == [chip]


@pytest.mark.parametrize(
    "available", [BOOSTS, frozenset({Chip.WILDCARD}), frozenset({Chip.FREE_HIT})]
)
def test_a_last_chip_waits_for_a_bigger_double(fixtures, available):
    fixtures[6] = _double([CLUBS[3]])
    fixtures[10] = _double([CLUBS[3], CLUBS[4]])

    decisions = decide_chips(_squad(), available, [6], "tag", season="2526")

    assert _chips(decisions) == [None]


@pytest.mark.parametrize(
    "players",
    [
        [],
        [_Player(i, CLUBS[i], "DEF") for i in range(11)],
    ],
)
def test_no_goalkeeper_is_no_eleven(players):
    assert not can_field_eleven(players)


def test_a_full_squad_can_field_eleven():
    assert can_field_eleven(_squad().players)


def test_a_chip_decided_twice_is_pinned_to_the_first_gameweek():
    decisions = [
        (19, Chip.WILDCARD, ""),
        (20, Chip.WILDCARD, ""),
        (21, Chip.BENCH_BOOST, ""),
    ]

    assert chip_gameweeks(decisions) == ChipGameweeks(wildcard=19, bench_boost=21)


# ------------------------------------------ what the heuristic starts from


class _Fetcher:
    def __init__(self, names):
        self.names = names

    def get_available_chips(self, fpl_team_id=None):  # noqa: ARG002
        return self.names


def test_a_live_run_asks_the_api_which_chips_are_left():
    available = rt.available_chips(
        ChipGameweeks(), 5, season="2526", fetcher=_Fetcher(["freehit", "3xc"])
    )
    assert available == {Chip.FREE_HIT, Chip.TRIPLE_CAPTAIN}


def test_a_chip_the_api_calls_something_new_is_an_error():
    with pytest.raises(ValueError, match="does not know"):
        rt.available_chips(
            ChipGameweeks(), 5, season="2526", fetcher=_Fetcher(["mystery"])
        )


def test_a_replay_works_the_chips_left_out_from_what_it_played():
    chips = ChipGameweeks(played=((3, Chip.WILDCARD), (10, Chip.BENCH_BOOST)))

    assert rt.available_chips(chips, 12, season="2526") == {
        Chip.FREE_HIT,
        Chip.TRIPLE_CAPTAIN,
    }
    # a new half, a new set
    assert rt.available_chips(chips, 20, season="2526") == ALL_CHIPS


def test_the_heuristic_keeps_the_chips_already_played(fixtures):
    fixtures[4] = _blank({CLUBS[0], CLUBS[1]})
    chips = ChipGameweeks(played=((2, Chip.WILDCARD),), heuristic=True)

    decided = rt._chips_by_heuristic(chips, _squad(), [3, 4, 5], "tag", season="2526")

    assert decided.free_hit == 4
    assert decided.wildcard == -1
    assert decided.played == chips.played
    assert decided.heuristic
