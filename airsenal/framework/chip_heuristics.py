"""
Chip-timing heuristic: a rule-based baseline for when to play FPL's four chips
(wildcard, free hit, bench boost, triple captain), built to compare against
free-choice MCTS chip selection (airsenal.framework.mcts_optimization). This module
is a direct translation of the flowchart in chip_timing_heuristic.svg/.pdf into
code - one building-block predicate per flowchart question, plus a walk-forward
simulator (simulate_chip_decisions) that evaluates the tree one gameweek at a time,
exactly as the flowchart is meant to be read.

Assumes the squad passed in is held fixed for the whole gameweek_range - this is a
chip-timing heuristic, not a transfer search, so it doesn't simulate transfers
alongside chip decisions.
"""

from sqlalchemy.orm.session import Session

from airsenal.framework.optimization_utils import chip_half
from airsenal.framework.player import CandidatePlayer
from airsenal.framework.schema import session
from airsenal.framework.squad import FORMATIONS, Squad
from airsenal.framework.utils import (
    CURRENT_SEASON,
    get_fixtures_for_gameweek,
    get_player,
    was_historic_absence,
)

NORMAL_GAMEWEEK_FIXTURE_COUNT = 10
BENCH_BOOST_DOUBLE_THRESHOLD = 14
SEVERE_BLANK_FIELDABLE_THRESHOLD = 9
URGENT_GAMEWEEKS_REMAINING = 3
LOOKAHEAD_GAMEWEEKS = 3

CHIP_TYPES = ["wildcard", "free_hit", "bench_boost", "triple_captain"]
CHIP_HALF_BOUNDARIES = {1: 19, 2: 38}


def fixture_count(gameweek: int, season: str, dbsession: Session = session) -> int:
    """Number of fixtures scheduled for `gameweek` - =10 is a normal gameweek,
    <10 a blank, >10 a double.
    """
    return len(get_fixtures_for_gameweek(gameweek, season=season, dbsession=dbsession))


def weeks_until_chip_boundary(gameweek: int) -> int:
    """Gameweeks remaining until the current chip half expires (GW19 or GW38)."""
    return CHIP_HALF_BOUNDARIES[chip_half(gameweek)] - gameweek


def available_squad_players(
    squad: Squad, season: str, gameweek: int, dbsession: Session = session
) -> list[CandidatePlayer]:
    """Squad players whose team has a fixture this gameweek and who aren't
    injured/suspended - assumes the squad is unchanged from when it was passed in
    (see module docstring).
    """
    teams_playing = {
        team
        for fixture in get_fixtures_for_gameweek(
            gameweek, season=season, dbsession=dbsession
        )
        for team in (fixture.home_team, fixture.away_team)
    }
    available = []
    for candidate in squad.players:
        if candidate.team not in teams_playing:
            continue
        player = get_player(candidate.player_id, dbsession=dbsession)
        if player is None:
            continue
        if season == CURRENT_SEASON:
            unavailable = player.is_injured_or_suspended(season, gameweek, gameweek)
        else:
            unavailable = was_historic_absence(
                player, gameweek=gameweek, season=season, dbsession=dbsession
            )
        if not unavailable:
            available.append(candidate)
    return available


def can_field_valid_xi(available_players: list[CandidatePlayer]) -> bool:
    """Whether some formation in squad.FORMATIONS (1 GK + a (DEF, MID, FWD) tuple)
    can be filled from `available_players`.
    """
    counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for player in available_players:
        if player.position in counts:
            counts[player.position] += 1
    if counts["GK"] < 1:
        return False
    return any(
        counts["DEF"] >= defs and counts["MID"] >= mids and counts["FWD"] >= fwds
        for defs, mids, fwds in FORMATIONS
    )


def count_doubled_players(
    squad: Squad, season: str, gameweek: int, dbsession: Session = session
) -> int:
    """Number of squad players whose team has two (or more) fixtures this
    gameweek.
    """
    fixtures = get_fixtures_for_gameweek(gameweek, season=season, dbsession=dbsession)
    team_fixture_counts: dict[str, int] = {}
    for fixture in fixtures:
        for team in (fixture.home_team, fixture.away_team):
            team_fixture_counts[team] = team_fixture_counts.get(team, 0) + 1
    return sum(1 for p in squad.players if team_fixture_counts.get(p.team, 0) >= 2)


def captain_is_doubled_with_home_fixture(
    squad: Squad, gameweek: int, tag: str, season: str, dbsession: Session = session
) -> bool:
    """Whether the squad's highest-predicted-points player for `gameweek` (see
    Squad.pick_captains) has two fixtures this gameweek, at least one at home.
    """
    for candidate in squad.players:
        candidate.calc_predicted_points(tag)
    squad.pick_captains(gameweek, tag)
    captain = next((p for p in squad.players if p.is_captain), None)
    if captain is None:
        return False

    fixtures = get_fixtures_for_gameweek(gameweek, season=season, dbsession=dbsession)
    captain_fixtures = [
        f for f in fixtures if captain.team in (f.home_team, f.away_team)
    ]
    has_home_fixture = any(f.home_team == captain.team for f in captain_fixtures)
    return len(captain_fixtures) >= 2 and has_home_fixture


def is_biggest_fixture_pileup_before_boundary(
    gameweek: int, season: str, dbsession: Session = session
) -> bool:
    """Whether `gameweek` has the most fixtures of any gameweek between now and the
    current chip half's boundary (GW19/38) - i.e. the single best remaining week to
    force a chip on, if one must be forced this half.
    """
    boundary = CHIP_HALF_BOUNDARIES[chip_half(gameweek)]
    counts = {
        gw: fixture_count(gw, season, dbsession) for gw in range(gameweek, boundary + 1)
    }
    return counts[gameweek] == max(counts.values())


def another_unusual_gameweek_within(
    gameweek: int, n: int, season: str, dbsession: Session = session
) -> bool:
    """Whether any of the `n` gameweeks after `gameweek` is a blank or double
    (fixture count != NORMAL_GAMEWEEK_FIXTURE_COUNT).
    """
    return any(
        fixture_count(gw, season, dbsession) != NORMAL_GAMEWEEK_FIXTURE_COUNT
        for gw in range(gameweek + 1, gameweek + 1 + n)
    )


def _path_a(
    available_chip_types: list[str], gameweek: int, season: str, dbsession: Session
) -> tuple[str | None, str]:
    """Forced play - reached when there are more chips left than weeks to fit them
    into. See chip_timing_heuristic.svg, Path A.
    """
    if len(available_chip_types) == 1:
        return available_chip_types[0], "path A: only one chip left"

    count = fixture_count(gameweek, season, dbsession)
    if count < NORMAL_GAMEWEEK_FIXTURE_COUNT:
        preference = ["free_hit", "wildcard", "triple_captain", "bench_boost"]
    elif count > NORMAL_GAMEWEEK_FIXTURE_COUNT:
        preference = ["wildcard", "free_hit", "bench_boost", "triple_captain"]
    else:
        preference = ["triple_captain", "bench_boost", "free_hit", "wildcard"]

    for chip in preference:
        if chip in available_chip_types:
            return chip, f"path A: forced, {count} fixtures, preference order"
    return None, "path A: unreachable"  # preference always covers all 4 chip types


def _path_ba(
    squad: Squad,
    available_chip_types: list[str],
    gameweek: int,
    season: str,
    dbsession: Session,
) -> tuple[str | None, str]:
    """Blank gameweek - see chip_timing_heuristic.svg, Path BA."""
    available_players = available_squad_players(squad, season, gameweek, dbsession)
    if can_field_valid_xi(available_players):
        return None, "path BA: can still field 11"

    if "free_hit" in available_chip_types:
        return "free_hit", "path BA: blank breaks the XI, free hit available"

    if "wildcard" in available_chip_types and (
        len(available_players) <= SEVERE_BLANK_FIELDABLE_THRESHOLD
        or weeks_until_chip_boundary(gameweek) < URGENT_GAMEWEEKS_REMAINING
    ):
        return "wildcard", "path BA: severe blank or wildcard about to expire"

    return None, "path BA: no fallback chip available"


def _path_bb(
    squad: Squad,
    available_chip_types: list[str],
    gameweek: int,
    tag: str,
    season: str,
    dbsession: Session,
) -> tuple[str | None, str]:
    """Double gameweek - see chip_timing_heuristic.svg, Path BB."""
    if (
        "bench_boost" in available_chip_types
        and count_doubled_players(squad, season, gameweek, dbsession)
        > BENCH_BOOST_DOUBLE_THRESHOLD
    ):
        return "bench_boost", "path BB: whole squad doubled, bench boost available"

    if (
        "triple_captain" in available_chip_types
        and captain_is_doubled_with_home_fixture(
            squad, gameweek, tag, season, dbsession
        )
    ):
        return "triple_captain", "path BB: captain doubled with a home fixture"

    have_wildcard = "wildcard" in available_chip_types
    have_free_hit = "free_hit" in available_chip_types

    if not have_wildcard and not have_free_hit:
        have_bench_boost = "bench_boost" in available_chip_types
        have_triple_captain = "triple_captain" in available_chip_types
        if (
            have_bench_boost or have_triple_captain
        ) and is_biggest_fixture_pileup_before_boundary(gameweek, season, dbsession):
            # same bench_boost-before-triple_captain priority as the checks above
            fallback_chip = "bench_boost" if have_bench_boost else "triple_captain"
            return (
                fallback_chip,
                (
                    "path BB: neither wildcard nor free hit - falling back to "
                    "bench boost/triple captain on the biggest pile-up"
                ),
            )
        return None, "path BB: no chip fits"

    if have_wildcard != have_free_hit:
        only_chip = "wildcard" if have_wildcard else "free_hit"
        if is_biggest_fixture_pileup_before_boundary(gameweek, season, dbsession):
            return only_chip, "path BB: biggest pile-up before boundary"
        return None, "path BB: not the biggest pile-up, saving the last chip"

    if another_unusual_gameweek_within(
        gameweek, LOOKAHEAD_GAMEWEEKS, season, dbsession
    ):
        return (
            "wildcard",
            "path BB: another blank/double coming, squad overhaul justified",
        )
    return "free_hit", "path BB: isolated double, one-week fix"


def _path_b(
    squad: Squad,
    available_chip_types: list[str],
    gameweek: int,
    tag: str,
    season: str,
    dbsession: Session,
) -> tuple[str | None, str]:
    """Heuristic play - see chip_timing_heuristic.svg, Path B."""
    count = fixture_count(gameweek, season, dbsession)
    if count == NORMAL_GAMEWEEK_FIXTURE_COUNT:
        return None, "path B: normal gameweek"
    if count < NORMAL_GAMEWEEK_FIXTURE_COUNT:
        return _path_ba(squad, available_chip_types, gameweek, season, dbsession)
    return _path_bb(squad, available_chip_types, gameweek, tag, season, dbsession)


def simulate_chip_decisions(
    squad: Squad,
    gameweek_range: list[int],
    tag: str,
    season: str,
    dbsession: Session = session,
    chips_played: dict[int, str] | None = None,
) -> list[dict]:
    """Walk `gameweek_range` one gameweek at a time, applying the chip-timing
    flowchart (see chip_timing_heuristic.svg/.pdf) at each step. Returns one trace
    entry per gameweek - {"gameweek": gw, "chip_played": str | None, "reason":
    str} - the full trace rather than just the final answer, so a run can be
    inspected (which branch fired and why), not just which gameweek was ultimately
    chosen for each chip.

    `chips_played` seeds the walk with chips already used in gameweeks before
    `gameweek_range` - needed when this is called repeatedly over a rolling window
    (e.g. once per gameweek in replay_season.py) so a chip already played in an
    earlier call isn't suggested again. Defaults to none used yet.
    """
    chips_played = dict(chips_played) if chips_played else {}
    trace = []

    for gw in gameweek_range:
        used_this_half = {
            chip
            for played_gw, chip in chips_played.items()
            if chip_half(played_gw) == chip_half(gw)
        }
        available_chip_types = [c for c in CHIP_TYPES if c not in used_this_half]

        chip_played: str | None = None
        reason = "root: no chips available"

        if available_chip_types:
            weeks_remaining = weeks_until_chip_boundary(gw)
            if len(available_chip_types) > weeks_remaining:
                chip_played, reason = _path_a(
                    available_chip_types, gw, season, dbsession
                )
            else:
                chip_played, reason = _path_b(
                    squad, available_chip_types, gw, tag, season, dbsession
                )

        if chip_played is not None:
            chips_played[gw] = chip_played

        trace.append({"gameweek": gw, "chip_played": chip_played, "reason": reason})

    return trace


def suggest_chip_gameweeks(
    squad: Squad,
    gameweek_range: list[int],
    tag: str,
    season: str,
    dbsession: Session = session,
    chips_played: dict[int, str] | None = None,
) -> dict[str, int]:
    """Reduce simulate_chip_decisions()'s trace to the {chip_name: gameweek} shape
    construct_chip_dict (airsenal.scripts.fill_transfersuggestion_table) expects -
    -1 for any chip never triggered within gameweek_range, ready to pass straight
    into run_optimization()/run_mcts_optimization(). See simulate_chip_decisions
    for `chips_played`.
    """
    trace = simulate_chip_decisions(
        squad, gameweek_range, tag, season, dbsession, chips_played=chips_played
    )
    result = dict.fromkeys(CHIP_TYPES, -1)
    for entry in trace:
        chip_played = entry["chip_played"]
        if chip_played is not None:
            result[chip_played] = entry["gameweek"]
    return result
