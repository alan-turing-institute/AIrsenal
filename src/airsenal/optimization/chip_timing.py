"""
When to play each chip, decided by rules about the fixtures rather than by search.

The rules look at each gameweek of a window in turn, holding the squad as it is:

- More chips left than gameweeks to play them in before they expire: play one
  now, choosing by whether the gameweek is blank, double or neither.
- A blank that leaves the squad unable to field eleven: free hit, or a wildcard
  if the blank is severe or the wildcard is about to expire.
- A double: bench boost if nearly the whole squad plays twice, triple captain if
  the captain does and has a home game, otherwise a wildcard or free hit,
  depending on what is left and what else is coming.

A search offered a chip in any gameweek plays every one of them early, because
a plan's score gives nothing for a chip kept back. Deciding the gameweeks here
and pinning them avoids that.
"""

from collections import Counter
from collections.abc import Iterable

from sqlalchemy.orm import Session

from airsenal.core.copy import fastcopy
from airsenal.db.models import Fixture
from airsenal.db.queries.fixtures import get_fixtures_for_gameweeks
from airsenal.db.queries.gameweeks import get_max_gameweek
from airsenal.db.queries.players import get_player
from airsenal.db.session import get_session
from airsenal.game.chips import FIRST_HALF_LAST_GAMEWEEK, comes_twice, season_half
from airsenal.game.enums import Chip, Position
from airsenal.game.season import CURRENT_SEASON
from airsenal.optimization.moves import ChipGameweeks
from airsenal.squad.lineup import FORMATIONS, pick_captains
from airsenal.squad.player import SquadPlayer
from airsenal.squad.squad import Squad

NORMAL_GAMEWEEK_FIXTURES = 10
# More doubled players than this makes a bench boost worth it: fifteen is all of
# them.
BENCH_BOOST_DOUBLED_PLAYERS = 14
# A blank leaving this few players available is bad enough to wildcard for.
SEVERE_BLANK_AVAILABLE_PLAYERS = 9
# A wildcard this close to expiring may as well be played on a blank.
URGENT_GAMEWEEKS_LEFT = 3
# How far ahead another blank or double makes a wildcard better than a free hit.
LOOKAHEAD_GAMEWEEKS = 3

# The order to play chips in when one is forced, by the kind of gameweek.
_FORCED_BLANK = (Chip.FREE_HIT, Chip.WILDCARD, Chip.TRIPLE_CAPTAIN, Chip.BENCH_BOOST)
_FORCED_DOUBLE = (Chip.WILDCARD, Chip.FREE_HIT, Chip.BENCH_BOOST, Chip.TRIPLE_CAPTAIN)
_FORCED_NORMAL = (Chip.TRIPLE_CAPTAIN, Chip.BENCH_BOOST, Chip.FREE_HIT, Chip.WILDCARD)


class _Season:
    """The fixtures of one season, counted once and looked up by gameweek."""

    def __init__(self, season: str, dbsession: Session) -> None:
        self.season = season
        self.dbsession = dbsession
        self.last_gameweek = get_max_gameweek(season, dbsession=dbsession)
        self._fixtures: dict[int, list[Fixture]] = {}

    def fixtures(self, gameweek: int) -> list[Fixture]:
        if gameweek not in self._fixtures:
            self._fixtures[gameweek] = get_fixtures_for_gameweeks(
                [gameweek], season=self.season, dbsession=self.dbsession
            )
        return self._fixtures[gameweek]

    def n_fixtures(self, gameweek: int) -> int:
        return len(self.fixtures(gameweek))

    def expires(self, chip: Chip, gameweek: int) -> int:
        """The last gameweek the chip available in `gameweek` can be played in."""
        if comes_twice(chip, self.season) and season_half(gameweek) == 1:
            return FIRST_HALF_LAST_GAMEWEEK
        return self.last_gameweek

    def is_biggest_before(self, gameweek: int, last: int) -> bool:
        """Whether no gameweek from this one to `last` has more fixtures."""
        most = max(self.n_fixtures(later) for later in range(gameweek, last + 1))
        return self.n_fixtures(gameweek) == most

    def unusual_within(self, gameweek: int, n_gameweeks: int) -> bool:
        """Whether one of the next `n_gameweeks` gameweeks is a blank or a double."""
        return any(
            self.n_fixtures(later) != NORMAL_GAMEWEEK_FIXTURES
            for later in range(gameweek + 1, gameweek + 1 + n_gameweeks)
            if later <= self.last_gameweek
        )


def _fixture_counts(fixtures: Iterable[Fixture]) -> Counter[str]:
    """How many fixtures each club has."""
    counts: Counter[str] = Counter()
    for fixture in fixtures:
        counts[fixture.home_team] += 1
        counts[fixture.away_team] += 1
    return counts


def _available_players(
    squad: Squad,
    root_gameweek: int,
    gameweek: int,
    fixtures: list[Fixture],
    season: str,
    dbsession: Session,
) -> list[SquadPlayer]:
    """The squad's players who have a fixture and are expected to be fit for it."""
    playing = _fixture_counts(fixtures)
    available = []
    for candidate in squad.players:
        if not playing[candidate.team]:
            continue
        player = get_player(candidate.player_id, dbsession=dbsession)
        if player is None or player.is_injured_or_suspended(
            season, root_gameweek, gameweek
        ):
            continue
        available.append(candidate)
    return available


def can_field_eleven(players: Iterable[SquadPlayer]) -> bool:
    """Whether some formation can be filled from `players`."""
    counts = Counter(Position(p.position) for p in players)
    if counts[Position.GK] < 1:
        return False
    return any(
        counts[Position.DEF] >= n_def
        and counts[Position.MID] >= n_mid
        and counts[Position.FWD] >= n_fwd
        for n_def, n_mid, n_fwd in FORMATIONS
    )


def _captain_doubled_at_home(
    squad: Squad, tag: str, gameweek: int, fixtures: list[Fixture]
) -> bool:
    """Whether the captain has two fixtures this gameweek, one of them at home."""
    players = fastcopy(squad).players
    for player in players:
        player.calc_predicted_points(tag)
    pick_captains(players, tag, gameweek)
    captain = next(p for p in players if p.is_captain)
    theirs = [f for f in fixtures if captain.team in (f.home_team, f.away_team)]
    return len(theirs) >= 2 and any(f.home_team == captain.team for f in theirs)


def _forced(available: list[Chip], n_fixtures: int) -> tuple[Chip, str]:
    if len(available) == 1:
        return available[0], "the only chip left, and it is about to expire"
    if n_fixtures < NORMAL_GAMEWEEK_FIXTURES:
        order = _FORCED_BLANK
    elif n_fixtures > NORMAL_GAMEWEEK_FIXTURES:
        order = _FORCED_DOUBLE
    else:
        order = _FORCED_NORMAL
    chip = next(chip for chip in order if chip in available)
    return (
        chip,
        f"more chips left than gameweeks to play them in ({n_fixtures} fixtures)",
    )


def _blank(
    available: list[Chip],
    n_available_players: int,
    can_field: bool,
    gameweeks_left: int,
) -> tuple[Chip | None, str]:
    if can_field:
        return None, "a blank, but the squad can still field eleven"
    if Chip.FREE_HIT in available:
        return Chip.FREE_HIT, "a blank the squad cannot field eleven for"
    if Chip.WILDCARD in available and (
        n_available_players <= SEVERE_BLANK_AVAILABLE_PLAYERS
        or gameweeks_left < URGENT_GAMEWEEKS_LEFT
    ):
        return Chip.WILDCARD, "a severe blank, or a wildcard about to expire"
    return None, "a blank, with no chip to answer it"


def _double(
    available: list[Chip],
    squad: Squad,
    tag: str,
    gameweek: int,
    fixtures: list[Fixture],
    fixtures_by_season: _Season,
) -> tuple[Chip | None, str]:
    playing = _fixture_counts(fixtures)
    n_doubled = sum(1 for p in squad.players if playing[p.team] >= 2)
    if Chip.BENCH_BOOST in available and n_doubled > BENCH_BOOST_DOUBLED_PLAYERS:
        return Chip.BENCH_BOOST, "a double for the whole squad"
    if Chip.TRIPLE_CAPTAIN in available and _captain_doubled_at_home(
        squad, tag, gameweek, fixtures
    ):
        return Chip.TRIPLE_CAPTAIN, "a double for the captain, one game at home"

    def biggest(chip: Chip) -> bool:
        return fixtures_by_season.is_biggest_before(
            gameweek, fixtures_by_season.expires(chip, gameweek)
        )

    rebuilds = [c for c in (Chip.WILDCARD, Chip.FREE_HIT) if c in available]
    if not rebuilds:
        boosts = [c for c in (Chip.BENCH_BOOST, Chip.TRIPLE_CAPTAIN) if c in available]
        if boosts and biggest(boosts[0]):
            return boosts[0], "the biggest double left, and no wildcard or free hit"
        return None, "a double, with no chip that fits it"
    if len(rebuilds) == 1:
        if biggest(rebuilds[0]):
            return rebuilds[0], "the biggest double before the chip expires"
        return None, "a double, but not the biggest: keeping the last squad chip"
    if fixtures_by_season.unusual_within(gameweek, LOOKAHEAD_GAMEWEEKS):
        return Chip.WILDCARD, "a double with another blank or double close behind"
    return Chip.FREE_HIT, "a double on its own, for one gameweek"


def decide_chips(
    squad: Squad,
    available: Iterable[Chip],
    gameweeks: list[int],
    tag: str,
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> list[tuple[int, Chip | None, str]]:
    """
    Walk the window a gameweek at a time, deciding whether to play a chip in each.

    Players' fitness is read as at the first gameweek of the window, which is all
    a replay of a past season is allowed to know.

    Args:
        available: The chips the entry has left going into the window. Those
            the season gives twice come back at the start of the second half.

    Returns:
        (gameweek, chip or None, why) for each gameweek.
    """
    dbsession = get_session(dbsession)
    fixtures_by_season = _Season(season, dbsession)
    remaining = set(available)
    decisions: list[tuple[int, Chip | None, str]] = []
    for gameweek in gameweeks:
        if gameweek == FIRST_HALF_LAST_GAMEWEEK + 1:
            remaining |= {chip for chip in Chip if comes_twice(chip, season)}
        chips_left = [chip for chip in Chip if chip in remaining]
        chip, why = _decide(
            squad,
            chips_left,
            gameweeks[0],
            tag,
            gameweek,
            fixtures_by_season,
        )
        if chip is not None:
            remaining.discard(chip)
        decisions.append((gameweek, chip, why))
    return decisions


def _decide(
    squad: Squad,
    available: list[Chip],
    root_gameweek: int,
    tag: str,
    gameweek: int,
    fixtures_by_season: _Season,
) -> tuple[Chip | None, str]:
    if not available:
        return None, "no chips left"
    fixtures = fixtures_by_season.fixtures(gameweek)
    n_fixtures = len(fixtures)
    last = min(fixtures_by_season.expires(chip, gameweek) for chip in available)
    expiring = [c for c in available if fixtures_by_season.expires(c, gameweek) == last]
    if len(expiring) > last - gameweek:
        return _forced(expiring, n_fixtures)
    if n_fixtures == NORMAL_GAMEWEEK_FIXTURES:
        return None, "a normal gameweek"
    if n_fixtures < NORMAL_GAMEWEEK_FIXTURES:
        players = _available_players(
            squad,
            root_gameweek,
            gameweek,
            fixtures,
            fixtures_by_season.season,
            fixtures_by_season.dbsession,
        )
        return _blank(
            available,
            len(players),
            can_field_eleven(players),
            fixtures_by_season.expires(Chip.WILDCARD, gameweek) - gameweek,
        )
    return _double(available, squad, tag, gameweek, fixtures, fixtures_by_season)


def chip_gameweeks(decisions: Iterable[tuple[int, Chip | None, str]]) -> ChipGameweeks:
    """
    The gameweek to play each chip in, from `decide_chips`; -1 for one it leaves.

    A chip decided twice in the window, either side of the split, is pinned to
    the first gameweek; the next window decides the second.
    """
    gameweek_for: dict[Chip, int] = {}
    for gameweek, chip, _ in decisions:
        if chip is not None:
            gameweek_for.setdefault(chip, gameweek)
    return ChipGameweeks(
        wildcard=gameweek_for.get(Chip.WILDCARD, -1),
        free_hit=gameweek_for.get(Chip.FREE_HIT, -1),
        triple_captain=gameweek_for.get(Chip.TRIPLE_CAPTAIN, -1),
        bench_boost=gameweek_for.get(Chip.BENCH_BOOST, -1),
    )
