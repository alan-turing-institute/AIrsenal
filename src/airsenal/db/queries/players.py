"""Player lookups and listings."""

import re
import unicodedata

from sqlalchemy import case, or_, select
from sqlalchemy.orm import Session, selectinload

from airsenal.core.caching import cache_ignoring_session
from airsenal.core.logging import get_logger
from airsenal.db.models import Player, PlayerAttributes, PlayerMapping
from airsenal.db.queries.fixtures import (
    get_fixture_teams,
    get_fixtures_for_gameweeks,
)
from airsenal.db.queries.gameweeks import (
    get_max_gameweek,
    next_gameweek,
)
from airsenal.db.session import get_session
from airsenal.game.enums import Position
from airsenal.game.season import CURRENT_SEASON

logger = get_logger(__name__)


# list_players is called once per candidate player per strategy, so this warning
# is given once per condition rather than once per call.
_warned_incomplete: set[tuple[str, int, int]] = set()


def _warn_incomplete_data(gameweek: int, season: str, latest: int) -> None:
    key = (season, gameweek, latest)
    if key in _warned_incomplete:
        return
    _warned_incomplete.add(key)
    logger.warning(
        "Incomplete data in DB for %s GW%s, returning players from GW%s. "
        "Run 'airsenal db update' to fetch the latest player attributes.",
        season,
        gameweek,
        latest,
    )


def get_player(
    player_name_or_id: str | int,
    dbsession: Session | None = None,
) -> Player | None:
    """
    Look a player up by name, id, or opta_code. None if there is no match.

    An integer is this database's primary key, *not* the player's FPL API id.
    Use `get_player_from_api_id` for that.
    """
    dbsession = get_session(dbsession)
    # ID field match
    if isinstance(player_name_or_id, str) and player_name_or_id.isdigit():
        player_name_or_id = int(player_name_or_id)

    if isinstance(player_name_or_id, int):
        return dbsession.scalars(
            select(Player).where(Player.player_id == player_name_or_id).limit(1)
        ).first()

    # Name or Opta code match
    if p := dbsession.scalars(
        select(Player)
        .where(
            or_(
                Player.name == player_name_or_id,
                Player.opta_code == player_name_or_id,
            )
        )
        .limit(1)
    ).first():
        return p

    # Alternative name match
    if mapping := dbsession.scalars(
        select(PlayerMapping)
        .where(PlayerMapping.alt_name == player_name_or_id)
        .limit(1)
    ).first():
        return dbsession.scalars(
            select(Player).where(Player.player_id == mapping.player_id).limit(1)
        ).first()

    return dbsession.scalars(
        select(Player).where(Player.display_name == player_name_or_id).limit(1)
    ).first()


# Letters NFKD leaves whole, so a folded comparison has to spell them out.
# Transfermarkt and the FPL API disagree about every one of these.
_TRANSLITERATIONS = str.maketrans(
    {"ł": "l", "ø": "o", "đ": "d", "ð": "d", "þ": "th", "ß": "ss", "æ": "ae", "œ": "oe"}
)

# How much of a given name has to be written before it may stand in for a longer
# one, so that "Dan" matches "Daniel" but "Jo" does not match everyone called Joe,
# Jonny and Joao at once.
SHORTEST_GIVEN_NAME = 3


def fold_name(name: str) -> str:
    """A name reduced to unaccented lower case, for comparing two spellings."""
    decomposed = unicodedata.normalize(
        "NFKD", name.casefold().translate(_TRANSLITERATIONS)
    )
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def name_tokens(name: str) -> frozenset[str]:
    """The words of a folded name, split on everything that is not alphanumeric."""
    return frozenset(word for word in re.split(r"[^0-9a-z]+", fold_name(name)) if word)


def _stands_in_for(wanted: frozenset[str], known: frozenset[str]) -> bool:
    """Whether every wanted word is in `known`, or shortens a word that is."""
    return all(
        word in known
        or (
            len(word) >= SHORTEST_GIVEN_NAME
            and any(other.startswith(word) for other in known)
        )
        for word in wanted
    )


@cache_ignoring_session(maxsize=1)
def _name_tokens_by_player(
    dbsession: Session | None = None,
) -> tuple[tuple[int, frozenset[str]], ...]:
    """
    One (player id, folded words) pair for every name every player goes by.

    Rebuilt from scratch, so anything that adds a player has to call
    `clear_query_caches`. Ids rather than Players: a cached answer outlives the
    session that produced it.
    """
    dbsession = get_session(dbsession)
    names = [
        (player.player_id, name_tokens(name))
        for player in dbsession.scalars(select(Player))
        for name in (player.name, player.display_name)
        if name
    ]
    names += [
        (mapping.player_id, name_tokens(mapping.alt_name))
        for mapping in dbsession.scalars(select(PlayerMapping))
    ]
    return tuple(names)


def get_player_by_similar_name(
    player_name: str, dbsession: Session | None = None
) -> Player | None:
    """
    Look a player up by a name spelled differently from any on file.

    Matches across the ways two sources disagree about the same person: accents
    ("Josko Gvardiol" for "Joško Gvardiol"), word order ("Ao Tanaka" for
    "Tanaka Ao"), the family names one source keeps and the other drops
    ("Matheus Cunha" for "Matheus Santos Carneiro da Cunha"), and shortened
    given names ("Dan Ballard" for "Daniel Ballard"). The best match wins,
    counted by how many words it has that the name asked for does not.

    This can be wrong, so it is for names that arrive from a scrape rather than
    from a person: `get_player` is the exact lookup. A name that matches two
    players equally well is no match, because guessing between them would file
    one player's absence against another.

    Returns:
        None if nothing matches, or if the closest match is a tie.
    """
    dbsession = get_session(dbsession)
    wanted = name_tokens(player_name)
    if not wanted:
        return None

    closest: dict[int, int] = {}
    for player_id, known in _name_tokens_by_player(dbsession=dbsession):
        if _stands_in_for(wanted, known):
            unasked_for = len(known - wanted)
            closest[player_id] = min(closest.get(player_id, unasked_for), unasked_for)
    if not closest:
        return None

    fewest = min(closest.values())
    matched = [
        player_id for player_id, unasked_for in closest.items() if unasked_for == fewest
    ]
    if len(matched) > 1:
        logger.warning(
            "%s matches %s players equally well, so matching none of them",
            player_name,
            len(matched),
        )
        return None
    return get_player(matched[0], dbsession=dbsession)


def get_player_from_api_id(
    api_id: int, dbsession: Session | None = None
) -> Player | None:
    """
    The player with this `fpl_api_id`, or None.

    A missing player is a warning and None, not an error - the FPL API lists
    players before they reach a database that was seeded earlier.
    """
    dbsession = get_session(dbsession)
    if p := dbsession.scalars(
        select(Player).where(Player.fpl_api_id == api_id).limit(1)
    ).first():
        return p
    logger.warning("Unable to find player with fpl_api_id %s", api_id)
    return None


# The `require_` lookups are for callers that cannot carry on without the player,
# such as `apply/` building a request for the FPL API.


def require_player(
    player_name_or_id: str | int, dbsession: Session | None = None
) -> Player:
    """As `get_player`, but a missing player raises rather than returning None."""
    player = get_player(player_name_or_id, dbsession=dbsession)
    if player is None:
        msg = f"Player {player_name_or_id} not found in database"
        raise ValueError(msg)
    return player


def require_player_from_api_id(api_id: int, dbsession: Session | None = None) -> Player:
    """As `get_player_from_api_id`, but a missing player raises."""
    player = get_player_from_api_id(api_id, dbsession=dbsession)
    if player is None:
        msg = f"Player with FPL API ID {api_id} not found in database"
        raise ValueError(msg)
    return player


def require_api_id(player: Player) -> int:
    """
    A player's `fpl_api_id`, raising when the database does not have one.

    Players seeded from historical data alone never had one, and the FPL API
    cannot be told about a player it has no id for.
    """
    if player.fpl_api_id is None:
        msg = f"Player {player} has no FPL API ID"
        raise ValueError(msg)
    return player.fpl_api_id


def get_player_name(player_id: int, dbsession: Session | None = None) -> str | None:
    """Look a player's name up from their id, for human readability."""
    if p := get_player(player_id, dbsession):
        return str(p)
    logger.warning("Unknown player_id %s", player_id)
    return None


def _latest_filled_gameweek(gameweek: int, season: str, dbsession: Session) -> int:
    """
    `gameweek`, or the latest gameweek with attributes if it is after that.

    Current season only: a past season is complete.
    """
    if season != CURRENT_SEASON:
        return gameweek
    last_pa = dbsession.scalars(
        select(PlayerAttributes)
        .where(PlayerAttributes.season == season)
        .order_by(PlayerAttributes.gameweek.desc())
        .limit(1)
    ).first()
    if last_pa is None or gameweek <= last_pa.gameweek:
        return gameweek
    if gameweek < next_gameweek():
        _warn_incomplete_data(gameweek, season, last_pa.gameweek)
    return last_pa.gameweek


def _teams_playing(gameweek: int, season: str, dbsession: Session) -> set[str]:
    fixtures = get_fixtures_for_gameweeks(
        [gameweek], season=season, dbsession=dbsession
    )
    return {t for fixture in get_fixture_teams(fixtures) for t in fixture}


def _gameweeks_covering(
    team: str, gameweek: int, season: str, dbsession: Session
) -> list[int]:
    """
    `gameweek`, then the neighbouring gameweeks needed for `team` to have a fixture.

    A team without a fixture in a gameweek may have no attributes for it, so the
    gameweeks up to two either side fill in for it. `team` may be "all", which
    needs all 20 teams.
    """
    gameweeks = [gameweek]
    teams_with_fixture = _teams_playing(gameweek, season, dbsession)
    if (team == "all" and len(teams_with_fixture) >= 20) or (
        team != "all" and team in teams_with_fixture
    ):
        return gameweeks

    max_gameweek = get_max_gameweek(season, dbsession=dbsession)
    for neighbour in [gameweek - 1, gameweek + 1, gameweek - 2, gameweek + 2]:
        if neighbour <= 0 or neighbour > max_gameweek:
            continue
        new_teams = _teams_playing(neighbour, season, dbsession)
        if team == "all" and not new_teams <= teams_with_fixture:
            # this gameweek has some teams we haven't seen before
            gameweeks.append(neighbour)
            teams_with_fixture.update(new_teams)
            if len(teams_with_fixture) == 20:
                break
        elif team != "all" and team in new_teams:
            gameweeks.append(neighbour)
            break
    return gameweeks


def list_players(
    position: str = "all",
    team: str = "all",
    gameweek: int | None = None,
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> list[Player]:
    """The players in a position and team at a gameweek, most expensive first."""
    gameweek = next_gameweek() if gameweek is None else gameweek
    dbsession = get_session(dbsession)
    # if trying to get players from after DB has filled, return most recent players
    gameweek = _latest_filled_gameweek(gameweek, season, dbsession)
    gameweeks = _gameweeks_covering(team, gameweek, season, dbsession)

    query = select(PlayerAttributes).where(
        PlayerAttributes.season == season,
        PlayerAttributes.gameweek.in_(gameweeks),
    )
    if team != "all":
        query = query.where(PlayerAttributes.team == team)
    if position != "all":
        query = query.where(PlayerAttributes.position == position)
    else:
        # "all" is all the positions AIrsenal models, which leaves out managers
        query = query.where(PlayerAttributes.position.in_(Position.modelled()))
    if len(gameweeks) > 1:
        # Sort query results by order of gameweeks - i.e. make sure the input
        # query gameweek comes first.
        _whens = {queried: i for i, queried in enumerate(gameweeks)}
        sort_order = case(_whens, value=PlayerAttributes.gameweek)
        query = query.order_by(sort_order)
    query = query.order_by(PlayerAttributes.price.desc())
    players = []
    prices = []
    seen_player_ids = set()
    for pa in dbsession.scalars(query.options(selectinload(PlayerAttributes.player))):
        # might have queried multiple gameweeks with same player returned
        # multiple times - only add if it's a new player
        if pa.player_id in seen_player_ids:
            continue
        seen_player_ids.add(pa.player_id)
        players.append(pa.player)
        prices.append(pa.price)
        if len(gameweeks) == 1:
            logger.debug("%s %s %s %s", pa.player, pa.team, pa.position, pa.price)
    if len(gameweeks) > 1:
        # Query sorted by gameweek first, so need to do a final sort here to
        # get final price order if more than one gameweek queried.
        sort_players = sorted(
            zip(prices, players, strict=True), reverse=True, key=lambda p: p[0]
        )
        for price, player in sort_players:
            logger.debug("%s %s", player, price)
        players = [p for _, p in sort_players]
    return players


def get_player_attributes(
    player_id: int,
    gameweek: int | None = None,
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> PlayerAttributes | None:
    """A player's attributes for one gameweek, by default the next one, or None."""
    gameweek = next_gameweek() if gameweek is None else gameweek
    dbsession = get_session(dbsession)
    return dbsession.scalars(
        select(PlayerAttributes)
        .where(
            PlayerAttributes.season == season,
            PlayerAttributes.gameweek == gameweek,
            PlayerAttributes.player_id == player_id,
        )
        .limit(1)
    ).first()
