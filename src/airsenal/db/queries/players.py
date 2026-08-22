"""Player lookups and listings."""

from sqlalchemy import case, or_, select
from sqlalchemy.orm import Session, selectinload

from airsenal.core.logging import get_logger
from airsenal.db.models import Player, PlayerAttributes, PlayerMapping, PlayerScore
from airsenal.db.queries.fixtures import (
    get_fixture_teams,
    get_fixtures_for_gameweeks,
    get_fixtures_for_player,
)
from airsenal.db.queries.gameweeks import (
    get_max_gameweek,
    is_future_gameweek,
    next_gameweek,
)
from airsenal.db.session import get_session
from airsenal.domain.season import CURRENT_SEASON

logger = get_logger(__name__)


# list_players is called once per candidate player per strategy, so an unguarded
# warning here fires ~90 times in a single `optimize transfers` run and buries
# everything else. The condition is a property of the database, not of the call, so
# say it once.
_warned_incomplete: set[tuple[str, int, int]] = set()


def _warn_incomplete_data(season: str, gameweek: int, latest: int) -> None:
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
    Query the player table by name, id, or opta_code, and return the player object
    (or None).

    NOTE the player_id that can be passed as an argument here is NOT
    guaranteed to be the id for that player in the FPL API. The one here
    is the entry (primary key) in our database.
    Use the function get_player_from_api_id() to find the player corresponding
    to the FPL API ID.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    # ID field match
    if isinstance(player_name_or_id, str) and player_name_or_id.isdigit():
        player_name_or_id = int(player_name_or_id)

    if isinstance(player_name_or_id, int):
        if p := dbsession.scalars(
            select(Player).where(Player.player_id == player_name_or_id).limit(1)
        ).first():
            return p
        # failed to find player by ID
        return None

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

    if p := dbsession.scalars(
        select(Player).where(Player.display_name == player_name_or_id).limit(1)
    ).first():
        return p

    # No match found
    return None


def get_player_from_api_id(
    api_id: int, dbsession: Session | None = None
) -> Player | None:
    """
    Query the database and return the player with corresponding attribute fpl_api_id.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if p := dbsession.scalars(
        select(Player).where(Player.fpl_api_id == api_id).limit(1)
    ).first():
        return p
    logger.warning("Unable to find player with fpl_api_id %s", api_id)
    return None


def get_player_name(player_id: int, dbsession: Session | None = None) -> str | None:
    """
    Lookup player name, for human readability.
    """
    if p := get_player(player_id, dbsession):
        return str(p)
    logger.warning("Unknown player_id %s", player_id)
    return None


def get_player_id(player_name: str, dbsession: Session | None = None) -> int | None:
    if p := get_player(player_name, dbsession):
        return p.player_id
    logger.warning("Unknown player_name %s", player_name)
    return None


def list_players(
    position: str = "all",
    team: str = "all",
    order_by: str = "price",
    season: str = CURRENT_SEASON,
    gameweek: int | None = None,
    dbsession: Session | None = None,
) -> list[Player]:
    """
    Print list of players and return a list of player_ids.
    """
    gameweek = next_gameweek() if gameweek is None else gameweek
    dbsession = dbsession if dbsession is not None else get_session()
    # if trying to get players from after DB has filled, return most recent players
    if season == CURRENT_SEASON:
        last_pa = dbsession.scalars(
            select(PlayerAttributes)
            .where(PlayerAttributes.season == season)
            .order_by(PlayerAttributes.gameweek.desc())
            .limit(1)
        ).first()
        if last_pa and gameweek > last_pa.gameweek:
            _warn_incomplete_data(season, gameweek, last_pa.gameweek)
            gameweek = last_pa.gameweek

    gameweeks = [gameweek]
    # check if the team (or all teams) play in the specified gameweek, if not
    # attributes might be missing
    fixtures = get_fixture_teams(
        get_fixtures_for_gameweeks([gameweek], season=season, dbsession=dbsession)
    )
    teams_with_fixture = {t for fixture in fixtures for t in fixture}

    if (team == "all" and len(teams_with_fixture) < 20) or (
        team != "all" and team not in teams_with_fixture
    ):
        # check neighbouring gameweeks to get all 20 teams/specified team
        gws_to_try = [gameweek - 1, gameweek + 1, gameweek - 2, gameweek + 2]
        max_gw = get_max_gameweek(season, dbsession)
        gws_to_try = [gw for gw in gws_to_try if gw > 0 and gw <= max_gw]

        for gw in gws_to_try:
            fixtures = get_fixture_teams(
                get_fixtures_for_gameweeks([gw], season=season, dbsession=dbsession)
            )
            new_teams = [t for fixture in fixtures for t in fixture]

            if team == "all" and any(t not in teams_with_fixture for t in new_teams):
                # this gameweek has some teams we haven't seen before
                gameweeks.append(gw)
                for t in new_teams:
                    teams_with_fixture.add(t)
                if len(teams_with_fixture) == 20:
                    break

            elif team != "all" and team in new_teams:
                # this gameweek has the team we're looking for
                gameweeks.append(gw)
                break

    query = select(PlayerAttributes).where(
        PlayerAttributes.season == season,
        PlayerAttributes.gameweek.in_(gameweeks),
    )
    if team != "all":
        query = query.where(PlayerAttributes.team == team)
    if position != "all":
        query = query.where(PlayerAttributes.position == position)
    else:
        # exclude managers
        query = query.where(PlayerAttributes.position != "MNG")
    if len(gameweeks) > 1:
        # Sort query results by order of gameweeks - i.e. make sure the input
        # query gameweek comes first.
        _whens = {gw: i for i, gw in enumerate(gameweeks)}
        sort_order = case(_whens, value=PlayerAttributes.gameweek)
        query = query.order_by(sort_order)
    if order_by == "price":
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
        if len(gameweeks) == 1 or order_by != "price":
            logger.debug("%s %s %s %s", pa.player, pa.team, pa.position, pa.price)
    if len(gameweeks) > 1 and order_by == "price":
        # Query sorted by gameweek first, so need to do a final sort here to
        # get final price order if more than one gameweek queried.
        sort_players = sorted(
            zip(prices, players, strict=False), reverse=True, key=lambda p: p[0]
        )
        for price, player in sort_players:
            logger.debug("%s %s", player, price)
        players = [p for _, p in sort_players]
    return players


def get_player_attributes(
    player_name_or_id: str | int,
    season: str = CURRENT_SEASON,
    gameweek: int | None = None,
    dbsession: Session | None = None,
) -> PlayerAttributes | None:
    """
    Get a player's attributes for a given gameweek in a given season.
    """
    gameweek = next_gameweek() if gameweek is None else gameweek
    dbsession = dbsession if dbsession is not None else get_session()
    if isinstance(player_name_or_id, str) and player_name_or_id.isdigit():
        player_id = int(player_name_or_id)
    elif isinstance(player_name_or_id, int):
        player_id = player_name_or_id
    elif isinstance(player_name_or_id, str):
        player = get_player(player_name_or_id)
        if player:
            player_id = player.player_id
        else:
            return None
    return dbsession.scalars(
        select(PlayerAttributes)
        .where(
            PlayerAttributes.season == season,
            PlayerAttributes.gameweek == gameweek,
            PlayerAttributes.player_id == player_id,
        )
        .limit(1)
    ).first()


def get_next_fixture_for_player(
    player: Player | str | int,
    season: str = CURRENT_SEASON,
    gameweek: int | None = None,
    dbsession: Session | None = None,
) -> str:
    """
    Get a players next fixture as a string, for easy displaying.
    """
    gameweek = next_gameweek() if gameweek is None else gameweek
    dbsession = dbsession if dbsession is not None else get_session()
    # given a player name or id, convert to player object
    if isinstance(player, str | int):
        maybe_player = get_player(player, dbsession)
        if not maybe_player:
            logger.warning("Couldn't find player %s in database", player)
            return ""
        player = maybe_player
    team = player.team(season, gameweek)
    fixtures_for_player = get_fixtures_for_player(player, season, [gameweek], dbsession)
    output_string = ""
    for fixture in fixtures_for_player:
        if fixture.home_team == team:
            output_string += fixture.away_team + " (h)"
        else:
            output_string += fixture.home_team + " (a)"
        output_string += ", "
    return output_string[:-2]


def get_max_matches_per_player(
    position: str = "all",
    season: str = CURRENT_SEASON,
    gameweek: int | None = None,
    dbsession: Session | None = None,
) -> int:
    """
    Can be used e.g. in bpl_interface.get_player_history_df
    to help avoid a ragged dataframe.
    """
    gameweek = next_gameweek() if gameweek is None else gameweek
    dbsession = dbsession if dbsession is not None else get_session()
    players = list_players(
        position=position, season=season, gameweek=gameweek, dbsession=dbsession
    )
    player_ids = [p.player_id for p in players if p.player_id is not None]
    if not player_ids:
        return 0

    scores = dbsession.scalars(
        select(PlayerScore)
        .options(selectinload(PlayerScore.fixture))
        .where(PlayerScore.player_id.in_(player_ids))
    ).all()

    matches_per_player = dict.fromkeys(player_ids, 0)
    for score in scores:
        if score.fixture is None or score.player_id is None:
            continue
        if not is_future_gameweek(
            score.fixture.season,
            score.fixture.gameweek,
            current_season=season,
            next_gameweek=gameweek,
        ):
            matches_per_player[score.player_id] += 1

    return max(matches_per_player.values(), default=0)
