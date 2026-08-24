"""Reading predicted points back out of the database."""

from collections.abc import Iterable, Sequence
from operator import itemgetter

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from airsenal.core.caching import cache_ignoring_session
from airsenal.core.logging import get_logger
from airsenal.core.season import CURRENT_SEASON
from airsenal.db.models import Fixture, Player, PlayerPrediction, TransferSuggestion
from airsenal.db.queries.gameweeks import get_max_gameweek
from airsenal.db.queries.players import get_player, list_players
from airsenal.db.session import get_session

logger = get_logger(__name__)


def get_predicted_points_for_player(
    player: Player | str | int,
    tag: str,
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> dict[int, float]:
    """
    Query the player prediction table for a given player.
    Return a dict, keyed by gameweek.

    This is the inner loop of the transfer optimisation - it is called once per
    candidate player per candidate squad - so the answer is cached. The cache is
    keyed on the player id rather than on the Player object, and does not
    include the session: see airsenal.core.caching for why.

    An `int` is taken to be a player id and is not looked up first. That check
    used to happen on every call, which cost a database round trip per candidate
    player and roughly doubled the time of a transfer optimisation, to validate
    an id the caller had just read out of the database.
    """
    if isinstance(player, int):
        player_id = player
    elif isinstance(player, str):
        dbsession = dbsession if dbsession is not None else get_session()
        maybe_player = get_player(player, dbsession=dbsession)
        if maybe_player is None:
            msg = f"Couldn't find player {player} in database"
            raise ValueError(msg)
        player_id = maybe_player.player_id
    else:
        player_id = player.player_id
    return _predicted_points_for_player_id(player_id, tag, season, dbsession=dbsession)


@cache_ignoring_session(maxsize=4096)
def _predicted_points_for_player_id(
    player_id: int,
    tag: str,
    season: str,
    dbsession: Session | None = None,
) -> dict[int, float]:
    dbsession = dbsession if dbsession is not None else get_session()
    pps = dbsession.scalars(
        select(PlayerPrediction)
        .options(selectinload(PlayerPrediction.fixture))
        .where(
            PlayerPrediction.fixture.has(Fixture.season == season),
            PlayerPrediction.player_id == player_id,
            PlayerPrediction.tag == tag,
        )
    ).all()
    ppdict = {}
    for prediction in pps:
        # there is one prediction per fixture.
        # for double gameweeks, we need to add the two together
        gameweek = prediction.fixture.gameweek
        if gameweek is None:
            logger.warning(
                "Player %s has no gameweek for fixture %s",
                player_id,
                prediction.fixture,
            )
            continue
        if gameweek not in ppdict:
            ppdict[gameweek] = 0.0
        ppdict[gameweek] += prediction.predicted_points
    # we still need to fill in zero for gameweeks that they're not playing.
    max_gw = get_max_gameweek(season, dbsession=dbsession)
    for gw in range(1, max_gw + 1):
        if gw not in ppdict:
            ppdict[gw] = 0.0
    return ppdict


def get_predicted_points(
    gameweeks: Iterable[int],
    tag: str,
    position: str = "all",
    team: str = "all",
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> list[tuple[Player, float]]:
    """
    Query the player_prediction table with selections, return
    list of tuples (player_id, predicted_points) ordered by predicted_points.
    Points are summed over the gameweeks given; callers wanting one gameweek
    pass `[gameweek]`.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    gameweeks = list(gameweeks)
    players = list_players(
        position,
        team,
        season=season,
        gameweek=gameweeks[0],
        dbsession=dbsession,
    )
    player_ids = [p.player_id for p in players]
    points_by_player = dict.fromkeys(player_ids, 0.0)

    if player_ids:
        rows = dbsession.execute(
            select(
                PlayerPrediction.player_id,
                Fixture.gameweek,
                PlayerPrediction.predicted_points,
            )
            .join(Fixture, PlayerPrediction.fixture_id == Fixture.fixture_id)
            .where(
                PlayerPrediction.player_id.in_(player_ids),
                PlayerPrediction.tag == tag,
                Fixture.season == season,
                Fixture.gameweek.in_(gameweeks),
            )
        ).all()
        for row in rows:
            if row.player_id is not None:
                points_by_player[row.player_id] += row.predicted_points

    output_list = [(p, points_by_player.get(p.player_id, 0.0)) for p in players]
    output_list.sort(key=itemgetter(1), reverse=True)
    return output_list


def get_transfer_suggestions(
    dbsession: Session,
    gameweek: int | None = None,
    season: str | None = None,
    fpl_team_id: int | None = None,
) -> Sequence[TransferSuggestion]:
    """
    The rows of the most recent transfer suggestion, optionally filtered.

    One row per player in-or-out per gameweek; rows belonging to the same
    suggested plan share a timestamp, which is how the latest one is picked out.
    """
    last_timestamp = dbsession.scalars(
        select(TransferSuggestion.timestamp).order_by(
            TransferSuggestion.timestamp.desc()
        )
    ).first()
    if last_timestamp is None:
        return []
    query = select(TransferSuggestion).where(
        TransferSuggestion.timestamp == last_timestamp
    )
    if gameweek:
        query = query.where(TransferSuggestion.gameweek == gameweek)
    if season:
        query = query.where(TransferSuggestion.season == season)
    if fpl_team_id:
        query = query.where(TransferSuggestion.fpl_team_id == fpl_team_id)

    return dbsession.scalars(query.order_by(TransferSuggestion.gameweek)).all()
