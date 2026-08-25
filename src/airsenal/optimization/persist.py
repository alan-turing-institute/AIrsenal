"""
Writing an optimised plan into the database.

These are database writes, but they take `Plan` and `Squad` arguments, so
they cannot live in db/ without the data layer having to know what a plan
is. They sit at the top of optimization/ instead, where both are already in
scope.

The four public functions are two pairs - a plan and a from-scratch squad,
written to each of two tables - so each pair shares one private writer. A
from-scratch squad is the degenerate plan: every player in, nobody out.
"""

from collections.abc import Iterable
from datetime import datetime

from sqlalchemy.orm import Session

from airsenal.core.logging import get_logger
from airsenal.db.models import TransferSuggestion
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.db.queries.players import get_player
from airsenal.db.queries.transactions import add_transaction
from airsenal.db.session import get_session
from airsenal.game.enums import Chip
from airsenal.game.season import CURRENT_SEASON
from airsenal.optimization.plan import Plan
from airsenal.squad.squad import Squad

logger = get_logger(__name__)

# (player id, price) - what a Transaction row needs beyond the fields every row
# in one write shares.
type PricedPlayer = tuple[int, int]


def _add_suggestions(
    player_ids: Iterable[int],
    in_or_out: int,
    gameweek: int,
    points_gain: float,
    chip: Chip | None,
    season: str,
    fpl_team_id: int,
    timestamp: str,
    dbsession: Session,
) -> None:
    """Add one suggestion row per player. Does not commit."""
    for player_id in player_ids:
        suggestion = TransferSuggestion()
        suggestion.player_id = player_id
        suggestion.in_or_out = in_or_out
        suggestion.gameweek = gameweek
        suggestion.points_gain = points_gain
        suggestion.timestamp = timestamp
        suggestion.season = season
        suggestion.fpl_team_id = fpl_team_id
        suggestion.chip_played = str(chip) if chip else None
        dbsession.add(suggestion)


def _add_transactions(
    priced_players: Iterable[PricedPlayer],
    in_or_out: int,
    gameweek: int,
    season: str,
    tag: str,
    free_hit: int,
    fpl_team_id: int,
    time: str,
    dbsession: Session | None,
) -> None:
    """Add one transaction row per priced player."""
    for player_id, price in priced_players:
        add_transaction(
            player_id=player_id,
            gameweek=gameweek,
            in_or_out=in_or_out,
            price=price,
            season=season,
            tag=tag,
            free_hit=free_hit,
            fpl_team_id=fpl_team_id,
            time=time,
            dbsession=dbsession,
        )


def _buy_prices(
    player_ids: Iterable[int],
    season: str,
    gameweek: int,
    dbsession: Session,
) -> Iterable[PricedPlayer]:
    """What each player cost, skipping any the database cannot price."""
    for player_id in player_ids:
        player = get_player(player_id, dbsession=dbsession)
        if player is None:
            logger.warning("Failed to find player %s in db for transaction", player_id)
            continue
        price = player.price(season, gameweek)
        if price is None:
            # Transaction.price is not nullable, so recording the transfer
            # anyway fails at flush time with an opaque integrity error.
            logger.warning(
                "No %s price for player %s, skipping transaction", season, player_id
            )
            continue
        yield player_id, price


def fill_suggestion_table(
    baseline_score: float,
    best_plan: Plan,
    season: str,
    fpl_team_id: int,
    dbsession: Session | None = None,
) -> None:
    """
    Fill the optimized plan into the table
    """
    dbsession = dbsession if dbsession is not None else get_session()
    timestamp = str(datetime.now())
    points_gain = best_plan.total_score - baseline_score

    for outcome in best_plan.outcomes:
        for players, in_or_out in (
            (outcome.players_out, -1),
            (outcome.players_in, 1),
        ):
            _add_suggestions(
                players,
                in_or_out,
                gameweek=outcome.gameweek,
                points_gain=points_gain,
                chip=outcome.chip,
                season=season,
                fpl_team_id=fpl_team_id,
                timestamp=timestamp,
                dbsession=dbsession,
            )
    dbsession.commit()


def fill_transaction_table(
    starting_squad: Squad,
    best_plan: Plan,
    season: str,
    fpl_team_id: int,
    tag: str | None = None,
    dbsession: Session | None = None,
) -> None:
    """Add transactions from an optimised plan to the transactions table in the
    database. Used for simulating seasons only, for playing the current FPL season
    the transactions status is kept up to date with transfers using the FPL API.
    Only transfers from the first gameweek in the plan are added to the Transaction
    table - it's assumed the plan will be re-optimised after each week rather than
    sticking with the originally proposed future transfers.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    outcome = best_plan.outcomes[0]
    fill_gw = outcome.gameweek
    if tag is None:
        tag = f"AIrsenal{season}"
    free_hit = int(outcome.chip is Chip.FREE_HIT)
    time = datetime.now().isoformat()

    sold = (
        (
            player_id,
            starting_squad.get_sell_price_for_player(
                player_id, gameweek=fill_gw, dbsession=dbsession
            ),
        )
        for player_id in outcome.players_out
    )
    for priced_players, in_or_out in (
        (sold, -1),
        (_buy_prices(outcome.players_in, season, fill_gw, dbsession), 1),
    ):
        _add_transactions(
            priced_players,
            in_or_out,
            gameweek=fill_gw,
            season=season,
            tag=tag,
            free_hit=free_hit,
            fpl_team_id=fpl_team_id,
            time=time,
            dbsession=dbsession,
        )


def fill_initial_suggestion_table(
    squad: Squad,
    fpl_team_id: int,
    tag: str,
    season: str = CURRENT_SEASON,
    gameweek: int | None = None,
    dbsession: Session | None = None,
) -> None:
    """
    Fill an initial squad into the table
    """
    gameweek = next_gameweek() if gameweek is None else gameweek
    dbsession = dbsession if dbsession is not None else get_session()
    _add_suggestions(
        [player.player_id for player in squad.players],
        in_or_out=1,
        gameweek=gameweek,
        points_gain=squad.get_expected_points(gameweek, tag),
        chip=None,
        season=season,
        fpl_team_id=fpl_team_id,
        timestamp=str(datetime.now()),
        dbsession=dbsession,
    )
    dbsession.commit()


def fill_initial_transaction_table(
    squad: Squad,
    fpl_team_id: int,
    tag: str,
    season: str = CURRENT_SEASON,
    gameweek: int | None = None,
    dbsession: Session | None = None,
) -> None:
    """Add transactions from an initial squad optimisation to the transactions table
    in the database. Used for simulating seasons only, for playing the current FPL
    season the transactions status is kepts up to date with transfers using the FPL API.
    """
    gameweek = next_gameweek() if gameweek is None else gameweek
    dbsession = dbsession if dbsession is not None else get_session()
    _add_transactions(
        [(player.player_id, player.purchase_price) for player in squad.players],
        in_or_out=1,
        gameweek=gameweek,
        season=season,
        tag=tag,
        free_hit=0,
        fpl_team_id=fpl_team_id,
        time=datetime.now().isoformat(),
        dbsession=dbsession,
    )
