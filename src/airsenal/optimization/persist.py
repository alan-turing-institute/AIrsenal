"""
Writing an optimised plan into the database.

These are database writes, but they take `Plan` and `Squad` arguments, so
they cannot live in db/ without the data layer having to know what a plan
is. They sit at the top of optimization/ instead, where both are already in
scope.
"""

from datetime import datetime

from sqlalchemy.orm import Session

from airsenal.core.enums import Chip
from airsenal.core.logging import get_logger
from airsenal.core.season import CURRENT_SEASON
from airsenal.db.models import TransferSuggestion
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.db.queries.players import get_player
from airsenal.db.queries.transactions import add_transaction
from airsenal.db.session import get_session
from airsenal.optimization.plan import Plan
from airsenal.squad.squad import Squad

logger = get_logger(__name__)


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
            for player in players:
                ts = TransferSuggestion()
                ts.player_id = player
                ts.in_or_out = in_or_out
                ts.gameweek = outcome.gameweek
                ts.points_gain = points_gain
                ts.timestamp = timestamp
                ts.season = season
                ts.fpl_team_id = fpl_team_id
                ts.chip_played = str(outcome.chip) if outcome.chip else None
                dbsession.add(ts)
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
    for player_id in outcome.players_out:
        price = starting_squad.get_sell_price_for_player(
            player_id, gameweek=fill_gw, dbsession=dbsession
        )
        add_transaction(
            player_id,
            fill_gw,
            -1,
            price,
            season,
            tag,
            free_hit,
            fpl_team_id,
            time,
            dbsession,
        )
    for player_id in outcome.players_in:
        if player := get_player(player_id, dbsession=dbsession):
            buy_price = player.price(season, fill_gw)
            if buy_price is None:
                # Transaction.price is not nullable, so recording the transfer
                # anyway fails at flush time with an opaque integrity error.
                logger.warning(
                    "No %s price for player %s, skipping transaction",
                    season,
                    player_id,
                )
                continue
            add_transaction(
                player_id,
                fill_gw,
                1,
                buy_price,
                season,
                tag,
                free_hit,
                fpl_team_id,
                time,
                dbsession,
            )
        else:
            logger.warning("Failed to find player %s in db for transaction", player_id)


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
    timestamp = str(datetime.now())
    score = squad.get_expected_points(gameweek, tag)
    for player in squad.players:
        ts = TransferSuggestion()
        ts.player_id = player.player_id
        ts.in_or_out = 1
        ts.gameweek = gameweek
        ts.points_gain = score
        ts.timestamp = timestamp
        ts.season = season
        ts.fpl_team_id = fpl_team_id
        ts.chip_played = None
        dbsession.add(ts)
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
    free_hit = 0
    time = datetime.now().isoformat()
    for player in squad.players:
        add_transaction(
            player.player_id,
            gameweek,
            1,
            player.purchase_price,
            season,
            tag,
            free_hit,
            fpl_team_id,
            time,
            dbsession,
        )
