"""
functions to optimize the transfers for N weeks ahead
"""

from copy import deepcopy
from datetime import datetime

from curl_cffi import requests
from sqlalchemy import select
from sqlalchemy.orm import Session

from airsenal.core.enums import Chip, Position
from airsenal.core.logging import get_logger
from airsenal.db.models import (
    Fixture,
    PlayerPrediction,
    Transaction,
    TransferSuggestion,
)
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.db.queries.players import get_player
from airsenal.db.session import get_session
from airsenal.db.writes.transactions import add_transaction
from airsenal.domain.season import CURRENT_SEASON
from airsenal.fetch.fpl_api import FPLDataFetcher, get_fetcher
from airsenal.optimization.config import SubWeights
from airsenal.optimization.moves import (
    NO_CHIPS,
    ChipSchedule,
    GameweekChips,
    GameweekMove,
)
from airsenal.squad.squad import Squad, get_current_squad_from_api

logger = get_logger(__name__)

positions = list(Position.front_to_back())  # front-to-back

# Derived from SubWeights so there is one definition. The squad builder used to
# hard-code {"GK": 0.01, "Outfield": (0.4, 0.1, 0.02)} instead, so `optimize
# squad` and `optimize transfers` scored benches differently - unintentionally,
# and the docstrings advertised the other set again.
DEFAULT_SUB_WEIGHTS = SubWeights().as_dict()
MAX_FREE_TRANSFERS = 5  # changed in 24/25 season (not accounted for in replay season)
POINTS_HIT_COST = 4  # points lost per transfer beyond the free ones
DEFAULT_DISCOUNT = 14 / 15  # weight applied per gameweek into the future


def check_tag_valid(
    pred_tag, gameweek_range, season=CURRENT_SEASON, dbsession: Session | None = None
):
    """Check a prediction tag contains predictions for all the specified gameweeks."""
    # get unique gameweek and season values associated with pred_tag
    dbsession = dbsession if dbsession is not None else get_session()
    fixtures = dbsession.execute(
        select(Fixture.season, Fixture.gameweek)
        .join(PlayerPrediction)
        .where(PlayerPrediction.tag == pred_tag)
        .distinct()
    ).all()
    pred_seasons = [f[0] for f in fixtures]
    pred_gws = [f[1] for f in fixtures]

    season_ok = all(s == season for s in pred_seasons)
    gws_ok = all(gw in pred_gws for gw in gameweek_range)

    return season_ok and gws_ok


def calc_points_hit(
    move: GameweekMove, free_transfers: int, cost: int = POINTS_HIT_COST
) -> int:
    """
    Points lost for making more transfers than we have free.

    Wildcard and free hit rebuild the squad without a hit; the other two chips
    are played alongside ordinary transfers and are charged as usual.
    """
    if move.rebuilds_squad:
        return 0
    return max(0, cost * (move.n_transfers - free_transfers))


def calc_free_transfers(
    move: GameweekMove,
    prev_free_transfers: int,
    max_free_transfers: int = MAX_FREE_TRANSFERS,
) -> int:
    """
    We get one extra free transfer per week, unless we use a wildcard or
    free hit, but we can't have more than max_free_transfers. So we should only
    be able to return 1 to max_free_transfers.
    """
    if move.rebuilds_squad:
        return prev_free_transfers  # changed in 24/25 season, previously 1
    return max(1, min(max_free_transfers, 1 + prev_free_transfers - move.n_transfers))


def get_starting_squad(
    next_gw: int | None = None,
    season=CURRENT_SEASON,
    fpl_team_id=None,
    use_api=False,
    fetcher: FPLDataFetcher | None = None,
    dbsession: Session | None = None,
):
    """
    use the transactions table in the db, or the API if requested
    """
    fetcher = fetcher if fetcher is not None else get_fetcher()
    next_gw = next_gameweek() if next_gw is None else next_gw
    if use_api:
        if season != CURRENT_SEASON:
            msg = "Can only use API for current season and gameweek"
            raise RuntimeError(msg)
        if season == CURRENT_SEASON and next_gw != next_gameweek():
            msg = "Can only use API for current season and gameweek"
            raise RuntimeError(msg)
        if not fpl_team_id:
            msg = "Please specify fpl_team_id to get current squad from API"
            raise RuntimeError(msg)
        try:
            return get_current_squad_from_api(fpl_team_id, fetcher=fetcher)

        except requests.exceptions.RequestException:
            logger.warning(
                "Failed to get current squad from API. Using DB instead, which "
                "may be out of date.",
                exc_info=True,
            )

    # otherwise, we use the Transaction table in the DB
    return get_squad_from_transactions(next_gw, season, fpl_team_id, dbsession)


def get_squad_from_transactions(
    gameweek, season=CURRENT_SEASON, fpl_team_id=None, dbsession: Session | None = None
):
    dbsession = dbsession if dbsession is not None else get_session()
    if not fpl_team_id:
        # use the most recent transaction in the table
        most_recent = dbsession.scalars(
            select(Transaction)
            .where(Transaction.free_hit == 0, Transaction.season == season)
            .order_by(Transaction.id.desc())
            .limit(1)
        ).first()
        if most_recent is None:
            msg = "No transactions in database."
            raise ValueError(msg)
        fpl_team_id = most_recent.fpl_team_id
    logger.debug("Getting starting squad for %s", fpl_team_id)

    # Don't include free hit transfers as they only apply for the week the
    # chip is activated
    transactions = dbsession.scalars(
        select(Transaction)
        .where(
            Transaction.fpl_team_id == fpl_team_id,
            Transaction.free_hit == 0,
            Transaction.season == season,
            Transaction.gameweek < gameweek,
        )
        .order_by(Transaction.gameweek, Transaction.id)
    ).all()
    if len(transactions) == 0:
        msg = f"No transactions in database for team ID {fpl_team_id}"
        raise ValueError(msg)

    s = Squad(season=season)
    for trans in transactions:
        if trans.bought_or_sold == -1:
            s.remove_player(trans.player_id, price=trans.price)
        else:
            # within an individual transfer we can violate the budget and squad
            # constraints, as long as the final squad for that gameweek obeys them
            s.add_player(
                trans.player_id,
                price=trans.price,
                gameweek=gameweek,  # not trans.gameweek, to get player's current club
                check_budget=False,
                check_team=False,
            )
    return s


def get_discounted_squad_score(
    squad: Squad,
    gameweeks: list,
    tag: str,
    root_gw: int | None = None,
    bench_boost_gw: int | None = None,
    triple_captain_gw: int | None = None,
    sub_weights: dict | None = None,
) -> float:
    """Get the number of points a squad is expected to score across a number of
    gameweeks, discounting the weight of gameweeks further into the future with respect
    to the root_gw.
    """
    if root_gw is None:
        root_gw = gameweeks[0]
    total_points = 0
    for gw in gameweeks:
        gw_weight = get_discount_factor(root_gw, gw)
        if gw == bench_boost_gw:
            total_points += (
                squad.get_expected_points(gw, tag, bench_boost=True) * gw_weight
            )
        elif gw == triple_captain_gw:
            total_points += (
                squad.get_expected_points(gw, tag, triple_captain=True) * gw_weight
            )
        else:
            total_points += squad.get_expected_points(gw, tag) * gw_weight

        if gw != bench_boost_gw and sub_weights:
            total_points += gw_weight * squad.total_points_for_subs(
                gw, tag, sub_weights=sub_weights
            )

    return total_points


def get_baseline_strat(squad, gameweeks, tag, root_gw=None):
    """
    Create the strategy dict used by the optimisation for the baseline of making no
    transfers.
    """
    strat_dict = {
        "total_score": 0,
        "points_per_gw": {},
        "players_in": {},
        "players_out": {},
        "chips_played": {},
        "root_gw": root_gw,
    }
    for gw in gameweeks:
        gw_score = get_discounted_squad_score(squad, [gw], tag, root_gw=root_gw)
        strat_dict["total_score"] += gw_score
        strat_dict["points_per_gw"][gw] = gw_score
        strat_dict["players_in"][gw] = []
        strat_dict["players_out"][gw] = []
        strat_dict["chips_played"][gw] = None

    return strat_dict


def fill_suggestion_table(
    baseline_score, best_strat, season, fpl_team_id, dbsession: Session | None = None
):
    """
    Fill the optimized strategy into the table
    """
    dbsession = dbsession if dbsession is not None else get_session()
    timestamp = str(datetime.now())
    best_score = best_strat["total_score"]

    points_gain = best_score - baseline_score
    for in_or_out in [("players_out", -1), ("players_in", 1)]:
        for gameweek, players in best_strat[in_or_out[0]].items():
            for player in players:
                ts = TransferSuggestion()
                ts.player_id = player
                ts.in_or_out = in_or_out[1]
                ts.gameweek = gameweek
                ts.points_gain = points_gain
                ts.timestamp = timestamp
                ts.season = season
                ts.fpl_team_id = fpl_team_id
                ts.chip_played = best_strat["chips_played"][gameweek]
                dbsession.add(ts)
    dbsession.commit()


def fill_transaction_table(
    starting_squad,
    best_strat,
    season,
    fpl_team_id,
    tag=None,
    dbsession: Session | None = None,
):
    """Add transactions from an optimised strategy to the transactions table in the
    database. Used for simulating seasons only, for playing the current FPL season
    the transactions status is kept up to date with transfers using the FPL API.
    Only transfers from the first gameweek in the strategy are added to the Transaction
    table - it's assumed the strategy will be re-optimised after each week rather than
    sticking with the originally proposed future transfers.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    strat_gws = [int(gw) for gw in best_strat["players_in"]]
    fill_gw = min(strat_gws)
    if tag is None:
        tag = f"AIrsenal{season}"
    free_hit = int(best_strat["chips_played"][str(fill_gw)] == "free_hit")
    time = datetime.now().isoformat()
    for player_id in best_strat["players_out"][str(fill_gw)]:
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
    for player_id in best_strat["players_in"][str(fill_gw)]:
        if player := get_player(player_id, dbsession=dbsession):
            price = player.price(season, fill_gw)
            add_transaction(
                player_id,
                fill_gw,
                1,
                price,
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
    squad,
    fpl_team_id,
    tag,
    season=CURRENT_SEASON,
    gameweek: int | None = None,
    dbsession: Session | None = None,
):
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
        ts.gameweek = next_gameweek()
        ts.points_gain = score
        ts.timestamp = timestamp
        ts.season = season
        ts.fpl_team_id = fpl_team_id
        ts.chip_played = None
        dbsession.add(ts)
    dbsession.commit()


def fill_initial_transaction_table(
    squad,
    fpl_team_id,
    tag,
    season=CURRENT_SEASON,
    gameweek: int | None = None,
    dbsession: Session | None = None,
):
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


def get_num_increments(move: GameweekMove, num_iterations: int = 100) -> int:
    """
    how many steps for the progress bar for this strategy
    """
    if move.rebuilds_squad or move.n_transfers > 2:
        # wildcard, free hit, or >2 transfers - all search num_iterations candidates
        return num_iterations
    if move.n_transfers == 0:
        return 1
    if move.n_transfers == 1:
        # single transfer - 15 increments (replace each player in turn)
        return 15
    # two transfers - remove each pair of players, 15*7=105 combinations
    return 105


def next_week_transfers(
    strat: tuple[int, int, dict],
    max_total_hit: int | None = None,
    allow_unused_transfers: bool = True,
    max_opt_transfers: int = 2,
    chips: GameweekChips | None = None,
    max_free_transfers: int = MAX_FREE_TRANSFERS,
) -> list[tuple[GameweekMove, int, int, int]]:
    """Given a previous strategy and some optimisation constraints, determine the valid
    moves (transfers, and any chip played) for the following gameweek.

    strat is a tuple (free_transfers, total_points_hit, strat_dict)
    strat_dict must have key chips_played, a dict indexed by gameweek with values
    None or a Chip.

    max_opt_transfers - maximum number of transfers to play each week as part of
    strategy in optimisation

    max_free_transfers - maximum number of free transfers saved in the game rules
    (2 before 2024/25, 5 from 2024/25 season)

    Returns (move, new_ft_available, total_points_hit, hit_this_gw) tuples.
        - total_points_hit is the total points hit so far including this gw
        - hit_this_gw is the points hit incurred this gameweek
    """
    chips = chips if chips is not None else NO_CHIPS
    ft_available, hit_so_far, strat_dict = strat
    chips_played = strat_dict["chips_played"].values()

    if not allow_unused_transfers and ft_available == max_free_transfers:
        # Force at least 1 free transfer if a free transfer will be lost otherwise.
        # NOTE: This can cause the baseline strategy to be excluded. Re-add it outside
        # this function in that case.
        ft_choices = list(range(1, max_opt_transfers + 1))
    else:
        ft_choices = list(range(max_opt_transfers + 1))

    if max_total_hit is not None:
        ft_choices = [
            nt
            for nt in ft_choices
            if hit_so_far + calc_points_hit(GameweekMove(nt), ft_available)
            <= max_total_hit
        ]

    # if we are definitely going to play a wildcard or free_hit deal with that first
    if chips.chip_to_play is not None and chips.chip_to_play.rebuilds_squad:
        moves = [GameweekMove(chip=chips.chip_to_play)]
    elif chips.chip_to_play is not None:
        # triple captain or bench boost - we can still do ft_choices transfers
        moves = [GameweekMove(nt, chips.chip_to_play) for nt in ft_choices]
    else:
        # no chip definitely played, but some might be allowed
        moves = [GameweekMove(nt) for nt in ft_choices]
        for chip in (Chip.WILDCARD, Chip.FREE_HIT):
            if chips.allows(chip, chips_played):
                moves.append(GameweekMove(chip=chip))
        for chip in (Chip.BENCH_BOOST, Chip.TRIPLE_CAPTAIN):
            if chips.allows(chip, chips_played):
                moves += [GameweekMove(nt, chip) for nt in ft_choices]

    hit_this_gw = [calc_points_hit(move, ft_available) for move in moves]
    total_points_hit = [hit_so_far + hit for hit in hit_this_gw]
    new_ft_available = [
        calc_free_transfers(move, ft_available, max_free_transfers) for move in moves
    ]

    return list(
        zip(moves, new_ft_available, total_points_hit, hit_this_gw, strict=True)
    )


def count_expected_outputs(
    gw_ahead: int,
    next_gw: int | None = None,
    free_transfers: int = 1,
    max_total_hit: int | None = None,
    allow_unused_transfers: bool = True,
    max_opt_transfers: int = 2,
    chip_schedule: ChipSchedule | None = None,
    max_free_transfers: int = MAX_FREE_TRANSFERS,
) -> tuple[int, bool]:
    """
    Count the number of possible transfer and chip strategies for gw_ahead gameweeks
    ahead, subject to:
    * Start with free_transfers free transfers.
    * Spend a max of max_total_hit points on transfers across whole period
    (None for no limit)
    * Allow playing the chips permitted by chip_schedule
    * Exclude strategies that waste free transfers (make 0 transfers if 2 free tramsfers
    are available), if allow_unused_transfers is False.
    * Make a maximum of max_opt_transfers transfers each gameweek.
    * Each chip only allowed once.

    Returns
    -------
        Tuple of int: number of strategies that will be computed, and bool: whether the
        baseline strategy will be excluded from the main optimization tree and will need
        to be computed separately (this can be the case if allow_unused_transfers is
        False). Either way, the total count of strategies will include the baseline.
    """
    next_gw = next_gameweek() if next_gw is None else next_gw
    chip_schedule = chip_schedule if chip_schedule is not None else ChipSchedule()
    init_strat_dict: dict[str, dict[int, list[int] | Chip | None]] = {
        "players_in": {},
        "chips_played": {},
    }
    init_free_transfers = free_transfers  # used below for baseline strategy logic
    strategies = [(init_free_transfers, 0, init_strat_dict)]

    for gw in range(next_gw, next_gw + gw_ahead):
        new_strategies = []
        for s in strategies:
            possibilities = next_week_transfers(
                s,
                max_total_hit=max_total_hit,
                max_opt_transfers=max_opt_transfers,
                allow_unused_transfers=allow_unused_transfers,
                chips=chip_schedule.for_gameweek(gw),
                max_free_transfers=max_free_transfers,
            )

            for move, new_free_transfers, new_hit, _ in possibilities:
                # make a copy of the strategy up to this point, then add on this gw.
                # Only the shape of players_in matters here - the count of strategies
                # is what we are after, not the transfers themselves.
                new_dict = deepcopy(s[2])
                new_dict["players_in"][gw] = [1] * move.n_players_in
                new_dict["chips_played"][gw] = move.chip
                new_strategies.append((new_free_transfers, new_hit, new_dict))

        strategies = new_strategies

    # if allow_unused_transfers is False baseline of no transfers can be removed above.
    # Check whether 1st strategy is the baseline and if not add it back in here
    baseline_strat_dict: dict[str, dict[int, list[int] | Chip | None]] = {
        "players_in": {gw: [] for gw in range(next_gw, next_gw + gw_ahead)},
        "chips_played": dict.fromkeys(range(next_gw, next_gw + gw_ahead)),
    }
    if strategies[0][2] != baseline_strat_dict:
        baseline_dict = (max_free_transfers, 0, baseline_strat_dict)
        strategies.insert(0, baseline_dict)
        baseline_excluded = True
    else:
        baseline_excluded = False

    return len(strategies), baseline_excluded


def get_discount_factor(
    next_gw: int,
    pred_gw: int,
    discount_type: str = "exp",
    discount: float = DEFAULT_DISCOUNT,
) -> float:
    """
    given the next gw and a predicted gw, retrieve discount factor. Either:
        - exp: discount**n_ahead (discount reduces each gameweek)
        - const: 1-(1-discount)*n_ahead (constant discount each gameweek, goes to
          zero at gw 15 with default discount)
    """
    allowed_types = ["exp", "const", "constant"]
    if discount_type not in allowed_types:
        msg = "unrecognised discount type, should be exp or const"
        raise Exception(msg)

    n_ahead = pred_gw - next_gw

    if discount_type in ["exp"]:
        score = discount**n_ahead
    elif discount_type in ["const", "constant"]:
        score = max(1 - (1 - discount) * n_ahead, 0)

    return score
