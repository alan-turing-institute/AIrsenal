"""
functions to optimize the transfers for N weeks ahead
"""
import warnings
from copy import deepcopy
from datetime import datetime
from typing import Optional

import requests

from airsenal.framework.schema import (
    Fixture,
    PlayerPrediction,
    Transaction,
    TransferSuggestion,
)
from airsenal.framework.squad import Squad
from airsenal.framework.transaction_utils import add_transaction
from airsenal.framework.utils import (
    CURRENT_SEASON,
    NEXT_GAMEWEEK,
    get_bank,
    get_current_squad_from_api,
    get_player,
    session,
)

positions = ["FWD", "MID", "DEF", "GK"]  # front-to-back

DEFAULT_SUB_WEIGHTS = {"GK": 0.03, "Outfield": (0.65, 0.3, 0.1)}


def check_tag_valid(pred_tag, gameweek_range, season=CURRENT_SEASON, dbsession=session):
    """Check a prediction tag contains predictions for all the specified gameweeks."""
    # get unique gameweek and season values associated with pred_tag
    fixtures = (
        (
            dbsession.query(Fixture.season, Fixture.gameweek)
            .filter(PlayerPrediction.tag == pred_tag)
            .join(PlayerPrediction)
        )
        .distinct()
        .all()
    )
    pred_seasons = [f[0] for f in fixtures]
    pred_gws = [f[1] for f in fixtures]

    season_ok = all(s == season for s in pred_seasons)
    gws_ok = all(gw in pred_gws for gw in gameweek_range)

    return season_ok and gws_ok


def calc_points_hit(num_transfers, free_transfers):
    """
    Current rules say we lose 4 points for every transfer beyond
    the number of free transfers we have.
    Num transfers can be an integer, or "W", "F", "Bx", or "Tx"
    (wildcard, free hit, bench-boost or triple-caption).
    For Bx and Tx the "x" corresponds to the number of transfers
    in addition to the chip being played.
    """
    if num_transfers in ["W", "F"]:
        return 0
    elif isinstance(num_transfers, int):
        return max(0, 4 * (num_transfers - free_transfers))
    elif (num_transfers.startswith("B") or num_transfers.startswith("T")) and len(
        num_transfers
    ) == 2:
        num_transfers = int(num_transfers[-1])
        return max(0, 4 * (num_transfers - free_transfers))
    else:
        raise RuntimeError(f"Unexpected argument for num_transfers {num_transfers}")


def calc_free_transfers(num_transfers, prev_free_transfers):
    """
    We get one extra free transfer per week, unless we use a wildcard or
    free hit, but we can't have more than 2.  So we should only be able
    to return 1 or 2.
    """
    if num_transfers in ["W", "F"]:
        return 1
    elif isinstance(num_transfers, int):
        return max(1, min(2, 1 + prev_free_transfers - num_transfers))
    elif (num_transfers.startswith("B") or num_transfers.startswith("T")) and len(
        num_transfers
    ) == 2:
        # take the 'x' out of Bx or Tx
        num_transfers = int(num_transfers[-1])
        return max(1, min(2, 1 + prev_free_transfers - num_transfers))
    else:
        raise RuntimeError(f"Unexpected argument for num_transfers {num_transfers}")


def get_starting_squad(
    next_gw=NEXT_GAMEWEEK,
    season=CURRENT_SEASON,
    fpl_team_id=None,
    use_api=False,
    apifetcher=None,
):
    """
    use the transactions table in the db, or the API if requested
    """
    if use_api:
        if season != CURRENT_SEASON:
            raise RuntimeError("Can only use API for current season and gameweek")
        if season == CURRENT_SEASON and next_gw != NEXT_GAMEWEEK:
            raise RuntimeError("Can only use API for current season and gameweek")
        if not fpl_team_id:
            raise RuntimeError(
                "Please specify fpl_team_id to get current squad from API"
            )
        try:
            players_prices = get_current_squad_from_api(
                fpl_team_id, apifetcher=apifetcher
            )
            s = Squad(season=CURRENT_SEASON)
            for pp in players_prices:
                s.add_player(
                    pp[0],
                    price=pp[1],
                    gameweek=next_gw - 1,
                    check_budget=False,
                    check_team=False,
                )
            s.budget = get_bank(fpl_team_id, season=CURRENT_SEASON)
            return s
        except requests.exceptions.RequestException as e:
            warnings.warn(
                f"Failed to get current squad from API:\n{e}\nUsing DB instead, which "
                "may be out of date."
            )

    # otherwise, we use the Transaction table in the DB
    return get_squad_from_transactions(next_gw - 1, season, fpl_team_id)


def get_squad_from_transactions(gameweek, season=CURRENT_SEASON, fpl_team_id=None):
    if not fpl_team_id:
        # use the most recent transaction in the table
        most_recent = (
            session.query(Transaction)
            .order_by(Transaction.id.desc())
            .filter_by(free_hit=0)
            .filter_by(season=season)
            .first()
        )
        if most_recent is None:
            raise ValueError("No transactions in database.")
        fpl_team_id = most_recent.fpl_team_id
    print(f"Getting starting squad for {fpl_team_id}")

    # Don't include free hit transfers as they only apply for the week the
    # chip is activated
    transactions = (
        session.query(Transaction)
        .order_by(Transaction.gameweek, Transaction.id)
        .filter_by(fpl_team_id=fpl_team_id)
        .filter_by(free_hit=0)
        .filter_by(season=season)
        .filter(Transaction.gameweek <= gameweek)
        .all()
    )
    if len(transactions) == 0:
        raise ValueError(f"No transactions in database for team ID {fpl_team_id}")

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
                gameweek=trans.gameweek,
                check_budget=False,
                check_team=False,
            )
    return s


def get_discounted_squad_score(
    squad: Squad,
    gameweeks: list,
    tag: str,
    root_gw: Optional[int] = None,
    bench_boost_gw: Optional[int] = None,
    triple_captain_gw: Optional[int] = None,
    sub_weights: Optional[dict] = None,
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


def fill_suggestion_table(baseline_score, best_strat, season, fpl_team_id):
    """
    Fill the optimized strategy into the table
    """
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
                session.add(ts)
    session.commit()


def fill_transaction_table(
    starting_squad, best_strat, season, fpl_team_id, tag=None, dbsession=session
):
    """Add transactions from an optimised strategy to the transactions table in the
    database. Used for simulating seasons only, for playing the current FPL season
    the transactions status is kept up to date with transfers using the FPL API.
    Only transfers from the first gameweek in the strategy are added to the Transaction
    table - it's assumed the strategy will be re-optimised after each week rather than
    sticking with the originally proposed future transfers.
    """
    strat_gws = [int(gw) for gw in best_strat["players_in"].keys()]
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
        player = get_player(player_id, dbsession=dbsession)
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


def fill_initial_suggestion_table(
    squad,
    fpl_team_id,
    tag,
    season=CURRENT_SEASON,
    gameweek=NEXT_GAMEWEEK,
    dbsession=session,
):
    """
    Fill an initial squad into the table
    """
    timestamp = str(datetime.now())
    score = squad.get_expected_points(gameweek, tag)
    for player in squad.players:
        ts = TransferSuggestion()
        ts.player_id = player.player_id
        ts.in_or_out = 1
        ts.gameweek = NEXT_GAMEWEEK
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
    gameweek=NEXT_GAMEWEEK,
    dbsession=session,
):
    """Add transactions from an initial squad optimisation to the transactions table
    in the database. Used for simulating seasons only, for playing the current FPL
    season the transactions status is kepts up to date with transfers using the FPL API.
    """
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


def strategy_involves_N_or_more_transfers_in_gw(strategy, N):
    """
    Quick function to see if we need to do multiple iterations
    for a strategy, or if the result is deterministic
    (0 or 1 transfer for each gameweek).
    """
    strat_dict = strategy[0]
    return any(isinstance(v, int) and v >= N for v in strat_dict.values())


def make_strategy_id(strategy):
    """
    Return a string that will identify a strategy - just concatenate
    the numbers of transfers per gameweek.
    """
    return ",".join(str(nt) for nt in strategy[0].values())


def get_num_increments(num_transfers, num_iterations=100):
    """
    how many steps for the progress bar for this strategy
    """
    if (
        isinstance(num_transfers, str)
        and (num_transfers.startswith("B") or num_transfers.startswith("T"))
        and len(num_transfers) == 2
    ):
        num_transfers = int(num_transfers[1])

    if (
        num_transfers == "W"
        or num_transfers == "F"
        or (isinstance(num_transfers, int) and num_transfers > 2)
    ):
        # wildcard or free hit or >2 - needs num_iterations iterations
        return num_iterations

    elif num_transfers == 0:
        return 1

    elif num_transfers == 1:
        # single transfer - 15 increments (replace each player in turn)
        return 15
    elif num_transfers == 2:
        # remove each pair of players - 15*7=105 combinations
        return 105
    else:
        print(f"Unrecognized num_transfers: {num_transfers}")
        return 1


def next_week_transfers(
    strat,
    max_total_hit=None,
    allow_unused_transfers=True,
    max_transfers=2,
    chips={"chips_allowed": [], "chip_to_play": None},
):
    """Given a previous strategy and some optimisation constraints, determine the valid
    options for the number of transfers (or chip played) in the following gameweek.

    strat is a tuple (free_transfers, hit_so_far, strat_dict)
    strat_dict must have key chips_played, which is a dict indexed by gameweek with
    possible values None, "wildcard", "free_hit", "bench_boost" or triple_captain"
    """
    # check that the 'chips' dict we are given makes sense:
    if (
        "chips_allowed" in chips.keys()
        and len(chips["chips_allowed"]) > 0
        and "chip_to_play" in chips.keys()
        and chips["chip_to_play"]
    ):
        raise RuntimeError(
            f"Cannot allow {chips['chips_allowed']}"
            "in the same week as we play {chips['chip_to_play']}"
        )
    ft_available, hit_so_far, strat_dict = strat
    chip_history = strat_dict["chips_played"]

    if not allow_unused_transfers and ft_available == 2:
        # Force at least 1 free transfer.
        # NOTE: This will exclude the baseline strategy when allow_unused_transfers
        # is False. Re-add it outside this function in that case.
        ft_choices = list(range(1, max_transfers + 1))
    else:
        ft_choices = list(range(max_transfers + 1))

    if max_total_hit is not None:
        ft_choices = [
            nt
            for nt in ft_choices
            if hit_so_far + calc_points_hit(nt, ft_available) <= max_total_hit
        ]

    allow_wildcard = (
        "chips_allowed" in chips.keys()
        and "wildcard" in chips["chips_allowed"]
        and "wildcard" not in chip_history.values()
    )
    allow_free_hit = (
        "chips_allowed" in chips.keys()
        and "free_hit" in chips["chips_allowed"]
        and "free_hit" not in chip_history.values()
    )
    allow_bench_boost = (
        "chips_allowed" in chips.keys()
        and "bench_boost" in chips["chips_allowed"]
        and "bench_boost" not in chip_history.values()
    )
    allow_triple_captain = (
        "chips_allowed" in chips.keys()
        and "triple_captain" in chips["chips_allowed"]
        and "triple_captain" not in chip_history.values()
    )

    # if we are definitely going to play a wildcard or free_hit deal with
    # that first
    if "chip_to_play" in chips.keys() and chips["chip_to_play"] == "wildcard":
        new_transfers = ["W"]
    elif "chip_to_play" in chips.keys() and chips["chip_to_play"] == "free_hit":
        new_transfers = ["F"]
    # for triple captain or bench boost, we can still do ft_choices transfers
    elif "chip_to_play" in chips.keys() and chips["chip_to_play"] == "triple_captain":
        new_transfers = [f"T{nt}" for nt in ft_choices]
    elif "chip_to_play" in chips.keys() and chips["chip_to_play"] == "bench_boost":
        new_transfers = [f"B{nt}" for nt in ft_choices]
    else:
        # no chip definitely played, but some might be allowed
        new_transfers = [nt for nt in ft_choices]  # make a copy
        if allow_wildcard:
            new_transfers.append("W")
        if allow_free_hit:
            new_transfers.append("F")
        if allow_bench_boost:
            new_transfers += [f"B{nt}" for nt in ft_choices]
        if allow_triple_captain:
            new_transfers += [f"T{nt}" for nt in ft_choices]

    new_points_hits = [
        hit_so_far + calc_points_hit(nt, ft_available) for nt in new_transfers
    ]
    new_ft_available = [calc_free_transfers(nt, ft_available) for nt in new_transfers]

    # return list of (num_transfers, free_transfers, hit_so_far) tuples for each new
    # strategy
    return list(zip(new_transfers, new_ft_available, new_points_hits))


def count_expected_outputs(
    gw_ahead,
    next_gw=NEXT_GAMEWEEK,
    free_transfers=1,
    max_total_hit=None,
    allow_unused_transfers=True,
    max_transfers=2,
    chip_gw_dict={},
):
    """
    Count the number of possible transfer and chip strategies for gw_ahead gameweeks
    ahead, subject to:
    * Start with free_transfers free transfers.
    * Spend a max of max_total_hit points on transfers across whole period
    (None for no limit)
    * Allow playing the chips which have their allow_xxx argument set True
    * Exclude strategies that waste free transfers (make 0 transfers if 2 free tramsfers
    are available), if allow_unused_transfers is False.
    * Make a maximum of max_transfers transfers each gameweek.
    * Each chip only allowed once.
    """

    init_strat_dict = {
        "players_in": {},
        "chips_played": {},
    }
    init_free_transfers = free_transfers  # used below for baseline strategy logic
    strategies = [(init_free_transfers, 0, init_strat_dict)]

    for gw in range(next_gw, next_gw + gw_ahead):
        new_strategies = []
        for s in strategies:
            free_transfers = s[0]
            chips_for_gw = chip_gw_dict[gw] if gw in chip_gw_dict.keys() else {}
            possibilities = next_week_transfers(
                s,
                max_total_hit=max_total_hit,
                max_transfers=max_transfers,
                allow_unused_transfers=allow_unused_transfers,
                chips=chips_for_gw,
            )

            for n_transfers, new_free_transfers, new_hit in possibilities:
                # make a copy of the strategy up to this point, then add on this gw
                new_dict = deepcopy(s[2])

                # update dummy strat dict
                if n_transfers == "W":
                    # add dummy values to transfer dict for 15 possible transfers
                    new_dict["players_in"][gw] = [1] * 15
                    new_dict["chips_played"][gw] = "wildcard"
                elif n_transfers == "F":
                    # add dummy values to transfer dict for 15 possible transfers
                    new_dict["players_in"][gw] = [1] * 15
                    new_dict["chips_played"][gw] = "free_hit"
                else:
                    if isinstance(n_transfers, str) and (
                        n_transfers.startswith("T") or n_transfers.startswith("B")
                    ):
                        if n_transfers[0] == "T":
                            new_dict["chips_played"][gw] = "triple_captain"
                        elif n_transfers[0] == "B":
                            new_dict["chips_played"][gw] = "bench_boost"
                        n_transfers = int(n_transfers[1])
                    # add dummy values to transfer dict for n_transfers transfers
                    new_dict["players_in"][gw] = [1] * n_transfers

                new_strategies.append((new_free_transfers, new_hit, new_dict))

        strategies = new_strategies

    # if allow_unused_transfers is False baseline of no transfers will be removed above,
    # add it back in here, apart from edge cases where it's already included.
    if not allow_unused_transfers and (
        gw_ahead > 1 or (gw_ahead == 1 and init_free_transfers == 2)
    ):
        baseline_strat_dict = {
            "players_in": {gw: [] for gw in range(next_gw, next_gw + gw_ahead)},
            "chips_played": {},
        }
        baseline_dict = (2, 0, baseline_strat_dict)
        strategies.insert(0, baseline_dict)
    return len(strategies)


def get_discount_factor(next_gw, pred_gw, discount_type="exp", discount=14 / 15):
    """
    given the next gw and a predicted gw, retrieve discount factor. Either:
        - exp: discount**n_ahead (discount reduces each gameweek)
        - const: 1-(1-discount)*n_ahead (constant discount each gameweek, goes to
          zero at gw 15 with default discount)
    """
    allowed_types = ["exp", "const", "constant"]
    if discount_type not in allowed_types:
        raise Exception("unrecognised discount type, should be exp or const")

    if not next_gw:
        # during tests 'none' is passed as the root gw, default to zero so the
        # optimisation is done solely on pred_gw ahead.
        next_gw = pred_gw
    n_ahead = pred_gw - next_gw

    if discount_type in ["exp"]:
        score = discount**n_ahead
    elif discount_type in ["const", "constant"]:
        score = max(1 - (1 - discount) * n_ahead, 0)

    return score
