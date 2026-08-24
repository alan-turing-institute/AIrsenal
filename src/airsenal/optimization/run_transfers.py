"""
usage:
python fill_transfersuggestions_table.py --n_gameweeks <num_weeks_ahead>
                                          --num_iterations <num_iterations>
output for each strategy tried is going to be a dict
{ "total_points": <float>,
"points_per_gw": {<gw>: <float>, ...},
"players_sold" : {<gw>: [], ...},
"players_bought" : {<gw>: [], ...}
}
This is done via a recursive tree search, where nodes on the tree do an optimization
for a given number of transfers, then adds some children to the multiprocessing queue
representing 0, 1, 2 transfers for the next gameweek.

"""

import json
import sys
from pathlib import Path
from typing import Any

from rich.panel import Panel
from rich.text import Text
from sqlalchemy.orm import Session

from airsenal.core.concurrency import (
    set_multiprocessing_start_method,
)
from airsenal.core.console import console, price_str, table
from airsenal.core.enums import Chip, Position
from airsenal.core.logging import get_logger
from airsenal.core.season import CURRENT_SEASON
from airsenal.db.queries.gameweeks import get_gameweeks_array
from airsenal.db.queries.players import get_player, get_player_name
from airsenal.db.queries.tags import check_tag_valid, get_latest_prediction_tag
from airsenal.db.session import get_session
from airsenal.fetch.fpl_api import get_fetcher, require_fpl_team_id
from airsenal.optimization.config import ChipWeeks
from airsenal.optimization.moves import ChipSchedule, TransferConstraints
from airsenal.optimization.persist import fill_suggestion_table, fill_transaction_table
from airsenal.optimization.protocols import (
    SquadOptimizer,
    TransferOptimizer,
    TransferSearchRequest,
)
from airsenal.optimization.run_squad import fill_initial_squad
from airsenal.optimization.squad_optimizers import (
    GeneticSquadOptimizer,
    genetic_optimizer,
)
from airsenal.optimization.strategy import Strategy
from airsenal.optimization.transfer_optimizers import (
    TreeSearchConfig,
    TreeSearchOptimizer,
)
from airsenal.reporting.discord import post_webhook
from airsenal.reporting.squad_view import formation_table
from airsenal.squad.history import get_starting_squad
from airsenal.squad.player import bench_position
from airsenal.squad.squad import Squad
from airsenal.squad.state import get_entry_start_gameweek, get_free_transfers

logger = get_logger(__name__)


def save_strategy_dump(strategies: list[Strategy], directory: Path, tag: str) -> None:
    """
    Write every strategy considered to one JSON file, for debugging.

    The search itself keeps strategies in memory; this exists only because
    inspecting the whole tree is occasionally the fastest way to understand a
    surprising suggestion.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"strategies_{tag}.json"
    with path.open("w") as f:
        json.dump([s.to_dict() for s in strategies], f, indent=2)
    logger.info("Wrote %s strategies to %s", len(strategies), path)


def print_optimization_summary(
    strat: Strategy,
    baseline_score: float,
    season: str = CURRENT_SEASON,
    fpl_team_id: int | None = None,
    use_api: bool = False,
    dbsession: Session | None = None,
) -> None:
    """
    Rich-formatted summary of an optimisation result: total score, the
    chosen strategy (transfers/chips/points hits per gameweek), a table of
    the transfers in/out (with purchase/sale prices), and the resulting
    bank balance.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    first_gw, last_gw = strat.gameweeks[0], strat.gameweeks[-1]
    total_score = strat.total_score
    total_hits = strat.total_points_hit

    summary = Text()
    summary.append(
        f"Gameweeks: {first_gw}-{last_gw}\n"
        if first_gw != last_gw
        else f"Gameweek: {first_gw}\n",
        style="bold",
    )
    summary.append(f"Team ID: {fpl_team_id}\n")
    summary.append(f"Baseline Score: {baseline_score:.1f}pts\n")
    summary.append(
        f"Total Points Hits: -{total_hits}pts\n", style="red" if total_hits else None
    )
    chips_played = [chip for chip in strat.chips_played if chip]
    summary.append(
        f"Chips Played: {', '.join(chips_played) if chips_played else 'None'}\n",
        style="red" if chips_played else None,
    )
    summary.append(f"Optimised Score: {total_score:.1f}pts\n", style="bold green")
    summary.append(
        f"Points Gained: {total_score - baseline_score:+.1f}pts",
        style="bold green" if total_score > baseline_score else "bold red",
    )

    console.print(Panel(summary, title="Optimisation Result", expand=False))

    strategy_table = table(
        "Gameweek",
        "Transfers",
        "Chip",
        "Points Hit",
        "Predicted Score",
        title="Strategy",
    )
    for outcome in strat.outcomes:
        strategy_table.add_row(
            str(outcome.gameweek),
            outcome.move.label(),
            str(outcome.chip) if outcome.chip else "-",
            f"-{outcome.points_hit}pts" if outcome.points_hit else "0pts",
            f"{outcome.undiscounted_points:.1f}pts",
        )
    console.print(strategy_table)

    transfer_table = table(
        "GW",
        "Player Out",
        "Pos",
        "Team",
        "Sale Price",
        "Player In",
        "Pos",
        "Team",
        "Purchase Price",
        title="Transfers",
    )
    any_transfers = False
    squad = get_starting_squad(
        next_gw=first_gw,
        season=season,
        fpl_team_id=fpl_team_id,
        use_api=use_api,
    )
    for outcome in strat.outcomes:
        gw = outcome.gameweek
        for pid_out, pid_in in zip(
            outcome.players_out, outcome.players_in, strict=True
        ):
            any_transfers = True
            out_player = squad.get_player_from_id(pid_out)
            sale_price = squad.get_sell_price_for_player(
                pid_out, use_api=use_api, gameweek=gw, dbsession=dbsession
            )
            squad.remove_player(pid_out, price=sale_price, gameweek=gw)

            in_player_db = get_player(pid_in, dbsession=dbsession)
            purchase_price = in_player_db.price(season, gw) if in_player_db else None
            squad.add_player(
                pid_in,
                price=purchase_price,
                gameweek=gw,
                check_budget=False,
                check_team=False,
                dbsession=dbsession,
            )
            in_name = str(in_player_db) if in_player_db else get_player_name(pid_in)
            transfer_table.add_row(
                str(gw),
                str(out_player),
                out_player.position,
                out_player.team,
                price_str(sale_price),
                in_name,
                in_player_db.position(season) if in_player_db else "-",
                in_player_db.team(season, gw) if in_player_db else "-",
                price_str(purchase_price),
            )
    if any_transfers:
        console.print(transfer_table)
    else:
        console.print(f"{transfer_table.title}: no transfers made.")


def discord_payload(strat: Strategy, lineup: list[str]) -> dict[str, Any]:
    """
    json formated discord webhook content.
    """
    discord_embed = {
        "title": "AIrsenal webhook",
        "description": "Optimum strategy for gameweek(S)"
        f" {','.join(str(gw) for gw in strat.gameweeks)}:",
        "color": 0x35A800,
        "fields": [],
    }
    fields: list[dict[str, Any]] = []
    for outcome in strat.outcomes:
        gw = outcome.gameweek
        fields.append(
            {
                "name": f"GW{gw} chips:",
                "value": f"Chips played:  {outcome.chip}\n",
                "inline": False,
            }
        )
        pin = [str(get_player_name(p)) for p in outcome.players_in]
        pout = [str(get_player_name(p)) for p in outcome.players_out]
        fields.extend(
            [
                {
                    "name": f"GW{gw} transfers out:",
                    "value": "\n".join(pout),
                    "inline": True,
                },
                {
                    "name": f"GW{gw} transfers in:",
                    "value": "\n".join(pin),
                    "inline": True,
                },
            ]
        )
    discord_embed["fields"] = fields
    return {
        "content": "\n".join(lineup),
        "username": "AIrsenal",
        "embeds": [discord_embed],
    }


def print_team_for_next_gw(
    strat: Strategy,
    season: str = CURRENT_SEASON,
    fpl_team_id: int | None = None,
    use_api: bool = False,
) -> Squad:
    """
    Display the team (inc. subs and captain) for the next gameweek
    """
    outcome = strat.outcomes[0]
    next_gw = outcome.gameweek
    t = get_starting_squad(
        next_gw=next_gw, season=season, fpl_team_id=fpl_team_id, use_api=use_api
    )
    for pidout in outcome.players_out:
        t.remove_player(pidout)
    for pidin in outcome.players_in:
        t.add_player(pidin)
    tag = get_latest_prediction_tag(season=season)
    console.print(
        formation_table(
            t,
            tag,
            next_gw,
            bench_boost=outcome.chip is Chip.BENCH_BOOST,
            triple_captain=outcome.chip is Chip.TRIPLE_CAPTAIN,
        )
    )
    return t


def lineup_strings(
    squad: Squad, strategy: Strategy, baseline_score: float, fpl_team_id: int
) -> list[str]:
    """The squad, formatted as Discord markdown."""
    lines = [
        f"__Strategy for Team ID: **{fpl_team_id}**__",
        f"Baseline score: *{int(baseline_score)}*",
        f"Best score: *{int(strategy.total_score)}*",
        "\n__starting 11__",
    ]
    for position in list(Position.back_to_front()):
        lines.append(f"== **{position}** ==\n```")
        for p in squad.players:
            if p.position == position and p.is_starting:
                player_line = f"{p} ({p.team})"
                if p.is_captain:
                    player_line += "(C)"
                elif p.is_vice_captain:
                    player_line += "(VC)"
                lines.append(player_line)
        lines.append("```\n")
    lines += ["__subs__", "```"]
    subs = sorted((p for p in squad.players if not p.is_starting), key=bench_position)
    lines += [f"{p} ({p.team})" for p in subs]
    lines.append("```\n")
    return lines


def new_squad_from_scratch(
    gameweeks: list[int],
    tag: str,
    season: str,
    fpl_team_id: int,
    chip_gameweeks: dict[str, int],
    squad_optimizer: SquadOptimizer | None = None,
) -> Squad:
    """
    Build a squad from nothing, for the start of a season or a brand new team.

    There is nothing to transfer from, so the transfer search has nothing to do.
    """
    if squad_optimizer is None:
        squad_optimizer = GeneticSquadOptimizer()
    return fill_initial_squad(
        tag=tag,
        gameweeks=gameweeks,
        season=season,
        fpl_team_id=fpl_team_id,
        optimizer=squad_optimizer,
        chip_gameweeks=chip_gameweeks,
    )


def run_optimization(
    gameweeks: list[int],
    tag: str,
    season: str = CURRENT_SEASON,
    fpl_team_id: int | None = None,
    chip_gameweeks: dict[str, int] | None = None,
    num_free_transfers: int | None = None,
    constraints: TransferConstraints | None = None,
    optimizer: TransferOptimizer | None = None,
    squad_optimizer: SquadOptimizer | None = None,
    save_strategies: Path | None = None,
    is_replay: bool = False,  # for replaying seasons
) -> tuple[Squad, Strategy | None]:
    """
    This is the actual main function that sets up the multiprocessing
    and calls the optimize function for every move/gameweek
    combination, to find the best strategy.
    The chip-related variables e.g. wildcard_week are -1 if that chip
    is not to be played, 0 for 'play it any week', or the gw in which
    it should be played.
    """
    if chip_gameweeks is None:
        chip_gameweeks = {}
    if constraints is None:
        constraints = TransferConstraints()
    if optimizer is None:
        optimizer = TreeSearchOptimizer()
    fpl_team_id = require_fpl_team_id(fpl_team_id)

    # see if we are at the start of a season, or
    if gameweeks[0] == 1 or gameweeks[0] == get_entry_start_gameweek(
        fpl_team_id, fetcher=get_fetcher()
    ):
        logger.info(
            "This is the start of the season or a new team - will make a squad "
            "from scratch"
        )
        return new_squad_from_scratch(
            gameweeks, tag, season, fpl_team_id, chip_gameweeks, squad_optimizer
        ), None

    with console.status("Optimising transfers..."):
        logger.info("Running optimization with fpl_team_id %s", fpl_team_id)
        use_api = season == CURRENT_SEASON and not is_replay
        try:
            starting_squad = get_starting_squad(
                next_gw=gameweeks[0],
                season=season,
                fpl_team_id=fpl_team_id,
                use_api=use_api,
                fetcher=get_fetcher(),
            )
        except (ValueError, TypeError):
            # first week for this squad?
            logger.warning(
                "No existing squad or transfers found for team_id %s", fpl_team_id
            )
            logger.info("Will suggest a new starting squad:")
            return new_squad_from_scratch(
                gameweeks, tag, season, fpl_team_id, chip_gameweeks, squad_optimizer
            ), None
        # if we got to here, we can assume we are optimizing an existing squad.

        # How many free transfers are we starting with?
        if num_free_transfers is None:
            num_free_transfers = get_free_transfers(
                fpl_team_id,
                gameweeks[0],
                season=season,
                fetcher=get_fetcher(),
                is_replay=is_replay,
            )
        logger.info("Starting with %s free transfers", num_free_transfers)

        # Work out what chips we definitely or possibly will play in each gw
        chip_schedule = ChipSchedule.from_weeks(gameweeks, chip_gameweeks)

        result = optimizer.search(
            TransferSearchRequest(
                starting_squad=starting_squad,
                gameweeks=gameweeks,
                tag=tag,
                season=season,
                chip_schedule=chip_schedule,
                num_free_transfers=num_free_transfers,
                constraints=constraints,
            )
        )

        if save_strategies is not None:
            save_strategy_dump(list(result.considered), save_strategies, tag)
        best_strategy = result.best
        if result.baseline is None:
            logger.warning("No baseline strategy was evaluated")
        baseline_score = result.baseline_score
        fill_suggestion_table(baseline_score, best_strategy, season, fpl_team_id)
        if is_replay:
            # simulating a previous season, so imitate applying transfers by adding
            # the suggestions to the Transaction table
            fill_transaction_table(
                starting_squad, best_strategy, season, fpl_team_id, tag
            )

    console.print()

    print_optimization_summary(
        best_strategy,
        baseline_score,
        season=season,
        fpl_team_id=fpl_team_id,
        use_api=use_api,
    )
    best_squad = print_team_for_next_gw(
        best_strategy, season=season, fpl_team_id=fpl_team_id, use_api=use_api
    )

    post_webhook(
        discord_payload(
            best_strategy,
            lineup_strings(best_squad, best_strategy, baseline_score, fpl_team_id),
        )
    )

    return best_squad, best_strategy


def sanity_check_args(
    n_gameweeks: int | None,
    gameweek_start: int | None,
    gameweek_end: int | None,
    num_free_transfers: int | None,
) -> bool:
    """
    Check that command-line arguments are self-consistent.
    """
    if n_gameweeks and (gameweek_start or gameweek_end):
        msg = "Please only specify n_gameweeks OR gameweek_start/end"
        raise RuntimeError(msg)
    if (gameweek_start and not gameweek_end) or (gameweek_end and not gameweek_start):
        msg = "Need to specify both gameweek_start and gameweek_end"
        raise RuntimeError(msg)
    if num_free_transfers and num_free_transfers not in range(6):
        msg = "Number of free transfers must be 0 to 5"
        raise RuntimeError(msg)
    return True


def run_transfer_optimization(
    n_gameweeks: int | None,
    gameweek_start: int | None,
    gameweek_end: int | None,
    tag: str | None,
    chips: ChipWeeks,
    num_free_transfers: int | None,
    max_hit: int,
    allow_unused: bool,
    max_transfers: int,
    num_iterations: int,
    num_thread: int,
    season: str,
    profile: bool,
    fpl_team_id: int | None,
    is_replay: bool,
    save_strategies: Path | None = None,
) -> None:
    """Run transfer optimization for a gameweek range."""
    sanity_check_args(
        n_gameweeks,
        gameweek_start,
        gameweek_end,
        num_free_transfers,
    )
    gameweeks = get_gameweeks_array(
        n_gameweeks=n_gameweeks,
        gameweek_start=gameweek_start,
        gameweek_end=gameweek_end,
        season=season,
    )
    tag = tag or get_latest_prediction_tag(season=season)

    if not check_tag_valid(tag, gameweeks, season=season):
        logger.error(
            "Database does not contain predictions for all the specified "
            "optimsation gameweeks. Please run 'airsenal_run_prediction' first "
            "with the same input gameweeks and season you specified here."
        )
        sys.exit(1)

    set_multiprocessing_start_method()

    run_optimization(
        gameweeks,
        tag,
        season=season,
        fpl_team_id=fpl_team_id,
        chip_gameweeks=chips.as_dict(),
        num_free_transfers=num_free_transfers,
        constraints=TransferConstraints(
            max_total_hit=max_hit,
            allow_unused_transfers=allow_unused,
            max_opt_transfers=max_transfers,
        ),
        optimizer=TreeSearchOptimizer(
            TreeSearchConfig(
                num_thread=num_thread,
                num_iterations=num_iterations,
                profile=profile,
            )
        ),
        # the from-scratch fallback sizes its search from the same effort knob
        squad_optimizer=genetic_optimizer(num_iterations),
        save_strategies=save_strategies,
        is_replay=is_replay,
    )
