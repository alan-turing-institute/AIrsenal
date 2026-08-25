"""
Rendering what an optimisation decided.

Both entry points - the transfer search and the from-scratch squad build - share
the result panel and the per-gameweek table here, and supply the rows.

Nothing here queries the database or the FPL API, and nothing here simulates a
transfer: callers pass in what they already know.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rich.panel import Panel
from rich.text import Text

from airsenal.core.console import console, price_str, table
from airsenal.game.enums import Position
from airsenal.squad.player import CandidatePlayer, DummyPlayer, bench_position
from airsenal.squad.squad import Squad


@dataclass(frozen=True)
class GameweekRow:
    """One row of the per-gameweek plan table."""

    gameweek: int
    transfers: str
    chip: str | None
    points_hit: int
    predicted_points: float


@dataclass(frozen=True)
class TransferRow:
    """One transfer, with the prices it was made at."""

    gameweek: int
    player_out: str
    position_out: str
    team_out: str
    sale_price: int | None
    player_in: str
    position_in: str | None
    team_in: str | None
    purchase_price: int | None


def print_result_panel(
    gameweeks: Sequence[int],
    fpl_team_id: int | None,
    optimised_score: float,
    baseline_score: float | None = None,
    points_hit: int = 0,
    chips: tuple[str, ...] = (),
) -> None:
    """The headline panel: what was optimised, and what it gained."""
    first, last = min(gameweeks), max(gameweeks)
    summary = Text()
    summary.append(
        f"Gameweeks: {first}-{last}\n" if first != last else f"Gameweek: {first}\n",
        style="bold",
    )
    summary.append(f"Team ID: {fpl_team_id}\n")
    if baseline_score is not None:
        summary.append(f"Baseline Score: {baseline_score:.1f}pts\n")
        summary.append(
            f"Total Points Hits: -{points_hit}pts\n",
            style="red" if points_hit else None,
        )
        summary.append(
            f"Chips Played: {', '.join(chips) if chips else 'None'}\n",
            style="red" if chips else None,
        )
    summary.append(f"Optimised Score: {optimised_score:.1f}pts\n", style="bold green")
    if baseline_score is not None:
        summary.append(
            f"Points Gained: {optimised_score - baseline_score:+.1f}pts",
            style="bold green" if optimised_score > baseline_score else "bold red",
        )
    console.print(Panel(summary, title="Optimisation Result", expand=False))


def print_plan_table(rows: Sequence[GameweekRow]) -> None:
    """What the plan does in each gameweek."""
    plan_table = table(
        "Gameweek",
        "Transfers",
        "Chip",
        "Points Hit",
        "Predicted Score",
        title="Plan",
    )
    for row in rows:
        plan_table.add_row(
            str(row.gameweek),
            row.transfers,
            row.chip or "-",
            f"-{row.points_hit}pts" if row.points_hit else "0pts",
            f"{row.predicted_points:.1f}pts",
        )
    console.print(plan_table)


def print_transfer_table(rows: Sequence[TransferRow]) -> None:
    """The transfers the plan makes, and the prices they are made at."""
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
    for row in rows:
        transfer_table.add_row(
            str(row.gameweek),
            row.player_out,
            row.position_out,
            row.team_out,
            price_str(row.sale_price),
            row.player_in,
            row.position_in or "-",
            row.team_in or "-",
            price_str(row.purchase_price),
        )
    if rows:
        console.print(transfer_table)
    else:
        console.print(f"{transfer_table.title}: no transfers made.")


def print_squad_table(players: Sequence[CandidatePlayer | DummyPlayer]) -> None:
    """Every player in a squad built from scratch: all of them are incoming."""
    # str rather than Position: a squad player's position is a plain string
    order: list[str] = list(Position.front_to_back())
    squad_table = table("Player In", "Pos", "Team", "Purchase Price", title="Transfers")
    for player in sorted(players, key=lambda p: order.index(p.position)):
        squad_table.add_row(
            str(player), player.position, player.team, price_str(player.purchase_price)
        )
    console.print(squad_table)


def lineup_strings(
    squad: Squad, optimised_score: float, baseline_score: float, fpl_team_id: int
) -> list[str]:
    """The squad, formatted as Discord markdown."""
    lines = [
        f"__Plan for Team ID: **{fpl_team_id}**__",
        f"Baseline score: *{int(baseline_score)}*",
        f"Best score: *{int(optimised_score)}*",
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


def discord_payload(
    plan: Sequence[GameweekRow],
    transfers: Sequence[TransferRow],
    lineup: Sequence[str],
) -> dict[str, Any]:
    """
    The webhook body describing a plan.

    Takes the same rows the tables render rather than a `Plan`: the optimisation
    stage sits above this one, so a renderer here cannot name its types.
    """
    fields: list[dict[str, Any]] = []
    for row in plan:
        gw = row.gameweek
        made = [t for t in transfers if t.gameweek == gw]
        fields.extend(
            [
                {
                    "name": f"GW{gw} chips:",
                    "value": f"Chips played:  {row.chip}\n",
                    "inline": False,
                },
                {
                    "name": f"GW{gw} transfers out:",
                    "value": "\n".join(t.player_out for t in made),
                    "inline": True,
                },
                {
                    "name": f"GW{gw} transfers in:",
                    "value": "\n".join(t.player_in for t in made),
                    "inline": True,
                },
            ]
        )
    discord_embed: dict[str, Any] = {
        "title": "AIrsenal webhook",
        "description": "Optimum plan for gameweek(S)"
        f" {','.join(str(row.gameweek) for row in plan)}:",
        "color": 0x35A800,
        "fields": fields,
    }
    return {
        "content": "\n".join(lineup),
        "username": "AIrsenal",
        "embeds": [discord_embed],
    }
