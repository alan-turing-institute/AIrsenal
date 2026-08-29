"""Rendering a Squad for the terminal.

Kept out of the Squad class itself so that the optimiser, which builds and
discards thousands of Squads, does not depend on a rendering library.
"""

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from airsenal.game.enums import Position
from airsenal.squad.lineup import FORMATION_SLOTS
from airsenal.squad.player import SquadPlayer
from airsenal.squad.squad import Squad


def formation_table(
    squad: Squad,
    tag: str | None = None,
    gameweek: int | None = None,
    bench_boost: bool = False,
    triple_captain: bool = False,
) -> Group:
    """Render the squad in a football formation layout.

    Prediction values are displayed when both ``gameweek`` and ``tag`` are
    supplied. Set ``bench_boost`` or ``triple_captain`` to reflect a chip.
    """
    if (gameweek is None) != (tag is None):
        msg = "gameweek and tag must be provided together"
        raise ValueError(msg)
    if bench_boost and triple_captain:
        msg = "bench_boost and triple_captain cannot both be active"
        raise ValueError(msg)

    predicted_points = None
    chip_description = ""
    if gameweek is not None and tag is not None:
        predicted_points = squad.get_expected_points(
            tag,
            gameweek,
            bench_boost=bench_boost,
            triple_captain=triple_captain,
        )
        chip_description = (
            " with bench boost"
            if bench_boost
            else " with triple captain"
            if triple_captain
            else ""
        )

    def player_cell(player: SquadPlayer) -> RenderableType:
        lines = [f"[bold]{player}[/bold]", f"[dim]({player.team})[/dim]"]
        if gameweek is not None and tag is not None:
            points = getattr(player, "predicted_points", {}).get(tag, {}).get(gameweek)
            lines.append(
                f"[dim]{points:.1f} pts[/dim]" if points is not None else "[dim]-[/dim]"
            )
        if player.is_captain:
            marker = "(TC)" if triple_captain else "(C)"
            lines.append(f"[yellow]{marker}[/yellow]")
        elif player.is_vice_captain:
            lines.append("[cyan](VC)[/cyan]")
        player_display = "\n".join(lines)
        if triple_captain and player.is_captain:
            return Panel(player_display, border_style="green", padding=(0, 1))
        return player_display

    formation = Table.grid(expand=True, padding=(0, 1))
    for _ in range(5):
        formation.add_column(justify="center", ratio=1)

    positions = list(Position.back_to_front())
    for index, position in enumerate(positions):
        starters = [
            player
            for player in squad.players
            if player.position == position and player.is_starting
        ]
        slots = FORMATION_SLOTS[len(starters)]
        cells = iter(player_cell(player) for player in starters)
        formation.add_row(
            *(next(cells) if slot in slots else "" for slot in range(5)),
        )
        if index < len(positions) - 1:
            formation.add_row(*([""] * 5))

    substitutes = [player for player in squad.players if not player.is_starting]
    substitutes.sort(
        key=lambda player: (
            player.sub_position is None,
            player.sub_position if player.sub_position is not None else 0,
        )
    )
    substitutes_table = Table(
        show_header=False,
        box=None,
        border_style=None,
        expand=True,
        padding=(0, 1),
    )
    for _ in substitutes:
        substitutes_table.add_column(justify="center", ratio=1)
    substitutes_table.add_row(*(player_cell(player) for player in substitutes))
    # Highlighted with a border when the bench counts, so the name has to widen from
    # Table to any renderable.
    bench: RenderableType = substitutes_table
    if bench_boost:
        bench = Panel(substitutes_table, border_style="green")

    renderables: list[RenderableType] = []
    if predicted_points is not None and gameweek is not None:
        heading = (
            f"GAMEWEEK {gameweek}\n"
            f"{predicted_points:.1f}pts predicted {chip_description}, "
            f"£{squad.budget / 10:.1f}M in the bank"
        )
        renderables.append(Text(f"{heading}", style="bold", justify="center"))

    renderables.extend(
        [
            Panel(formation, title="Starting Lineup"),
            Panel(bench, title="Substitutes"),
        ]
    )
    return Group(*renderables)
