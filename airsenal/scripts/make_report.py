"""
Produce a single readable summary of the latest predictions and suggestions:
squad, starting XI, captain, transfers and chip status.

The pipeline prints this information across several stages as it runs, so by the
time it finishes the useful parts have scrolled away. This gathers them from the
database into one markdown report.
"""

import argparse
import sys
from datetime import datetime

from airsenal.framework.optimization_utils import get_starting_squad
from airsenal.framework.schema import session, session_scope
from airsenal.framework.utils import (
    CURRENT_SEASON,
    NEXT_GAMEWEEK,
    fetcher,
    get_bank,
    get_free_transfers,
    get_latest_prediction_tag,
    get_player_name,
    get_predicted_points,
)
from airsenal.scripts.fill_transfersuggestion_table import get_available_chips
from airsenal.scripts.get_transfer_suggestions import get_transfer_suggestions

POSITION_ORDER = ["GK", "DEF", "MID", "FWD"]


def format_price(price: int | None) -> str:
    return "?" if price is None else f"£{price / 10:.1f}m"


def squad_section(squad, gameweek: int, tag: str) -> list[str]:
    """The 15 players, split into the XI the optimiser would pick and the bench."""
    squad.get_expected_points(gameweek, tag)

    lines = ["## Squad", ""]
    formation = squad.get_formation()
    lines.append(
        f"Starting XI ({formation['DEF']}-{formation['MID']}-{formation['FWD']}):"
    )
    lines.append("")
    lines.append("| Pos | Player | Team | Price | Pred pts |")
    lines.append("| --- | --- | --- | --- | --- |")

    starting = [p for p in squad.players if p.is_starting]
    starting.sort(key=lambda p: POSITION_ORDER.index(p.position))
    for player in starting:
        marker = (
            " (C)" if player.is_captain else " (VC)" if player.is_vice_captain else ""
        )
        points = player.predicted_points[tag].get(gameweek, 0.0)
        lines.append(
            f"| {player.position} | {player}{marker} | {player.team} | "
            f"{format_price(player.purchase_price)} | {points:.2f} |"
        )

    lines += ["", "Bench, in the order they would come on:", ""]
    lines.append("| Order | Pos | Player | Team | Pred pts |")
    lines.append("| --- | --- | --- | --- | --- |")
    bench = sorted(
        (p for p in squad.players if not p.is_starting),
        key=lambda p: p.sub_position if p.sub_position is not None else 99,
    )
    for i, player in enumerate(bench, start=1):
        points = player.predicted_points[tag].get(gameweek, 0.0)
        lines.append(
            f"| {i} | {player.position} | {player} | {player.team} | {points:.2f} |"
        )
    return lines


def captain_section(squad, gameweek: int, tag: str) -> list[str]:
    """Captain, and how far clear of the next best option they are - a narrow
    margin is a coin toss, not a recommendation."""
    ranked = sorted(
        ((p, p.predicted_points[tag].get(gameweek, 0.0)) for p in squad.players),
        key=lambda x: x[1],
        reverse=True,
    )
    if len(ranked) < 2:
        return []

    (best, best_pts), (second, second_pts) = ranked[0], ranked[1]
    margin = best_pts - second_pts
    lines = [
        "",
        "## Captain",
        "",
        f"**{best}** ({best_pts:.2f} pts), ahead of {second} ({second_pts:.2f}) "
        f"by {margin:.2f}.",
    ]
    if margin < 0.5:
        lines.append("")
        lines.append(
            "That margin is well inside the model's own uncertainty - treat the two "
            "as equivalent and pick on fixture or ownership."
        )
    return lines


def transfer_section(season: str, fpl_team_id: int, dbsession) -> list[str]:
    rows = get_transfer_suggestions(dbsession, season=season, fpl_team_id=fpl_team_id)
    lines = ["", "## Suggested transfers", ""]
    if not rows:
        lines.append("No transfer suggestions in the database - run the optimisation.")
        return lines

    by_gameweek: dict[int, dict[str, list[str]]] = {}
    for row in rows:
        entry = by_gameweek.setdefault(row.gameweek, {"in": [], "out": []})
        name = str(get_player_name(row.player_id, dbsession=dbsession))
        entry["in" if row.in_or_out > 0 else "out"].append(name)

    for gw in sorted(by_gameweek):
        entry = by_gameweek[gw]
        if not entry["in"] and not entry["out"]:
            lines.append(f"- Gameweek {gw}: no transfers")
            continue
        lines.append(
            f"- Gameweek {gw}: out {', '.join(entry['out']) or '-'} "
            f"| in {', '.join(entry['in']) or '-'}"
        )

    chip = rows[0].chip_played
    if chip:
        lines += ["", f"Chip to play: **{chip}**"]

    lines += [
        "",
        f"Total predicted gain over making no transfers: "
        f"**{rows[0].points_gain:.2f} pts**.",
    ]
    if rows[0].points_gain < 2:
        lines.append("")
        lines.append(
            "A gain of under 2 points is small next to how uncertain the "
            "predictions are - rolling the transfer is a reasonable alternative."
        )
    return lines


def top_players_section(
    gameweek: int, tag: str, season: str, n_players: int, dbsession
) -> list[str]:
    lines = ["", f"## Top predicted players, gameweek {gameweek}", ""]
    lines.append("| Player | Team | Pos | Pred pts |")
    lines.append("| --- | --- | --- | --- |")
    for player, points in get_predicted_points(
        gameweek=gameweek, tag=tag, season=season, dbsession=dbsession
    )[:n_players]:
        lines.append(
            f"| {player} | {player.team(season, gameweek)} | "
            f"{player.position(season)} | {points:.2f} |"
        )
    return lines


def status_section(fpl_team_id: int, gameweek: int, season: str) -> list[str]:
    lines = ["", "## Status", ""]
    try:
        free_transfers = get_free_transfers(fpl_team_id, gameweek, season=season)
        lines.append(f"- Free transfers: {free_transfers}")
    except Exception as e:
        lines.append(f"- Free transfers: unknown ({e})")
    try:
        lines.append(f"- In the bank: {format_price(get_bank(fpl_team_id, gameweek))}")
    except Exception as e:
        lines.append(f"- In the bank: unknown ({e})")

    chips = get_available_chips(fpl_team_id)
    if chips is None:
        lines.append("- Chips remaining: unknown (not logged in to the FPL API)")
    else:
        lines.append(f"- Chips remaining: {', '.join(sorted(chips)) or 'none'}")
    return lines


def make_report(
    fpl_team_id: int | None = None,
    gameweek: int = NEXT_GAMEWEEK,
    season: str = CURRENT_SEASON,
    n_players: int = 10,
    dbsession=session,
) -> str:
    if fpl_team_id is None:
        fpl_team_id = fetcher.FPL_TEAM_ID
    if fpl_team_id is None:
        msg = "No FPL team ID given, and none set in the environment"
        raise ValueError(msg)

    tag = get_latest_prediction_tag(season, dbsession=dbsession)

    lines = [
        f"# AIrsenal report - gameweek {gameweek}, {season}",
        "",
        f"Team {fpl_team_id}, generated {datetime.now():%Y-%m-%d %H:%M}.",
        "",
    ]
    lines += status_section(fpl_team_id, gameweek, season)

    try:
        squad = get_starting_squad(
            next_gw=gameweek,
            season=season,
            fpl_team_id=fpl_team_id,
            use_api=season == CURRENT_SEASON,
        )
    except (ValueError, TypeError) as e:
        lines += ["", f"Couldn't load a squad for team {fpl_team_id}: {e}"]
        squad = None

    if squad is not None:
        lines += ["", *squad_section(squad, gameweek, tag)]
        lines += captain_section(squad, gameweek, tag)

    lines += transfer_section(season, fpl_team_id, dbsession)
    lines += top_players_section(gameweek, tag, season, n_players, dbsession)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Summarise the latest AIrsenal predictions and suggestions"
    )
    parser.add_argument("--fpl_team_id", help="specify fpl team id", type=int)
    parser.add_argument("--gameweek", help="gameweek to report on", type=int)
    parser.add_argument("--season", help="season, e.g. '2526'", default=CURRENT_SEASON)
    parser.add_argument(
        "--n_players", help="how many top players to list", type=int, default=10
    )
    parser.add_argument(
        "--output", help="write the report to this file instead of stdout"
    )
    args = parser.parse_args()

    with session_scope() as dbsession:
        report = make_report(
            fpl_team_id=args.fpl_team_id,
            gameweek=args.gameweek or NEXT_GAMEWEEK,
            season=args.season,
            n_players=args.n_players,
            dbsession=dbsession,
        )

    if args.output:
        with open(args.output, "w") as f:
            f.write(report + "\n")
        print(f"Wrote report to {args.output}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
