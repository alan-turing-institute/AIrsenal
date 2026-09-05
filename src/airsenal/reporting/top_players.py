"""Presenting predicted points: Rich tables and Discord payloads."""

from collections.abc import Iterable
from typing import Any

from sqlalchemy.orm import Session

from airsenal.core.console import console, table
from airsenal.core.logging import get_logger
from airsenal.db.models import Player
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.db.queries.predictions import get_predicted_points
from airsenal.db.queries.tags import get_latest_prediction_tag
from airsenal.db.session import get_session
from airsenal.game.enums import Position
from airsenal.game.season import CURRENT_SEASON
from airsenal.remote.discord import get_webhook_url, post_webhook

logger = get_logger(__name__)


def within_price(
    predictions: list[tuple[Player, float]],
    max_price: float | None,
    gameweek: int,
    season: str,
) -> list[tuple[Player, float]]:
    """Drop players costing more than `max_price`, keeping ones with no price."""
    if max_price is None:
        return predictions
    return [
        (player, points)
        for player, points in predictions
        if (price := player.price(gameweek, season)) is None or price <= max_price
    ]


def get_top_predicted_points(
    gameweeks: Iterable[int] | None = None,
    position: str = "all",
    team: str = "all",
    tag: str | None = None,
    n_players: int = 10,
    per_position: bool = False,
    max_price: float | None = None,
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> None:
    """
    Print the players with the top predicted points.

    Also posts them to Discord if a webhook URL is configured.

    Args:
        gameweeks: Gameweeks to total over. Defaults to the next gameweek only.
        tag: Prediction tag to query. Defaults to the latest one.
        per_position: If True, print a separate top `n_players` per position.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    discord_webhook = get_webhook_url()
    if not tag:
        tag = get_latest_prediction_tag()
    gameweeks = list(gameweeks) if gameweeks else [next_gameweek()]

    discord_embed = {
        "title": "AIrsenal webhook",
        "description": (
            f"PREDICTED TOP {n_players} PLAYERS FOR GAMEWEEK(S) {gameweeks}:"
        ),
        "color": 0x35A800,
        "fields": [],
    }

    first_gw, last_gw = gameweeks[0], gameweeks[-1]
    gameweek_label = (
        f"{first_gw}–{last_gw}"  # noqa: RUF001
        if last_gw != first_gw
        else f"{first_gw}"
    )
    table_title = f"Top {n_players} Predicted Players for Gameweek(s) {gameweek_label}"

    def print_predictions(predictions: list[tuple[Player, float]], title: str) -> None:
        prediction_table = table(
            "#", "Player", "Team", "Position", "Price", "Predicted Points", title=title
        )
        for rank, (player, predicted_points) in enumerate(predictions[:n_players], 1):
            price = player.price(first_gw, season)
            price_string = f"£{price / 10}m" if price is not None else "Unknown"
            prediction_table.add_row(
                str(rank),
                str(player),
                str(player.team(first_gw, season)),
                str(player.position(season)),
                price_string,
                f"{predicted_points:.2f}",
            )
        console.print(prediction_table)

    if not per_position:
        pts = get_predicted_points(
            gameweeks,
            position=position,
            team=team,
            tag=tag,
            season=season,
            dbsession=dbsession,
        )
        pts = within_price(pts, max_price, first_gw, season)
        pts = sorted(pts, key=lambda x: x[1], reverse=True)

        print_predictions(pts, table_title)

        # Maximum fields on a discord embed is 25, so limit this to n_players=8
        post_webhook(
            predicted_points_discord_payload(
                discord_embed=discord_embed,
                position=position,
                pts=pts[: min(n_players, 8)],
                season=season,
                first_gw=first_gw,
            ),
            discord_webhook,
        )
    else:
        for i, each_position in enumerate(list(Position.back_to_front())):
            pts = get_predicted_points(
                gameweeks,
                position=each_position,
                team=team,
                tag=tag,
                season=season,
                dbsession=dbsession,
            )
            pts = within_price(pts, max_price, first_gw, season)
            pts = sorted(pts, key=lambda x: x[1], reverse=True)
            title = f"{table_title}\n{each_position}" if i == 0 else str(each_position)
            print_predictions(pts, title)

            discord_embed["fields"] = []
            # Maximum fields on a discord embed is 25, so limit this to n_players=8
            post_webhook(
                predicted_points_discord_payload(
                    discord_embed=discord_embed,
                    position=each_position,
                    pts=pts[: min(n_players, 8)],
                    season=season,
                    first_gw=first_gw,
                ),
                discord_webhook,
            )


def predicted_points_discord_payload(
    discord_embed: dict[str, Any],
    position: str,
    pts: list[tuple[Player, float]],
    season: str,
    first_gw: int,
) -> dict[str, Any]:
    """The Discord webhook payload for a table of predicted points."""
    discord_embed["fields"].append(
        {
            "name": "Position",
            "value": str(position),
            "inline": False,
        }
    )
    for i, p in enumerate(pts):
        price = p[0].price(first_gw, season)
        price_str = str(price / 10) if price is not None else "UNKNOWN_PRICE"
        discord_embed["fields"].extend(
            [
                {
                    "name": "Player",
                    "value": f"{i + 1}. {p[0]}",
                    "inline": True,
                },
                {
                    "name": "Predicted points",
                    "value": f"{p[1]:.2f}pts",
                    "inline": True,
                },
                {
                    "name": "Attributes",
                    "value": (
                        f"£{price_str}m, "
                        f"{p[0].position(season)}, {p[0].team(first_gw, season)}"
                    ),
                    "inline": True,
                },
            ]
        )
    return {
        "content": "",
        "username": "AIrsenal",
        "embeds": [discord_embed],
    }
