"""Presenting predicted points: Rich tables and Discord payloads."""

import regex as re
from curl_cffi import requests
from sqlalchemy.orm import Session

from airsenal.core.console import console, table
from airsenal.core.enums import Position
from airsenal.core.logging import get_logger
from airsenal.db.models import Player
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.db.queries.predictions import get_predicted_points
from airsenal.db.queries.tags import get_latest_prediction_tag
from airsenal.db.session import get_session
from airsenal.domain.season import CURRENT_SEASON
from airsenal.fetch.fpl_api import get_fetcher

logger = get_logger(__name__)


def get_top_predicted_points(
    gameweek: int | list[int] | None = None,
    tag: str | None = None,
    position: str = "all",
    team: str = "all",
    n_players: int = 10,
    per_position: bool = False,
    max_price: float | None = None,
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> None:
    """
    Print players with the top predicted points.

    Keyword Arguments:
        gameweek {int or list} -- Single gameweek or list of gameweeks in which
        case returned totals are sums across all gameweeks (default: next
        gameweek).
        tag {str} -- Prediction tag to query (default: latest prediction tag)
        position {str} -- Player position to query (default: {"all"})
        per_position {boolean} -- If True print top n_players players for
        each position separately (default: {False})
        team {str} -- Team to query (default: {"all"})
        n_players {int} -- Number of players to return (default: {10})
        season {str} -- Season to query (default: {CURRENT_SEASON})
        dbsession {SQLAlchemy session} -- Database session (default: {None})
    """
    dbsession = dbsession if dbsession is not None else get_session()
    discord_webhook = get_fetcher().DISCORD_WEBHOOK
    if not tag:
        tag = get_latest_prediction_tag()
    if not gameweek:
        gameweek = next_gameweek()

    discord_embed = {
        "title": "AIrsenal webhook",
        "description": f"PREDICTED TOP {n_players} PLAYERS FOR GAMEWEEK(S) {gameweek}:",
        "color": 0x35A800,
        "fields": [],
    }

    first_gw = gameweek[0] if isinstance(gameweek, list) else gameweek
    gw_range = (
        f"{first_gw}–{gameweek[-1]}"  # noqa: RUF001
        if isinstance(gameweek, list) and gameweek[-1] != first_gw
        else f"{first_gw}"
    )
    table_title = f"Top {n_players} Predicted Players for Gameweek(s) {gw_range}"

    def print_predictions(predictions: list[tuple[Player, float]], title: str) -> None:
        prediction_table = table(
            "#", "Player", "Team", "Position", "Price", "Predicted Points", title=title
        )
        for rank, (player, predicted_points) in enumerate(predictions[:n_players], 1):
            price = player.price(season, first_gw)
            price_string = f"£{price / 10}m" if price is not None else "Unknown"
            prediction_table.add_row(
                str(rank),
                str(player),
                str(player.team(season, first_gw)),
                str(player.position(season)),
                price_string,
                f"{predicted_points:.2f}",
            )
        console.print(prediction_table)

    if not per_position:
        pts = get_predicted_points(
            gameweek,
            tag,
            position=position,
            team=team,
            season=season,
            dbsession=dbsession,
        )
        if max_price is not None:
            for p in pts:
                price = p[0].price(season, first_gw)
                if price is not None and price > max_price:
                    pts.remove(p)

        pts = sorted(pts, key=lambda x: x[1], reverse=True)

        print_predictions(pts, table_title)

        # If a valid discord webhook URL has been stored
        # in env variables, send a webhook message
        if discord_webhook:
            # Use regex to check the discord webhook url is correctly formatted
            if re.match(
                r"^.*(discord|discordapp)\.com\/api"
                r"\/webhooks\/([\d]+)\/([a-zA-Z0-9_-]+)$",
                discord_webhook,
            ):
                # Maximum fields on a discord embed is 25, so limit this to n_players=8
                payload = predicted_points_discord_payload(
                    discord_embed=discord_embed,
                    position=position,
                    pts=pts[: min(n_players, 8)],
                    season=season,
                    first_gw=first_gw,
                )
                result = requests.post(discord_webhook, json=payload)
                if 200 <= result.status_code < 300:
                    logger.info(
                        "Discord webhook sent, status code: %s", result.status_code
                    )
                else:
                    logger.warning(
                        "Not sent with %s,response:\n{result.json()}",
                        result.status_code,
                    )
            else:
                logger.warning("Discord webhook url is malformed!\n%s", discord_webhook)
    else:
        for i, position in enumerate(list(Position.back_to_front())):
            pts = get_predicted_points(
                gameweek,
                tag,
                position=position,
                team=team,
                season=season,
                dbsession=dbsession,
            )
            if max_price is not None:
                for p in pts:
                    maybe_price = p[0].price(season, first_gw)
                    if maybe_price is not None and maybe_price > max_price:
                        pts.remove(p)

            pts = sorted(pts, key=lambda x: x[1], reverse=True)
            title = f"{table_title}\n{position}" if i == 0 else position
            print_predictions(pts, title)

            discord_embed["fields"] = []
            # If a valid discord webhook URL has been stored
            # in env variables, send a webhook message
            if discord_webhook is not None:
                # Use regex to check the discord webhook url is correctly formatted
                if re.match(
                    r"^.*(discord|discordapp)\.com\/api"
                    r"\/webhooks\/([\d]+)\/([a-zA-Z0-9_-]+)$",
                    discord_webhook,
                ):
                    # create a formatted team lineup message for the discord webhook
                    # Maximum fields on a discord embed is 25
                    # limit this to n_players=8
                    payload = predicted_points_discord_payload(
                        discord_embed=discord_embed,
                        position=position,
                        pts=pts[: min(n_players, 8)],
                        season=season,
                        first_gw=first_gw,
                    )
                    result = requests.post(discord_webhook, json=payload)
                    if 200 <= result.status_code < 300:
                        logger.info(
                            "Discord webhook sent, status code: %s", result.status_code
                        )
                    else:
                        logger.warning(
                            "Not sent with %s, response:\n%s",
                            result.status_code,
                            result.json(),
                        )
                else:
                    logger.warning(
                        "Discord webhook url is malformed!\n%s", discord_webhook
                    )


def predicted_points_discord_payload(
    discord_embed: dict,
    position: str,
    pts: list[tuple[Player, float]],
    season: str,
    first_gw: int,
) -> dict:
    """
    json formated discord webhook contentent.
    """
    discord_embed["fields"].append(
        {
            "name": "Position",
            "value": str(position),
            "inline": False,
        }
    )
    for i, p in enumerate(pts):
        price = p[0].price(season, first_gw)
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
                        f"{p[0].position(season)}, {p[0].team(season, first_gw)}"
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
