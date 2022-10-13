"""
Useful functions for constructing Alexa responses.
"""

import os
import time

import boto3

from airsenal.framework.fpl_team_utils import (
    get_league_standings,
    get_overall_points,
    get_overall_ranking,
)
from airsenal.framework.schema import Player, TransferSuggestion


def download_sqlite_file():
    """
    get from S3 using boto3
    """
    bucket_name = os.environ["BUCKET_NAME"]
    try:
        client = boto3.client(
            "s3",
            aws_access_key_id=os.environ["KEY_ID"],
            aws_secret_access_key=os.environ["ACCESS_KEY"],
        )
    except Exception as e:
        return f"Problem initializing client {e}"
    try:
        client.download_file(bucket_name, "data.db", "/tmp/datas3.db")
        return "OK"
    except Exception as e:
        return f"Problem downloading file {e}"


def get_league_standings_string():
    """
    Query the FPL API for our mini-league.
    """
    output_string = ""
    try:
        league_name, standings = get_league_standings()
        output_string += f"Standings for league {league_name} :"
        for i, entry in enumerate(standings):
            output_string += (
                f"{i + 1,}: "
                f"{entry['name']}, "
                f"managed by {entry['manager']}, "
                f"with {entry['points']} points, "
            )
        return output_string
    except Exception as e:
        return f"Problem {e}"


def get_suggestions_string():
    """
    Query the suggested_transfers table and format the output.
    """
    #  first need to download sqlite file from S3

    result = download_sqlite_file()
    if result != "OK":
        return result

    time.sleep(1)
    try:
        from airsenal.framework.schema import session
    except Exception as e:
        return f"Problem importing stuff {e}"
    try:
        return build_suggestion_string(session, TransferSuggestion, Player)

    except Exception as e:
        return f"Problem with the query {e}"


def build_suggestion_string(session, TransferSuggestion, Player):
    all_rows = session.query(TransferSuggestion).all()
    last_timestamp = all_rows[-1].timestamp
    rows = (
        session.query(TransferSuggestion)
        .filter_by(timestamp=last_timestamp)
        .order_by(TransferSuggestion.gameweek)
        .all()
    )
    output_string = "Suggested transfer strategy: \n"
    current_gw = 0
    for row in rows:
        if row.gameweek != current_gw:
            output_string += f" gameweek {row.gameweek}: "
            current_gw = row.gameweek
        output_string += " sell " if row.in_or_out < 0 else " buy "
        player_name = (
            session.query(Player).filter_by(player_id=row.player_id).first().name
        )
        output_string += player_name + ","

    points_gain = round(rows[0].points_gain, 1)
    output_string += f" for a total gain of {points_gain} points."
    return output_string


def get_score_ranking_string(query, gameweek=None):
    """
    query the FPL API for team history.
    """
    f = get_overall_ranking if query == "ranking" else get_overall_points
    result = f(gameweek)
    output_string = f"Our {query} "
    if gameweek:
        output_string += f"for gameweek {gameweek} "
    output_string += f"is {result}"
    return output_string
