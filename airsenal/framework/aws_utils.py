"""
Useful functions for constructing Alexa responses.
"""

import os
import time

import boto3
from sqlalchemy.orm import sessionmaker

from airsenal.framework.fpl_team_utils import (
    get_league_standings,
    get_overall_ranking,
    get_overall_points
)


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
        return "Problem initializing client {}".format(e)
    try:
        client.download_file(bucket_name, "data.db", "/tmp/datas3.db")
        return "OK"
    except Exception as e:
        return "Problem downloading file {}".format(e)


def get_league_standings_string():
    """
    Query the FPL API for our mini-league.
    """
    output_string = ""
    try:
        league_name, standings = get_league_standings()
        output_string += "Standings for league {} :".format(league_name)
        for i, entry in enumerate(standings):
            output_string += "{}: {}, managed by {}, with {} points, ".format(
                i + 1, entry["name"], entry["manager"], entry["points"]
            )
        return output_string
    except Exception as e:
        return "Problem {}".format(e)


def get_suggestions_string():
    """
    Query the suggested_transfers table and format the output.
    """
    ##  first need to download sqlite file from S3

    result = download_sqlite_file()
    if result != "OK":
        return result

    time.sleep(1)
    try:
        from airsenal.framework.schema import Player, TransferSuggestion, Base, engine

        Base.metadata.bind = engine
        DBSession = sessionmaker()
        session = DBSession()
    except Exception as e:
        return "Problem importing stuff {}".format(e)
    try:
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
                output_string += " gameweek {}: ".format(row.gameweek)
                current_gw = row.gameweek
            if row.in_or_out < 0:
                output_string += " sell "
            else:
                output_string += " buy "
            player_name = (
                session.query(Player).filter_by(player_id=row.player_id).first().name
            )
            output_string += player_name + ","

        points_gain = round(rows[0].points_gain, 1)
        output_string += " for a total gain of {} points.".format(points_gain)
        return output_string
    except Exception as e:
        return "Problem with the query {}".format(e)


def get_score_ranking_string(query, gameweek=None):
    """
    query the FPL API for team history.
    """
    f = get_overall_ranking if query == "ranking" else get_overall_points
    result = f(gameweek)
    output_string = "Our {} ".format(query)
    if gameweek:
        output_string += "for gameweek {} ".format(gameweek)
    output_string += "is {}".format(result)
    return output_string
