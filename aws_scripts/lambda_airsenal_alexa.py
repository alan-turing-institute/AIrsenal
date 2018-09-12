"""
function to be called by Alexa Skill, read
suggested transfers from sqlite file on an S3 bucket, and
return a response.
"""
import os
import sys
import time

import json
import requests
import logging
logging.basicConfig()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
from sqlalchemy.orm import sessionmaker
import boto3


response_tmpl = {
  "version": "1.0",
  "response": {
    "outputSpeech": {
      "type": "PlainText",
      "text": "",
    },
    "shouldEndSession": True
  }
}


print('Loading function get_suggestions')

def download_sqlite_file():
    """
    get from S3 using boto3
    """
    bucket_name = os.environ["BUCKET_NAME"]
    try:
        client = boto3.client('s3',
                              aws_access_key_id=os.environ["KEY_ID"],
                              aws_secret_access_key=os.environ["ACCESS_KEY"])
    except Exception as e:
        return "Problem initializing client {}".format(e)
    try:
        client.download_file(bucket_name, 'data.db', '/tmp/datas3.db')
        return "OK"
    except Exception as e:
        return "Problem downloading file {}".format(e)


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
        from framework.schema import Player, TransferSuggestion, Base, engine
        Base.metadata.bind = engine
        DBSession = sessionmaker()
        session = DBSession()
    except Exception as e:
        return "Problem importing stuff {}".format(e)
    try:
        all_rows = session.query(TransferSuggestion).all()
        last_timestamp = all_rows[-1].timestamp
        rows = session.query(TransferSuggestion)\
                      .filter_by(timestamp=last_timestamp)\
                      .order_by(TransferSuggestion.gameweek)\
                      .all()
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
            player_name = session.query(Player)\
                                 .filter_by(player_id=row.player_id)\
                                 .first().name
            output_string += player_name+","

        points_gain = round(rows[0].points_gain,1)
        output_string += " for a total gain of {} points."\
                         .format(points_gain)
        return output_string
    except:
        return "Problem with the query"

def lambda_handler(event, context):
    logger.info('got event{}'.format(event))

    if event["request"]["intent"]["name"] == "Suggestion":
        response_text = "A.I. Arsenal forever."
        try:
            if "value" in event["request"]["intent"]["slots"]["Topic"].keys():
                topic = event["request"]["intent"]["slots"]["Topic"]["value"]
                if topic == "best manager":
                    response_text = "Hmmm that's a tough one.  Both Angus and Nick are pretty good, but I think I am the best."
                    pass
                elif topic == "transfer":
                    response_text = get_suggestions_string()
                else:
                    response_text = "unknown query"
        except Exception as e:
            response_text = "Exception {}".format(e)

    response_tmpl["response"]["outputSpeech"]["text"] = response_text
    return response_tmpl
