"""
function to be called by Alexa Skill, read
suggested transfers from sqlite file on an S3 bucket, and
return a response.
"""
import sys
#sys.path.append("..")

import json
import requests
import logging
logging.basicConfig()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

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

def get_suggestions_string():
    from framework.schema import TransferSuggestion
    from framework.utils import session, get_player_name
    client = boto3.client('s3')
    obj = client.get_object(Bucket='my-bucket', Key='data.db')
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
        output_string += get_player_name(row.player_id)+","
    points_gain = round(rows[0].points_gain,1)
    output_string += " for a total gain of {} points."\
                     .format(points_gain)
    return output_string


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
        except:
            response_text = "Not sure what you meant there"

    response_tmpl["response"]["outputSpeech"]["text"] = response_text
    return response_tmpl
