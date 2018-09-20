"""
function to be called by Alexa Skill, read
suggested transfers from sqlite file on an S3 bucket, and
return a response.
"""
import os
import sys
import time

import logging

logging.basicConfig()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

from framework.aws_utils import *

response_tmpl = {
    "version": "1.0",
    "response": {
        "outputSpeech": {"type": "PlainText", "text": ""},
        "shouldEndSession": True,
    },
}


print("Loading AIrsenal function")


def lambda_handler(event, context):
    logger.info("got event{}".format(event))

    response_text = "A.I. Arsenal forever."
    if event["request"]["intent"]["name"] == "Question":

        try:
            if "value" in event["request"]["intent"]["slots"]["Topic"].keys():
                topic = event["request"]["intent"]["slots"]["Topic"]["value"]
                if topic == "best manager":
                    response_text = "Hmmm that's a tough one.  Both Angus and Nick are pretty good, but I think I am the best."
                    pass
                elif topic == "transfer":
                    response_text = get_suggestions_string()
                elif topic == "score" or topic == "ranking":
                    if (
                        "Gameweek" in event["request"]["intent"]["slots"].keys()
                        and "value"
                        in event["request"]["intent"]["slots"]["Gameweek"].keys()
                    ):
                        gameweek = event["request"]["intent"]["slots"]["Gameweek"][
                            "value"
                        ]
                    else:
                        gameweek = None
                    response_text = get_score_ranking_string(topic, gameweek)
                elif topic == "league":
                    response_text = get_league_standings_string()
                else:
                    response_text = "unknown query"
        except Exception as e:
            response_text = "Exception {}".format(e)

    response_tmpl["response"]["outputSpeech"]["text"] = response_text
    return response_tmpl
