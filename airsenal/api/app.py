#!/usr/bin/env python

"""
API for calling airsenal functions.
"""

import os
import sys
from flask import Blueprint, Flask, Response, session, request, jsonify
from flask_cors import CORS
from flask_session import Session
import requests
import json
from uuid import uuid4
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker

from airsenal.framework.utils import CURRENT_SEASON, list_players, get_last_finished_gameweek, fetcher
from airsenal.framework.api_utils import *
from airsenal.framework.schema import SessionTeam, engine

from exceptions import ApiException


def get_session_id():
    """
    Get the ID from the flask_session Session instance if
    it exists, otherwise just get a default string, which
    will enable us to test some functionality just via python requests.
    """
    if "key" in session.keys():
        return session["key"]
    else:
        return "DEFAULT_SESSION_ID"


## Use a flask blueprint rather than creating the app directly
## so that we can also make a test app

blueprint = Blueprint("airsenal",__name__)

@blueprint.errorhandler(ApiException)
def handle_exception(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

@blueprint.teardown_request
def remove_session(ex=None):
    dbsession.remove()


@blueprint.route("/players/<team>/<pos>", methods=["GET"])
def get_player_list(team, pos):
    """
    return a list of all players in that team and/or position
    """
    return create_response(list_players_teams_prices(position=pos,
                                                     team=team))


@blueprint.route("/new")
def set_session_key():
    key = str(uuid4())
    session['key'] = key
    return create_response(key)


@blueprint.route("/team/new")
def reset_team():
    dbsession.query(SessionTeam).filter_by(session_id=get_session_id()).delete()
    sb = SessionBudget(session_id=get_session_id(), budget=1000)
    dbsession.add(sb)
    dbsession.commit()
    return create_response("OK")


@blueprint.route("/team/add/<player_id>")
def add_player(player_id):
    st = SessionTeam(session_id=get_session_id(), player_id=player_id)
    dbsession.add(st)
    dbsession.commit()
    return create_response("OK")


@blueprint.route("/team/remove/<player_id>")
def remove_player(player_id):
    st = dbsession.query(SessionTeam).filter_by(session_id=get_session_id(),
                                                player_id=player_id).delete()
    dbsession.commit()
    return create_response("OK")


@blueprint.route("/team/list",methods=["GET"])
def list_session_players():
    players = dbsession.query(SessionTeam).filter_by(session_id=session['key']).all()
    player_list = [p.player_id for p in players]
    return create_response(player_list)


@blueprint.route("/team/fill/<team_id>")
def fill_team_from_team_id(team_id):
    players = fetcher.get_fpl_team_data(get_last_finished_gameweek(), team_id)
    player_ids = [p['element'] for p in players]
    for pid in player_ids:
        add_player(pid)
    team_history = fetcher.get_fpl_team_history_data()['current']
    index = get_last_finished_gameweek() - 1 # as gameweek starts counting from 1 but list index starts at 0
    budget = team_history[index]['value']
    set_session_budget(budget)
    return create_response(player_ids)


@blueprint.route("/budget", methods=["GET","POST"])
def session_budget():

    if request.method == 'POST':
        data = json.loads(request.data.decode("utf-8"))
        budget = data["budget"]
        set_session_budget(session['key'], budget)
        return create_response("OK")
    else:
        return create_response(get_session_budget(session['key']))


###########################################

def create_app(name = __name__):
    app = Flask(name)
    app.config['SESSION_TYPE'] = 'filesystem'
    app.secret_key = 'blah'
    CORS(app, supports_credentials=True)
    app.register_blueprint(blueprint)
    Session(app)
    return app


if __name__ == "__main__":

    app = create_app()
    app.run(host='0.0.0.0',port=5002, debug=True)
