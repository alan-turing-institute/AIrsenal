#!/usr/bin/env python

"""
API for calling airsenal functions.
HTTP requests to the endpoints defined here will give rise
to calls to functions in api_utils.py
"""
import json
from uuid import uuid4

from flask import Blueprint, Flask, jsonify, request, session
from flask_cors import CORS
from flask_session import Session

from airsenal.api.exceptions import ApiException
from airsenal.framework.api_utils import (
    add_session_player,
    best_transfer_suggestions,
    combine_player_info,
    create_response,
    fill_session_squad,
    get_session_budget,
    get_session_players,
    get_session_predictions,
    list_players_for_api,
    list_teams_for_api,
    remove_db_session,
    remove_session_player,
    set_session_budget,
    validate_session_squad,
)


def get_session_id():
    """
    Get the ID from the flask_session Session instance if
    it exists, otherwise just get a default string, which
    will enable us to test some functionality just via python requests.
    """
    print(f"Session keys {session.keys()}")
    if "key" in session.keys():
        return session["key"]
    else:
        return "DEFAULT_SESSION_ID"


# Use a flask blueprint rather than creating the app directly
# so that we can also make a test app

blueprint = Blueprint("airsenal", __name__)


@blueprint.errorhandler(ApiException)
def handle_exception(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@blueprint.teardown_request
def remove_session(ex=None):
    remove_db_session()


@blueprint.route("/teams", methods=["GET"])
def get_team_list():
    """
    Return a list of all teams for the current season
    """
    team_list = list_teams_for_api()
    return create_response(team_list)


@blueprint.route("/players/<team>/<pos>", methods=["GET"])
def get_player_list(team, pos):
    """
    Return a list of all players in that team and/or position
    """
    player_list = [
        {"id": p.player_id, "name": p.name}
        for p in list_players_for_api(position=pos, team=team)
    ]
    return create_response(player_list)


@blueprint.route("/new")
def set_session_key():
    """
    Create a new and unique session ID
    """
    key = str(uuid4())
    session["key"] = key
    return create_response(key)


def reset_session_team(param):
    pass


@blueprint.route("/team/new")
def reset_team():
    """
    Remove all players from the DB table with this session_id and
    reset the budget to 100M
    """
    reset_session_team(get_session_id())
    return create_response("OK")


@blueprint.route("/player/<player_id>")
def get_player_info(player_id):
    """
    Return a dict containing player's name, team, recent points,
    and upcoming fixtures and predictions.
    """
    player_info = combine_player_info(player_id)
    return create_response(player_info)


@blueprint.route("/team/add/<player_id>")
def add_player(player_id):
    """
    Add a selected player to this session's squad.
    """
    added_ok = add_session_player(player_id, session_id=get_session_id())
    return create_response(added_ok)


@blueprint.route("/team/remove/<player_id>")
def remove_player(player_id):
    """
    Remove selected player to this session's squad.
    """
    removed_ok = remove_session_player(player_id, session_id=get_session_id())
    return create_response(removed_ok)


@blueprint.route("/team/list", methods=["GET"])
def list_session_players():
    """
    List all players currently in this session's squad.
    """
    player_list = get_session_players(session_id=get_session_id())
    return create_response(player_list)


@blueprint.route("/team/pred", methods=["GET"])
def list_session_predictions():
    """
    Get predicted points for all players in this sessions squad
    """
    pred_dict = get_session_predictions(session_id=get_session_id())
    return create_response(pred_dict)


@blueprint.route("/team/validate", methods=["GET"])
def validate_session_players():
    """
    Check that the squad has 15 players, and obeys constraints.
    """
    valid = validate_session_squad(session_id=get_session_id())
    return create_response(valid)


@blueprint.route("/team/fill/<team_id>")
def fill_team_from_team_id(team_id):
    """
    Use the ID of a team in the FPL API to fill a squad for this session.
    """
    player_ids = fill_session_squad(team_id=team_id, session_id=get_session_id())
    return create_response(player_ids)


@blueprint.route("/team/optimize/<n_transfers>")
def get_optimum_transfers(n_transfers):
    """
    Find the best n_transfers transfers for the next gameweek.
    """
    transfers = best_transfer_suggestions(n_transfers, session_id=get_session_id())
    return create_response(transfers)


@blueprint.route("/budget", methods=["GET", "POST"])
def session_budget():
    """
    Set or get the budget for this team.
    """
    if request.method != "POST":
        return create_response(get_session_budget(get_session_id()))

    data = json.loads(request.data.decode("utf-8"))
    budget = data["budget"]
    set_session_budget(budget, get_session_id())
    return create_response("OK")


def create_app(name=__name__):
    app = Flask(name)
    app.config["SESSION_TYPE"] = "filesystem"
    app.secret_key = "blah"
    CORS(app, supports_credentials=True)
    app.register_blueprint(blueprint)
    Session(app)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5002, debug=True)
