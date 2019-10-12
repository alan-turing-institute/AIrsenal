
"""
Functions used by the AIrsenal API
"""

from uuid import uuid4
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker
from flask import jsonify

from airsenal.framework.utils import (
    CURRENT_SEASON,
    fetcher,
    list_players,
    get_last_finished_gameweek,
    get_latest_prediction_tag,
    get_next_gameweek,
    get_predicted_points_for_player,
    get_fixtures_for_player,
    get_next_fixture_for_player,
    get_player,
    get_recent_scores_for_player
)

from airsenal.framework.team import Team
from airsenal.framework.schema import SessionTeam, SessionBudget, engine
from airsenal.framework.optimization_utils import (
    make_optimum_transfer,
    make_optimum_double_transfer
)

DBSESSION = scoped_session(sessionmaker(bind=engine))


def remove_db_session(dbsession=DBSESSION):
    dbsession.remove()


def create_response(orig_response, dbsession=DBSESSION):
    """
    Add headers to the response
    """
    response = jsonify(orig_response)
    response.headers.add('Access-Control-Allow-Headers',
                         "Origin, X-Requested-With, Content-Type, Accept, x-auth")
    return response


def reset_session_team(session_id, dbsession=DBSESSION):
    """
    remove any rows with the given session ID and add a new budget of
    100M
    """
    # remove all players with this session id
    dbsession.query(SessionTeam).filter_by(session_id=session_id).delete()
    dbsession.commit()
    # now remove the budget, and make a new one
    dbsession.query(SessionBudget).filter_by(session_id=session_id).delete()
    dbsession.commit()
    sb = SessionBudget(session_id=session_id, budget=1000)
    dbsession.add(sb)
    dbsession.commit()
    return True



def combine_player_info(player_id, dbsession=DBSESSION):
    """
    Get player's name, club, recent scores, upcoming fixtures, and 
    upcoming predictions if available
    """
    info_dict = {"player_id": player_id}
    p = get_player(player_id)
    info_dict["player_name"] = p.name
    team = p.team(CURRENT_SEASON)
    info_dict["team"] = team
    ## get recent scores for the player
    recent_scores = get_recent_scores_for_player(p, dbsession=dbsession)
    info_dict["recent_scores"] = recent_scores
    ## get upcoming fixtures
    fixtures = get_fixtures_for_player(p, dbsession=dbsession)[:3]
    info_dict["fixtures"] = []
    for f in fixtures:
        is_home = True if f.home_team==team else False
        opponent = f.away_team if is_home else f.home_team
        info_dict["fixtures"].append({"gameweek": f.gameweek,
                                      "opponent": opponent,
                                      "home": is_home})
    try:
        tag = get_latest_prediction_tag()
        predicted_points = get_predicted_points_for_player(p,tag)
        info_dict[predicted_points] = predicted_points
    except(RuntimeError):
        pass
    return info_dict
        


def add_session_player(player_id, session_id, dbsession=DBSESSION):
    """
    Add a row in the SessionTeam table.
    """
    pids = get_session_players(session_id, dbsession)
    if player_id in pids: # don't add the same player twice!
        return False
    st = SessionTeam(session_id=session_id, player_id=player_id)
    dbsession.add(st)
    dbsession.commit()
    return True


def remove_session_player(player_id, session_id, dbsession=DBSESSION):
    """
    Remove row from SessionTeam table.
    """
    pids = get_session_players(session_id, dbsession)
    player_id = int(player_id)
    if player_id not in pids: # player not there
        return False
    st = dbsession.query(SessionTeam).filter_by(session_id=session_id,
                                                player_id=player_id)\
                                     .delete()
    dbsession.commit()
    return True


def list_players_teams_prices(position="all", team="all", dbsession=DBSESSION):
    """
    Return a list of players, each with their current team and price
    """
    return ["{} ({}): {}".format(p.name,
                                 p.team(CURRENT_SEASON),
                                 p.current_price(CURRENT_SEASON)) \
     for p in list_players(position=position,
                           team=team,
                           dbsession=dbsession)]


def get_session_budget(session_id, dbsession=DBSESSION):
    """
    query the sessionbudget table in the db - there should hopefully
    be one and only one row for this session_id
    """

    sb = dbsession.query(SessionBudget).filter_by(session_id=session_id).all()
    if len(sb) !=1:
        raise RuntimeError("{}  SessionBudgets for session key {}"\
                           .format(len(sb),session_id))
    return sb[0].budget


def set_session_budget(budget, session_id, dbsession=DBSESSION):
    """
    delete the existing entry for this session_id in the sessionbudget table,
    then enter a new row
    """
    print("Deleting old budget")
    old_budget = dbsession.query(SessionBudget)\
                          .filter_by(session_id=session_id).delete()
    dbsession.commit()
    print("Setting budget for {} to {}".format(session_id, budget))
    sb = SessionBudget(session_id=session_id, budget=budget)
    dbsession.add(sb)
    dbsession.commit()
    return True


def get_session_players(session_id, dbsession=DBSESSION):
    """
    query the dbsession for the list of players with the requested player_id
    """
    players = dbsession.query(SessionTeam)\
                       .filter_by(session_id=session_id).all()
    player_list = [p.player_id for p in players]
    return player_list


def validate_session_squad(session_id, dbsession=DBSESSION):
    """
    get the list of player_ids for this session_id, and see if we can
    make a valid 15-player squad out of it
    """
    budget = get_session_budget(session_id, dbsession)

    players = get_session_players(session_id, dbsession)
    if len(players) != 15:
        return False
    t = Team(budget)
    for p in players:
        added_ok = t.add_player(p)
        if not added_ok:
            return False
    return True


def fill_session_team(team_id, session_id, dbsession=DBSESSION):
    """
    Use the FPL API to get list of players in an FPL squad with id=team_id,
    then fill the session team with these players.
    """
    # first reset the team
    reset_session_team(session_id, dbsession)
    # now query the API
    players = fetcher.get_fpl_team_data(get_last_finished_gameweek(),
                                        team_id)
    player_ids = [p['element'] for p in players]
    for pid in player_ids:
        add_session_player(pid, session_id, dbsession)
    team_history = fetcher.get_fpl_team_history_data()['current']
    index = get_last_finished_gameweek() - 1 # as gameweek starts counting from 1 but list index starts at 0
    budget = team_history[index]['value']
    set_session_budget(budget, session_id)
    return player_ids


def get_session_prediction(player_id, session_id,
                           gw=None, pred_tag=None,
                           dbsession=DBSESSION):
    """
    Query the fixture and predictedscore tables for a specified player
    """
    if not gw:
        gw = get_next_gameweek(CURRENT_SEASON, dbsession)
    if not pred_tag:
        pred_tag = get_latest_prediction_tag()
    return_dict = {
        "predicted_score": get_predicted_points_for_player(pid,
                                                           pred_tag,
                                                           CURRENT_SEASON,
                                                           dbsession)[gw],
        "fixture": get_next_fixture_for_player(pid,CURRENT_SEASON,dbsession)
    }
    return return_dict

def get_session_predictions(session_id, dbsession=DBSESSION):
    """
    Query the fixture and predictedscore tables for all
    players in our session squad
    """
    players = get_session_players(session_id, dbsession)
    pred_tag = get_latest_prediction_tag()
    gw = get_next_gameweek(CURRENT_SEASON, dbsession)
    pred_scores = {}
    for pid in players:

        pred_scores[pid] = get_session_prediction(pid,
                                                  session_id,
                                                  gw, pred_tag,
                                                  dbsession)
    return pred_scores


def best_transfer_suggestions(n_transfer, session_id, dbsession=DBSESSION):
    """
    Use our predicted playerscores to suggest the best transfers.
    """
    n_transfer = int(n_transfer)
    if not n_transfer in range(1,3):
        raise RuntimeError("Need to choose 1 or 2 transfers")
    if not validate_session_squad(session_id, dbsession):
        raise RuntimeError("Cannot suggest transfer without complete squad")

    budget = get_session_budget(session_id, dbsession)
    players = get_session_players(session_id, dbsession)
    t = Team(budget)
    for p in players:
        added_ok = t.add_player(p)
        if not added_ok:
            raise RuntimeError("Cannot add player {}".format(p))
    pred_tag = get_latest_prediction_tag()
    gw=get_next_gameweek(CURRENT_SEASON, dbsession)
    if n_transfer == 1:
        new_team, pid_out, pid_in = make_optimum_transfer(t, pred_tag)
    elif n_transfer == 2:
        new_team, pid_out, pid_in = make_optimum_double_transfer(t, pred_tag)
    return {
        "transfers_out": pid_out,
        "transfers_in": pid_in
    }
