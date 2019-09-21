
"""
Functions used by the AIrsenal API
"""

from uuid import uuid4
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker

from airsenal.framework.utils import CURRENT_SEASON, list_players, get_last_finished_gameweek, fetcher
from airsenal.framework.schema import SessionTeam, engine


dbsession = scoped_session(sessionmaker(bind=engine))


def create_response(orig_response):
    """
    Add headers to the response
    """
    response = jsonify(orig_response)
    response.headers.add('Access-Control-Allow-Headers',
                         "Origin, X-Requested-With, Content-Type, Accept, x-auth")
    return response


def list_players_teams_prices(position="all", team="all"):
    return ["{} ({}): {}".format(p.name,
                                 p.team(CURRENT_SEASON),
                                 p.current_price(CURRENT_SEASON)) \
     for p in list_players(position=position,
                           team=team)]



def get_session_budget(session_id):
    """
    query the sessionbudget table in the db - there should hopefully be one and only
    one row for this session_id
    """
    budget = dbsession.query(SessionBudget).filter_by(session_id=session_id).first()
    return budget


def set_session_budget(session_id, budget):
    """
    delete the existing entry for this session_id in the sessionbudget table,
    then enter a new row
    """
    old_budget = dbsession.query(SessionBudget).filter_by(session_id=session_id).delete()
    sb = SessionBudget(session_id=session_id, budget=budget)
    dbsession.add(sb)
    dbsession.commit()
