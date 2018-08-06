"""
Useful commands to query the db
"""

from operator import itemgetter

from .schema import Base, Player, Match, Fixture, PlayerScore, PlayerPrediction, engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import and_, or_

Base.metadata.bind = engine
DBSession = sessionmaker()
session = DBSession()


def get_player_name(player_id):
    """
    lookup player name, for human readability
    """
    p = session.query(Player).filter_by(player_id=player_id).first()
    if not p:
        print("Unknown player_id {}".format(player_id))
        return None
    return p.name


def get_player_id(player_name):
    """
    lookup player id, for machine readability
    """
    p = session.query(Player).filter_by(name=player_name).first()
    if not p:
        print("Unknown player_name {}".format(player_name))
        return None
    return p.player_id


def get_player_data(player):
    """
    can call with either player name or player ID
    """
    if isinstance(player, str):
        return session.query(Player).filter_by(name=player).first()
    elif isinstance(player, int):
        return session.query(Player).filter_by(player_id=player).first()
    else:
        print("Unknown type in get_player_data request")
        return None


def list_players(position="all", team="all", order_by="current_price", verbose=False):
    """
    print list of players, and
    return a list of player_ids
    """
    q = session.query(Player)
    if team != "all":
        q = q.filter_by(team=team)
    if position != "all":
        q = q.filter_by(position=position)
    if order_by == "current_price":
        q = q.order_by(Player.current_price.desc())
    player_ids = []
    for player in q.all():
        player_ids.append(player.player_id)
        if verbose:
            print(player.name, player.team, player.position, player.current_price)
    return player_ids


def get_fixtures_for_player(player, verbose=False):
    """
    search for upcoming fixtures for a player, specified either by id or name.
    """
    player_query = session.query(Player)
    if isinstance(player,str):
        player_record = player_query.filter_by(name=player).first()
    else:
        player_record = player_query.filter_by(player_id=player).first()
    if not player_record:
        print("Couldn't find {} in database".format(player))
        return []
    team = player_record.team
    fixtures = session.query(Fixture).filter(or_ (Fixture.home_team==team,
                                                  Fixture.away_team==team))\
                                                  .order_by(Fixture.gameweek)\
                                                  .all()
    fixture_ids = []
    for fixture in fixtures:
        if verbose:
            print("{} vs {} gameweek {}".format(fixture.home_team,
                                                fixture.away_team,
                                                fixture.gameweek))
        fixture_ids.append(fixture.fixture_id)
    return fixture_ids


def get_previous_points_for_same_fixture(player, fixture_id):
    """
    Search the past matches for same fixture in past seasons,
    and how many points the player got.
    """
    if isinstance(player,str):
        player_record = session.query(Player).filter_by(name=player).first()
        if not player_record:
            print("Can't find player {}".format(player))
            return {}
        player_id = player_record.player_id
    else:
        player_id = player
    fixture = session.query(Fixture).filter_by(fixture_id=fixture_id).first()
    if not fixture:
        print("Couldn't find fixture_id {}".format(fixture_id))
        return {}
    home_team = fixture.home_team
    away_team = fixture.away_team

    previous_matches = session.query(Match).filter_by(home_team=home_team)\
                                           .filter_by(away_team=away_team)\
                                           .order_by(Match.season).all()
    match_ids = [(match.match_id,match.season) for match in previous_matches]
    previous_points = {}
    for m in match_ids:
        scores = session.query(PlayerScore).filter_by(player_id=player_id,
                                                      match_id = m[0]).all()
        for s in scores:
            previous_points[m[1]] = s.points

    return previous_points


def get_predicted_points_for_player(player, method="EP", fixture_id=None):
    """
    Query the player prediction table for a given player.
    If no fixture_id is specified, return the next fixture.
    """
    if isinstance(player,str):
        player_record = session.query(Player).filter_by(name=player).first()
        if not player_record:
            print("Can't find player {}".format(player))
            return {}
        player_id = player_record.player_id
    else:
        player_id = player
    pps = session.query(PlayerPrediction).filter_by(player_id=player_id,method=method)
    if not fixture_id:
        fixture_id = get_fixtures_for_player(player_id)[0]
    pps.filter_by(fixture_id=fixture_id)
    if not pps.first():
        print("Couldnt find prediction for player {} fixture {} method {}"\
              .format(player, fixture_id, method))
        return 0.
    return pps.first().predicted_points


def get_predicted_points(position="all",team="all",method="EP"):
    """
    Query the player_prediction table with selections, return
    list of tuples (player_id, predicted_points) ordered by predicted_points
    """
    player_ids = list_players(position, team)
    output_list = []
    for player_id in player_ids:
        predicted_score = get_predicted_points_for_player(player_id)
        output_list.append((player_id, predicted_score))
    output_list.sort(key=itemgetter(1), reverse=True)
    return output_list
