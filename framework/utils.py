"""
Useful commands to query the db
"""

from .schema import Base, Player, Match, Fixture, PlayerScore, engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import and_, or_

Base.metadata.bind = engine
DBSession = sessionmaker()
session = DBSession()


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


def list_players(position="all", team="all", order_by="current_price"):
    """
    print list of players, and
    return a list of player_ids
    """
    q = session.query(Player).order_by(Player.current_price.desc())
    if team != "all":
        q = q.filter_by(team=team)
    if position != "all":
        q = q.filter_by(position=position)

    player_ids = []
    for player in q.all():
        player_ids.append(player.player_id)
        print(player.name, player.team, player.position, player.current_price)
    return player_ids


def get_fixtures_for_player(player):
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
