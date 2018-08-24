"""
Useful commands to query the db
"""

from operator import itemgetter
from datetime import datetime
import dateparser

from .mappings import alternative_team_names
from .data_fetcher import DataFetcher
from .schema import Base, Player, Match, Fixture, \
    PlayerScore, PlayerPrediction, Transaction, engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import and_, or_



Base.metadata.bind = engine
DBSession = sessionmaker()
session = DBSession()

df = DataFetcher() # in global scope so it can keep cached data


def get_current_team(gameweek=None):
    """
    Use the transactions table to find the team as of specified gameweek,
    then add up the values at that gameweek using the FPL API data.
    If gameweek is None, get team for next gameweek
    """
    current_players = []
    transactions = session.query(Transaction).order_by(Transaction.gameweek).all()
    for t in transactions:
        if gameweek and t.gameweek > gameweek:
            break
        if t.bought_or_sold == 1:
            current_players.append(t.player_id)
        else:
            current_players.remove(t.player_id)
    assert(len(current_players)==15)
    return current_players

def get_team_value(gameweek=None):
    """
    Use the transactions table to find the team as of specified gameweek,
    then add up the values at that gameweek using the FPL API data.
    If gameweek is None, get team for next gameweek
    """
    total_value = 0
    current_players = get_current_team(gameweek)
    for pid in current_players:
        if gameweek:
            total_value += df.get_gameweek_data_for_player(pid,gameweek)[0]["value"]
        else:
            total_value += df.get_player_summary_data()[pid]['now_cost']
    return total_value


def get_sell_price_for_player(player_id, gameweek=None):
    """
    find the price we bought the player for, and the price at the specified gameweek,
    if the price increased in that time, we only get half the profit.
    if gameweek is None, get price we could sell the player for now.
    """
    buy_price = 0
    transactions = session.query(Transaction).filter_by(player_id=player_id)\
                                             .order_by(Transaction.gameweek).all()
    for t in transactions:
        if gameweek and t.gameweek > gameweek:
            break
        if t.bought_or_sold == 1:
            gw_bought = t.gameweek
        else:
            gw_bought = None
    if not gw_bought:
        print("Player {} is was not in the team at gameweek {}"\
              .format(player_id,gameweek))

    price_bought = df.get_gameweek_data_for_player(player_id, gw_bought)[0]["value"]
    if not gameweek:  ### assume we want the current (i.e. next) gameweek
        price_now = df.get_player_summary_data()[player_id]['now_cost']
    else:
        price_now = df.get_gameweek_data_for_player(player_id, gameweek)[0]["value"]
    if price_now > price_bought:
        value = (price_now + price_bought) // 2   # integer arithmetic, as we round down
    else:
        value = price_now
    return value


def get_next_gameweek():
    """
    Use the current time to figure out which gameweek we're in
    """
    timenow = datetime.now()
    fixtures = session.query(Fixture).all()
    earliest_future_gameweek = 38
    for fixture in fixtures:
        if dateparser.parse(fixture.date) > timenow and \
           fixture.gameweek < earliest_future_gameweek:
            earliest_future_gameweek = fixture.gameweek
    ## now make sure we aren't in the middle of a gameweek
    for fixture in fixtures:
        if dateparser.parse(fixture.date) < timenow and \
           fixture.gameweek == earliest_future_gameweek:
            earliest_future_gameweek += 1

    return earliest_future_gameweek


def get_gameweek_by_date(date):
    """
    Use the dates of the fixtures to find the gameweek.
    """
    # convert date to a datetime object if it isn't already one.
    if not isinstance(date, datetime):
        date = dateparser.parse(date)
    fixtures = session.query(Fixture).all()
    for fixture in fixtures:
        fixture_date = dateparser.parse(fixture.date)
        if fixture_date.date() == date.date():
            return fixture.gameweek
    return None


def get_team_name(team_id):
    """
    return 3-letter team name given a numerical id
    """
    for k, v in alternative_team_names.items():
        for vv in v:
            if str(team_id) == vv:
                return k
    return None

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
    next_gameweek = get_next_gameweek()
    for fixture in fixtures:
        if fixture.gameweek < next_gameweek:
            continue
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


def get_player_history_table(position="all"):
    """
    Query the player_score table.
    """
    output_file = open("player_history_{}.csv".format(position),"w")
    output_file.write("player_id,player_name,match_id,goals,assists,minutes,team_goals\n")
    player_ids = list_players(position)
    for pid in player_ids:
        player_name = get_player_name(pid)
        results = session.query(PlayerScore).filter_by(player_id=pid).all()
        row_count = 0
        for row in results:
            minutes = row.minutes
#            if minutes == 0:
#                continue
            opponent = row.opponent
            match_id = row.match_id
            goals = row.goals
            assists = row.assists
            # find the match, in order to get team goals
            match = session.query(Match).filter_by(match_id = row.match_id).first()
            if match.home_team == row.opponent:
                team_goals = match.away_score
            elif match.away_team == row.opponent:
                team_goals = match.home_score
            else:
                print("Unknown opponent!")
                team_goals = -1
            output_file.write("{},{},{},{},{},{},{}\n".format(pid,
                                                             player_name,
                                                             match_id,
                                                             goals,
                                                             assists,
                                                             minutes,
                                                             team_goals))
            row_count += 1
        if row_count < 38*3:
            for i in range(row_count,38*3):
                output_file.write("{},{},0,0,0,0,0\n".format(pid,player_name))
    output_file.close()
