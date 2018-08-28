"""
Useful commands to query the db
"""
import copy
from operator import itemgetter
from datetime import datetime
import dateparser
import pandas as pd

from .mappings import alternative_team_names, alternative_player_names
from .data_fetcher import DataFetcher
from .schema import Base, Player, Match, Fixture, \
    PlayerScore, PlayerPrediction, Transaction, engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import and_, or_


Base.metadata.bind = engine
DBSession = sessionmaker()
session = DBSession()

fetcher = DataFetcher() # in global scope so it can keep cached data


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
            total_value += fetcher.get_gameweek_data_for_player(pid,gameweek)[0]["value"]
        else:
            total_value += fetcher.get_player_summary_data()[pid]['now_cost']
    return total_value


def get_sell_price_for_player(player_id, gameweek=None):
    """
    find the price we bought the player for,
    and the price at the specified gameweek,
    if the price increased in that time, we only get half the profit.
    if gameweek is None, get price we could sell the player for now.
    """
    buy_price = 0
    transactions = session.query(Transaction)
    transactions = transactions.filter_by(player_id=player_id)
    transactions = transactions.order_by(Transaction.gameweek).all()
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
    pdata_bought = fetcher.get_gameweek_data_for_player(player_id, gw_bought)
    ## will be a list - can be more than one match in a gw - just use the 1st.
    price_bought = pdata_bought[0]["value"]

    if not gameweek:  # assume we want the current (i.e. next) gameweek
        price_now = fetcher.get_player_summary_data()[player_id]['now_cost']
    else:
        pdata_now = fetcher.get_gameweek_data_for_player(player_id, gw_bought)
        price_now = pdata_now[0]["value"]
    ## take off our half of the profit - boo!
    if price_now > price_bought:
        value = (price_now + price_bought) // 2   # round down
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
    if p:
        return p.player_id
    ## not found by name in DB - try alternative names
    for k,v in alternative_player_names.items():
        if player_name in v:
            p = session.query(Player).filter_by(name=k).first()
            if p:
                return p.player_id
            break
    ## still not found
    print("Unknown player_name {}".format(player_name))
    return None



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

def get_max_matches_per_player(position="all"):
    """
    can be used e.g. in bpl_interface.get_player_history_df
    to help avoid a ragged dataframe.
    """
    players = list_players()
    max_matches = 0
    for p in players:
        num_match = len(session.query(PlayerScore).filter_by(player_id=p).all())
        if num_match > max_matches:
            max_matches = num_match
    return max_matches


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


def get_predicted_points_for_player(player, method="EP", gameweeks_ahead=1):
    """
    Query the player prediction table for a given player.
    Return a dictionary keyed by gameweek.
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
    next_gw = get_next_gameweek()
    predicted_points = {}
    for gw in range(next_gw, next_gw+gameweeks_ahead):
        prediction = pps.filter_by(gameweek=gw).first()
        if not prediction:
            predicted_points[gw] = 0.
        else:
            predicted_points[gw] = prediction.predicted_points
    return predicted_points


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




def get_expected_minutes_for_player(player_id, num_match_to_use=3):
    """
    Look back num_match_to_use matches, and take an average
    of the number of minutes they played.
    But first, check the current data to see if they are injured.
    """
    pdata = fetcher.get_player_summary_data()[player_id]
    if pdata['chance_of_playing_next_round'] and \
       pdata['chance_of_playing_next_round'] < 0.75:
        return 0
    ## now get the playerscore rows from the db
    rows = session.query(PlayerScore)\
                  .filter_by(player_id=player_id).all()
    ## for speed, we use the fact that matches from this season
    ## are uploaded in order, so we can just take the last n
    ## rows, no need to look up dates and sort.
    total_mins = sum([r.minutes for r in rows[-num_match_to_use:]])
    average = total_mins // num_match_to_use
    return average


def generate_transfer_strategies(gw_ahead, transfers_last_gw=1):
    """
    Constraint: we want to take no more than a 4-point hit each week.
    So, for each gameweek, we can make 0, 1, or 2 changes, or, if we made 0
    the previous week, we can make 3.
    Generate all possible sequences, for N gameweeks ahead, and return along
    with the total points hit.
    """
    next_gw = get_next_gameweek()
    strategy_list = []
    possibilities = list(range(4)) if transfers_last_gw==0 else list(range(3))
    strategies = [ {next_gw:i} for i in possibilities]

    for gw in range(next_gw+1, next_gw+gw_ahead):
        new_strategies = []
        for s in strategies:
            possibilities = list(range(4)) if s[gw-1]==0 else list(range(3))
            ss=copy.deepcopy(s)
            new_strategies.append([ss.update({gw:p}) for p in possibilities])
            print("new_strategies",new_strategies)
        strategies = new_strategies
    return strategies
