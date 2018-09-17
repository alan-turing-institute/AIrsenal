"""
Useful commands to query the db
"""
import copy
from operator import itemgetter
from datetime import datetime
import dateparser

from .mappings import alternative_team_names, alternative_player_names

from .data_fetcher import FPLDataFetcher, MatchDataFetcher
from .schema import (
    Base,
    Player,
    Match,
    Fixture,
    PlayerScore,
    PlayerPrediction,
    Transaction,
    engine,
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy import and_, or_


Base.metadata.bind = engine
DBSession = sessionmaker()
session = DBSession()

fetcher = FPLDataFetcher()  # in global scope so it can keep cached data


def get_current_players(gameweek=None):
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
    assert len(current_players) == 15
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
            total_value += fetcher.get_gameweek_data_for_player(pid, gameweek)[0][
                "value"
            ]
        else:
            total_value += fetcher.get_player_summary_data()[pid]["now_cost"]
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
        print(
            "Player {} is was not in the team at gameweek {}".format(
                player_id, gameweek
            )
        )
    pdata_bought = fetcher.get_gameweek_data_for_player(player_id, gw_bought)
    ## will be a list - can be more than one match in a gw - just use the 1st.
    price_bought = pdata_bought[0]["value"]

    if not gameweek:  # assume we want the current (i.e. next) gameweek
        price_now = fetcher.get_player_summary_data()[player_id]["now_cost"]
    else:
        pdata_now = fetcher.get_gameweek_data_for_player(player_id, gw_bought)
        price_now = pdata_now[0]["value"]
    ## take off our half of the profit - boo!
    if price_now > price_bought:
        value = (price_now + price_bought) // 2  # round down
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
        if (
            dateparser.parse(fixture.date) > timenow
            and fixture.gameweek < earliest_future_gameweek
        ):
            earliest_future_gameweek = fixture.gameweek
    ## now make sure we aren't in the middle of a gameweek
    for fixture in fixtures:
        if (
            dateparser.parse(fixture.date) < timenow
            and fixture.gameweek == earliest_future_gameweek
        ):
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


def get_overall_points(gameweek=None):
    """
    Get our total points
    """
    data = fetcher.get_fpl_team_data()
    if not gameweek:
        return data['entry']['summary_overall_points']
    elif isinstance(gameweek,int) and \
         gameweek <= len(data['history']):
        return data['history'][gameweek-1]['points']
    else:
        print("Unknown gameweek")
        return 0


def get_overall_ranking(gameweek=None):
    """
    Get our total points
    """
    data = fetcher.get_fpl_team_data()
    if not gameweek:
        return data['entry']['summary_overall_rank']
    elif isinstance(gameweek,int) and \
         gameweek <= len(data['history']):
        return data['history'][gameweek-1]['rank']
    else:
        print("Unknown gameweek")
        return 0


def get_league_standings():
    """
    Get stuff about our mini-league
    """
    data = fetcher.get_fpl_league_data()
    team_name = data['league']['name']
    standings = []
    for s in data['standings']['results']:
        standings.append({'name': s['entry_name'],
                          'manager': s['player_name'],
                          'points': s['total']
                          })
    return team_name, standings


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
    for k, v in alternative_player_names.items():
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
    if isinstance(player, str):
        player_record = player_query.filter_by(name=player).first()
    else:
        player_record = player_query.filter_by(player_id=player).first()
    if not player_record:
        print("Couldn't find {} in database".format(player))
        return []
    team = player_record.team
    fixtures = (
        session.query(Fixture)
        .filter(or_(Fixture.home_team == team, Fixture.away_team == team))
        .order_by(Fixture.gameweek)
        .all()
    )
    fixture_ids = []
    next_gameweek = get_next_gameweek()
    for fixture in fixtures:
        if fixture.gameweek < next_gameweek:
            continue
        if verbose:
            print(
                "{} vs {} gameweek {}".format(
                    fixture.home_team, fixture.away_team, fixture.gameweek
                )
            )
        fixture_ids.append(fixture.fixture_id)
    return fixture_ids


def get_previous_points_for_same_fixture(player, fixture_id):
    """
    Search the past matches for same fixture in past seasons,
    and how many points the player got.
    """
    if isinstance(player, str):
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

    previous_matches = (
        session.query(Match)
        .filter_by(home_team=home_team)
        .filter_by(away_team=away_team)
        .order_by(Match.season)
        .all()
    )
    match_ids = [(match.match_id, match.season) for match in previous_matches]
    previous_points = {}
    for m in match_ids:
        scores = (
            session.query(PlayerScore)
            .filter_by(player_id=player_id, match_id=m[0])
            .all()
        )
        for s in scores:
            previous_points[m[1]] = s.points

    return previous_points


def get_predicted_points_for_player(player, method):
    """
    Query the player prediction table for a given player.
    Return a dict, keyed by gameweek.
    """
    if isinstance(player, str):
        player_id = get_player_id(player)
    else:
        player_id = player
    pps = (
        session.query(PlayerPrediction)
        .filter_by(player_id=player_id, method=method)
        .all()
    )
    ppdict = {}
    for prediction in pps:
        ppdict[prediction.gameweek] = prediction.predicted_points
    return ppdict


def get_predicted_points(gameweek, method, position="all", team="all"):
    """
    Query the player_prediction table with selections, return
    list of tuples (player_id, predicted_points) ordered by predicted_points
    "gameweek" argument can either be a single integer for one gameweek, or a
    list of gameweeks, in which case we will get the sum over all of them
    """
    player_ids = list_players(position, team)

    if isinstance(gameweek, int):
        output_list = [
            (p, get_predicted_points_for_player(p,method)[gameweek]) for p in player_ids
        ]
    else:
        output_list = [
            (p, sum(get_predicted_points_for_player(p,method)[gw] for gw in gameweek))
            for p in player_ids
        ]

    output_list.sort(key=itemgetter(1), reverse=True)
    return output_list


def get_recent_minutes_for_player(player_id, num_match_to_use=3):

    """
    Look back num_match_to_use matches, and return an array
    containing minutes played in each.
    But first, check the current data to see if they are injured.
    """
    pdata = fetcher.get_player_summary_data()[player_id]
    if (
        pdata["chance_of_playing_next_round"]
        and pdata["chance_of_playing_next_round"] < 0.75
    ):
        return 0
    ## now get the playerscore rows from the db
    rows = session.query(PlayerScore).filter_by(player_id=player_id).all()
    ## for speed, we use the fact that matches from this season
    ## are uploaded in order, so we can just take the last n
    ## rows, no need to look up dates and sort.
    return [r.minutes for r in rows[-num_match_to_use:]]


def get_last_gameweek_in_db():
    """
    query the match table to see what was the last gameweek for which
    we have filled the data.
    """
    last_match = session.query(Match).filter_by(season="1819")\
                                     .order_by(Match.gameweek).all()[-1]
    return last_match.gameweek


def get_last_finished_gameweek():
    """
    query the API to see what the last gameweek marked as 'finished' is.
    """
    event_data = fetcher.get_event_data()
    last_finished = 0
    for gw in sorted(event_data.keys()):
        if event_data[gw]["is_finished"]:
            last_finished = gw
        else:
            return last_finished
    return last_finished


def get_latest_prediction_tag():
    """
    query the predicted_score table and get the method
    field for the last row.
    """
    rows = session.query(PlayerPrediction).all()
    return rows[-1].method
