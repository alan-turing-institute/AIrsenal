"""
Useful commands to query the db
"""
import copy
from operator import itemgetter
from datetime import datetime, timezone
import dateparser
import re

from .mappings import alternative_team_names, alternative_player_names

from .data_fetcher import FPLDataFetcher, MatchDataFetcher
from .schema import (
    Base,
    Player,
    PlayerAttributes,
    Result,
    Fixture,
    PlayerScore,
    PlayerPrediction,
    Transaction,
    FifaTeamRating,
    engine,
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy import and_, or_


Base.metadata.bind = engine
DBSession = sessionmaker()
##session = DBSession()

fetcher = FPLDataFetcher()  # in global scope so it can keep cached data


def get_current_players(gameweek=None,season="1819"):
    """
    Use the transactions table to find the team as of specified gameweek,
    then add up the values at that gameweek using the FPL API data.
    If gameweek is None, get team for next gameweek
    """
    current_players = []
    transactions = session.query(Transaction).filter_by(season=season)\
                                             .order_by(Transaction.gameweek)\
                                             .all()
    for t in transactions:
        if gameweek and t.gameweek > gameweek:
            break
        if t.bought_or_sold == 1:
            current_players.append(t.player_id)
        else:
            current_players.remove(t.player_id)
    assert len(current_players) == 15
    return current_players


def get_team_value(gameweek=None, season="1819"):
    """
    Use the transactions table to find the team as of specified gameweek,
    then add up the values at that gameweek using the FPL API data.
    If gameweek is None, get team for next gameweek
    """
    total_value = 0
    current_players = get_current_players(gameweek,season)
    for pid in current_players:
        if season=="1819":
            if gameweek:
                total_value += fetcher.get_gameweek_data_for_player(pid,
                                                                    gameweek)[0][
                                                                        "value"
                                                                    ]
            else:
                total_value += fetcher.get_player_summary_data()[pid]["now_cost"]
        else:
            player = session.query(Player).filter_by(season=season)\
                                          .filter_by(player_id=pid)\
                                          .first()
            total_value += player.cost
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


def get_next_gameweek(season="1819", dbsession=None):
    """
    Use the current time to figure out which gameweek we're in
    """
    if not dbsession:
        dbsession = session
    timenow = datetime.now(timezone.utc)
    fixtures = dbsession.query(Fixture)\
                        .filter_by(season=season)\
                        .all()
    earliest_future_gameweek = 39
    for fixture in fixtures:
        fixture_date = dateparser.parse(fixture.date)
        fixture_date = fixture_date.replace(tzinfo=timezone.utc)
        if (
            fixture_date > timenow
            and fixture.gameweek < earliest_future_gameweek
        ):
            earliest_future_gameweek = fixture.gameweek

    ## now make sure we aren't in the middle of a gameweek
    for fixture in fixtures:
        if (
            dateparser.parse(fixture.date)\
                .replace(tzinfo=timezone.utc) < timenow
            and fixture.gameweek == earliest_future_gameweek
        ):

            earliest_future_gameweek += 1

    return earliest_future_gameweek


def get_gameweek_by_date(date, dbsession=None):
    """
    Use the dates of the fixtures to find the gameweek.
    """
    # convert date to a datetime object if it isn't already one.
    if not dbsession:
        dbsession=session
    if not isinstance(date, datetime):
        date = dateparser.parse(date)
    fixtures = dbsession.query(Fixture).all()
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


def get_player(player_name_or_id, dbsession=None):
    """
    query the player table by name or id, return the player object (or None)
    """
    if not dbsession:
        dbsession = session # use the one defined in this module
    if isinstance(player_name_or_id, int):
        filter_attr = Player.player_id
    else:
        filter_attr = Player.name
    p = dbsession.query(Player)\
                 .filter(filter_attr==player_name_or_id).first()
    if p:
        return p
    if isinstance(player_name_or_id, int): # didn't find by id - return None
        return None
    # assume we have a name, now try alternative names
    for k, v in alternative_player_names.items():
        if player_name_or_id in v:
            p = dbsession.query(Player)\
                         .filter_by(name=k).first()
            if p:
                return p
    # didn't find it - return None
    return None


def get_player_name(player_id, dbsession=None):
    """
    lookup player name, for human readability
    """
    if not dbsession:
        dbsession = session
    p = dbsession.query(Player).filter_by(player_id=player_id).first()
    if not p:
        print("Unknown player_id {}".format(player_id))
        return None
    return p.name


def get_player_id(player_name, dbsession=None):
    """
    lookup player id, for machine readability
    """
    if not dbsession:
        dbsession = session
    p = dbsession.query(Player).filter_by(name=player_name).first()
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


def list_players(position="all", team="all",
                 order_by="current_price",
                 season="1819",
                 dbsession=None,
                 verbose=False):
    """
    print list of players, and
    return a list of player_ids
    """
    if not dbsession:
        dbsession = session
    q = dbsession.query(PlayerAttributes)\
                 .filter_by(season=season)
    if team != "all":
        q = q.filter_by(team=team)
    if position != "all":
        q = q.filter_by(position=position)
    if order_by == "current_price":
        q = q.order_by(PlayerAttributes.current_price.desc())
    players = []
    for p in q.all():
        players.append(p.player)
        if verbose:
            print(p.player.name, p.team, p.position, p.current_price)
    return players


def get_max_matches_per_player(position="all",season="1819",dbsession=None):
    """
    can be used e.g. in bpl_interface.get_player_history_df
    to help avoid a ragged dataframe.
    """
    players = list_players(position=position,season=season, dbsession=dbsession)
    max_matches = 0
    for p in players:
        num_match = len(p.scores)
        if num_match > max_matches:
            max_matches = num_match
    return max_matches


def get_fixtures_for_player(player, season="1819", gw_range=None, dbsession=None,
                            verbose=False):
    """
    search for upcoming fixtures for a player, specified either by id or name.
    If gw_range not specified:
       for current season: return fixtures from now to end of season
       for past seasons: return all fixtures in the season
    """
    if not dbsession:
        dbsession=session
    player_query = dbsession.query(Player)
    if isinstance(player, str): # given a player name
        player_record = player_query.filter_by(name=player).first()
    elif isinstance(player, int): # given a player id
        player_record = player_query.filter_by(player_id=player).first()
    else: # given a player object
        player_record = player
    if not player_record:
        print("Couldn't find {} in database".format(player))
        return []
    team = player_record.team(season)
    tag = get_latest_fixture_tag(season,dbsession)
    fixtures = (
        dbsession.query(Fixture).filter_by(season=season)
        .filter_by(tag=tag)
        .filter(or_(Fixture.home_team == team, Fixture.away_team == team))
        .order_by(Fixture.gameweek)
        .all()
    )
    fixture_ids = []
    next_gameweek = get_next_gameweek(season,dbsession)
    for fixture in fixtures:
        if gw_range:
            if fixture.gameweek in gw_range:
                fixture_ids.append(fixture.fixture_id)
        else:
            if season == "1819" and fixture.gameweek < next_gameweek:
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
        session.query(Fixture)
        .filter_by(home_team=home_team)
        .filter_by(away_team=away_team)
        .order_by(Fixture.season)
        .all()
    )
    fixture_ids = [(f.fixture_id, f.season) for f in previous_matches]
    previous_points = {}
    for fid in fixture_ids:
        scores = (
            session.query(PlayerScore)
            .filter_by(player_id=player_id, fixture_id=m[0])
            .all()
        )
        for s in scores:
            previous_points[m[1]] = s.points

    return previous_points


def get_predicted_points_for_player(player, method, season="1819", dbsession=None):
    """
    Query the player prediction table for a given player.
    Return a dict, keyed by gameweek.
    """
    if not dbsession:
        dbsession=session
    if isinstance(player, str) or isinstance(player,int):
        # we want the actual player object
        player= get_player(player,season=season,dbsession=dbsession)
    pps = (
        dbsession.query(PlayerPrediction)
        .filter_by(season=season)
        .filter_by(player_id=player_id, method=method)
        .all()
    )
    ppdict = {}
    for prediction in pps:
        ppdict[prediction.gameweek] = prediction.predicted_points
    return ppdict


def get_predicted_points(gameweek, method, position="all", team="all",
                         season="1819", dbsession=None):
    """
    Query the player_prediction table with selections, return
    list of tuples (player_id, predicted_points) ordered by predicted_points
    "gameweek" argument can either be a single integer for one gameweek, or a
    list of gameweeks, in which case we will get the sum over all of them
    """
    player_ids = list_players(position, team, season=season, dbsession=dbsession)

    if isinstance(gameweek, int):
        output_list = [
            (p, get_predicted_points_for_player(p, method)[gameweek])
            for p in player_ids
        ]
    else:
        output_list = [
            (p, sum(get_predicted_points_for_player(p, method)[gw] for gw in gameweek))
            for p in player_ids
        ]

    output_list.sort(key=itemgetter(1), reverse=True)
    return output_list


def get_return_gameweek_for_player(player_id, dbsession=None):
    """
    If  a player is injured and there is 'news' about them on FPL,
    parse this string to get expected return date.
    """
    pdata = fetcher.get_player_summary_data()[player_id]
    rd_rex = '(Expected back|Suspended until)[\\s]+([\\d]+[\\s][\\w]{3})'
    if 'news' in pdata.keys() and re.search(rd_rex, pdata['news']):
        return_str = re.search(rd_rex, pdata['news']).groups()[1]+" 2018"
        return_date = dateparser.parse(return_str)
        return_gameweek = get_gameweek_by_date(return_date,dbsession=dbsession)
        return return_gameweek
    return None


def get_recent_minutes_for_player(player,
                                  num_match_to_use=3,
                                  season="1819",
                                  last_gw=None,
                                  dbsession=None):

    """
    Look back num_match_to_use matches, and return an array
    containing minutes played in each.
    If current_gw is not given, we take it to be the most
    recent finished gameweek.
    """
    if not dbsession:
        dbsession=session
    if not last_gw:
        last_gw = get_next_gameweek(season, dbsession=dbsession)
    first_gw = last_gw - num_match_to_use
    ## get the playerscore rows from the db
    rows = dbsession.query(PlayerScore)\
                  .filter(PlayerScore.fixture.has(season=season))\
                  .filter_by(player_id=player.player_id)\
                  .filter(PlayerScore.fixture.has(Fixture.gameweek >= first_gw))\
                  .filter(PlayerScore.fixture.has(Fixture.gameweek < last_gw))\
                  .all()
    ## for speed, we use the fact that matches from this season
    ## are uploaded in order, so we can just take the last n
    ## rows, no need to look up dates and sort.
    return [r.minutes for r in rows[-num_match_to_use:]]


def get_last_gameweek_in_db(season="1819", dbsession=None):
    """
    query the result table to see what was the last gameweek for which
    we have filled the data.
    """
    if not dbsession:
        dbsession=session
    last_result = (
        dbsession.query(Fixture).filter_by(season=season)\
        .filter(Fixture.result != None)\
        .order_by(Fixture.gameweek).all()[-1]
    )
    return last_result.gameweek


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


def get_latest_prediction_tag(season="1819",dbsession=None):
    """
    query the predicted_score table and get the method
    field for the last row.
    """
    if not dbsession:
        dbsession=session
    rows = dbsession.query(PlayerPrediction)\
                  .filter_by(season=season).all()
    return rows[-1].tag


def get_latest_fixture_tag(season="1819",dbsession=None):
    """
    query the predicted_score table and get the method
    field for the last row.
    """
    if not dbsession:
        dbsession=session
    rows = dbsession.query(Fixture)\
                  .filter_by(season=season).all()
    return rows[-1].tag
