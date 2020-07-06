"""
Useful commands to query the db
"""
import copy
from operator import itemgetter
from datetime import datetime, timezone
import pandas as pd
import dateparser
import re

from .mappings import alternative_team_names, alternative_player_names

from .data_fetcher import FPLDataFetcher
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
    Team,
    engine,
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy import and_, or_


Base.metadata.bind = engine
DBSession = sessionmaker()
session = DBSession()

fetcher = FPLDataFetcher()  # in global scope so it can keep cached data


def get_current_season():
    """
    use the current time to find what season we're in.
    """
    current_time = datetime.now()
    if current_time.month > 6:
        start_year = current_time.year
    else:
        start_year = current_time.year - 1
    end_year = start_year + 1
    return "{}{}".format(str(start_year)[2:],str(end_year)[2:])

# make this a global variable in this module, import into other modules
CURRENT_SEASON = get_current_season()

# TODO make this a database table so we can look at past seasons
CURRENT_TEAMS = [t["short_name"] for t in fetcher.get_current_team_data().values()]


from .bpl_interface import get_fitted_team_model


def get_previous_season(season):
    """
    Convert string e.g. '1819' into one for previous season, i.e. '1718'
    """
    start_year = int(season[:2])
    end_year = int(season[2:])
    prev_start_year = start_year - 1
    prev_end_year = end_year - 1
    prev_season = "{}{}".format(prev_start_year, prev_end_year)
    return prev_season


def get_past_seasons(num_seasons):
    """
    Go back num_seasons from the current one
    """
    season = CURRENT_SEASON
    seasons = []
    for i in range(num_seasons):
        season = get_previous_season(season)
        seasons.append(season)
    return seasons


def get_current_players(gameweek=None,season=None, dbsession=None):
    """
    Use the transactions table to find the team as of specified gameweek,
    then add up the values at that gameweek using the FPL API data.
    If gameweek is None, get team for next gameweek
    """
    if not season:
        season = CURRENT_SEASON
    if not dbsession:
        dbsession=session
    current_players = []
    transactions = dbsession.query(Transaction).filter_by(season=season)\
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


def get_team_value(team=None, gameweek=None, season=None):
    """
    Use the transactions table to find the team as of specified gameweek,
    then add up the values at that gameweek using the FPL API data.
    If gameweek is None, get team for next gameweek
    """
    if not season:
        season = CURRENT_SEASON
    total_value = 0
    if not team:
        players = get_current_players(gameweek,season)
    else:
        players = [p.player_id for p in team.players]
    for pid in players:
        if season==CURRENT_SEASON:
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
    
    gw_bought = None
    for t in transactions:
        if gameweek and t.gameweek > gameweek:
            break
        if t.bought_or_sold == 1:
            gw_bought = t.gameweek

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


def get_next_gameweek(season=CURRENT_SEASON, dbsession=None):
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
        try:
            fixture_date = dateparser.parse(fixture.date)
            fixture_date = fixture_date.replace(tzinfo=timezone.utc)
            if (
                    fixture_date > timenow
                    and fixture.gameweek < earliest_future_gameweek
            ):
                earliest_future_gameweek = fixture.gameweek
        except(TypeError): ## date could be null if fixture not scheduled
            continue
    ## now make sure we aren't in the middle of a gameweek
    for fixture in fixtures:
        try:
            if (
                    dateparser.parse(fixture.date)\
                    .replace(tzinfo=timezone.utc) < timenow
                    and fixture.gameweek == earliest_future_gameweek
            ):

                earliest_future_gameweek += 1
        except(TypeError):
            continue
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
        try:
            fixture_date = dateparser.parse(fixture.date)
            if fixture_date.date() == date.date():
                return fixture.gameweek
        except(TypeError):  # NULL date if fixture not scheduled
            continue
    return None


def get_team_name(team_id, season=CURRENT_SEASON, dbsession=None):
    """
    return 3-letter team name given a numerical id.
    These ids are based on alphabetical order of all teams in that season,
    so can vary from season to season.
    """
    if not dbsession:
        dbsession = session
    team = dbsession.query(Team).filter_by(season=season, team_id=team_id).first()
    if team:
        return team.name
    else:
        print("Unknown team_id {} for {} season".format(team_id, season))
        return None


def get_teams_for_season(season, dbsession=None):
    """
    Query the Team table and get a list of teams for a given
    season.
    """
    if not dbsession:
        dbsession = session
    teams = dbsession.query(Team).filter_by(season=season).all()
    return [t.name for t in teams]


def get_player(player_name_or_id, dbsession=None):
    """
    query the player table by name or id, return the player object (or None)
    """
    if not dbsession:
        dbsession = session # use the one defined in this module

    # if an id has been passed as a string, convert it to an integer
    if isinstance(player_name_or_id, str) and player_name_or_id.isdigit():
        player_name_or_id = int(player_name_or_id)

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


def list_teams(season=CURRENT_SEASON,
               dbsession=None):
    """
    Print all teams from current season.
    """
    if not dbsession:
        dbsession = session
    rows = dbsession.query(Team)\
                 .filter_by(season=season) .all()
    return [{"name": row.name, "full_name": row.full_name} \
            for row in rows]


def list_players(position="all", team="all",
                 order_by="current_price",
                 season=CURRENT_SEASON,
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


def get_max_matches_per_player(position="all",season=CURRENT_SEASON,dbsession=None):
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


def get_fixtures_for_player(player, season=CURRENT_SEASON, gw_range=None, dbsession=None,
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
    fixture_rows = (
        dbsession.query(Fixture).filter_by(season=season)
        .filter_by(tag=tag)
        .filter(or_(Fixture.home_team == team, Fixture.away_team == team))
        .order_by(Fixture.gameweek)
        .all()
    )
    fixtures = []
    next_gameweek = get_next_gameweek(season,dbsession)
    for fixture in fixture_rows:
        if not fixture.gameweek: # fixture not scheduled yet
            continue
        if gw_range:
            if fixture.gameweek in gw_range:
                fixtures.append(fixture)
        else:
            if season == CURRENT_SEASON and fixture.gameweek < next_gameweek:
                continue
            if verbose:
                print(
                    "{} vs {} gameweek {}".format(
                        fixture.home_team, fixture.away_team, fixture.gameweek
                    )
                )
            fixtures.append(fixture)
    return fixtures


def get_next_fixture_for_player(player, season=CURRENT_SEASON, dbsession=None):
    """
    Get a players next fixture as a string, for easy displaying
    """
    if not dbsession:
        dbsession=session
    # given a player name or id, convert to player object
    if isinstance(player, str) or isinstance(player, int):
        player = get_player(player)
    team = player.team(season)
    fixtures_for_player = get_fixtures_for_player(player,
                                                  season,
                                                  [get_next_gameweek()],
                                                  dbsession)
    output_string = ""
    for fixture in fixtures_for_player:
        is_home = False
        if fixture.home_team == team:
            is_home = True
            output_string += fixture.away_team + " (h)"
        else:
            output_string += fixture.home_team + " (a)"
        output_string += ", "
    return output_string[:-2]


def get_fixtures_for_season(season=CURRENT_SEASON, dbsession=session):
    """Return all fixtures for a season."""
    fixtures = dbsession.query(Fixture)\
                        .filter_by(season=season)\
                        .all()
    return fixtures


def get_result_for_fixture(fixture, dbsession=session):
    """Get result for a fixture."""
    result = session.query(Result)\
                    .filter_by(fixture=fixture)\
                    .all()
    return result


def get_player_scores_for_fixture(fixture, dbsession=session):
    """Get player scores for a fixture."""
    player_scores = session.query(PlayerScore)\
                           .filter_by(fixture=fixture)\
                           .all()
    return player_scores


def get_players_for_gameweek(gameweek):
    """
    Use FPL API to get the players for a given gameweek.
    """
    player_data = fetcher.get_fpl_team_data(gameweek)
    player_list = [p['element'] for p in player_data]
    return player_list


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


def get_predicted_points_for_player(player, tag, season=CURRENT_SEASON, dbsession=None):
    """
    Query the player prediction table for a given player.
    Return a dict, keyed by gameweek.
    """
    if not dbsession:
        dbsession=session
    if isinstance(player, str) or isinstance(player,int):
        # we want the actual player object
        player= get_player(player,dbsession=dbsession)
    pps = (
        dbsession.query(PlayerPrediction)
        .filter(PlayerPrediction.fixture.has(
            Fixture.season==season) )\
        .filter_by(player_id=player.player_id, tag=tag)
        .all()
    )
    ppdict = {}
    for prediction in pps:
        ## there is one prediction per fixture.
        ## for double gameweeks, we need to add the two together
        gameweek = prediction.fixture.gameweek
        if not gameweek in ppdict.keys():
            ppdict[gameweek] = 0
        ppdict[gameweek] += prediction.predicted_points
    ## we still need to fill in zero for gameweeks that they're not playing.
    for gw in range(1,39):
        if not gw in ppdict.keys():
            ppdict[gw] = 0.
    return ppdict


def get_predicted_points(gameweek, tag, position="all", team="all",
                         season=CURRENT_SEASON, dbsession=None):
    """
    Query the player_prediction table with selections, return
    list of tuples (player_id, predicted_points) ordered by predicted_points
    "gameweek" argument can either be a single integer for one gameweek, or a
    list of gameweeks, in which case we will get the sum over all of them
    """
    players = list_players(position, team, season=season, dbsession=dbsession)

    if isinstance(gameweek, int):  # predictions for a single gameweek
        output_list = [
            (p, get_predicted_points_for_player(p, tag=tag,
                                                season=season,
                                                dbsession=dbsession)[gameweek])
            for p in players
        ]
    else:  # predictions for a list of gameweeks
        output_list = [
            (p, sum(get_predicted_points_for_player(p, tag=tag,
                                                    season=season,
                                                    dbsession=dbsession)[gw]
                    for gw in gameweek))
            for p in players
        ]

    output_list.sort(key=itemgetter(1), reverse=True)
    return output_list


def get_top_predicted_points(gameweek=None, tag=None,
                             position="all", team="all",
                             n_players=10, per_position=False,
                             season=CURRENT_SEASON, dbsession=None):
    """Print players with the top predicted points.

    
    Keyword Arguments:
        gameweek {int or list} -- Single gameweek or list of gameweeks in which
        case returned totals are sums across all gameweeks (default: next
        gameweek).
        tag {str} -- Prediction tag to query (default: latest prediction tag)
        position {str} -- Player position to query (default: {"all"})
        per_position {boolean} -- If True print top n_players players for
        each position separately (default: {False})
        team {str} -- Team to query (default: {"all"})
        n_players {int} -- Number of players to return (default: {10})
        season {str} -- Season to query (default: {CURRENT_SEASON})
        dbsession {SQLAlchemy session} -- Database session (default: {None})
    """
    if not tag:
        tag = get_latest_prediction_tag()
    if not gameweek:
        gameweek = get_next_gameweek()
    
    print("="*50)
    print("PREDICTED TOP {} PLAYERS FOR GAMEWEEK(S) {}:".format(n_players,
                                                                gameweek))
    print("="*50)
    
    if not per_position:
        pts = get_predicted_points(gameweek, tag, position=position, team=team,
                                season=season, dbsession=dbsession)
        pts = sorted(pts, key=lambda x: x[1], reverse=True)
    
        for i, p in enumerate(pts[:n_players]):
            print("{}. {}, {:.2f}pts".format(i+1, p[0].name, p[1]))
            
    else:
        for position in ["GK", "DEF", "MID", "FWD"]:
            pts = get_predicted_points(gameweek, tag, position=position,
                                       team=team, season=season,
                                       dbsession=dbsession)
            pts = sorted(pts, key=lambda x: x[1], reverse=True)
            print("{}:".format(position))
            for i, p in enumerate(pts[:n_players]):
                print("{}. {}, {:.2f}pts".format(i+1, p[0].name, p[1]))
            print("-"*25)


def get_return_gameweek_for_player(player_id, dbsession=None):
    """
    If  a player is injured and there is 'news' about them on FPL,
    parse this string to get expected return date.
    """
    pdata = fetcher.get_player_summary_data()[player_id]
    rd_rex = '(Expected back|Suspended until)[\\s]+([\\d]+[\\s][\\w]{3})'
    if 'news' in pdata.keys() and re.search(rd_rex, pdata['news']):
        
        return_str = re.search(rd_rex, pdata['news']).groups()[1]
        # return_str should be a day and month string (without year)
    
        # create a date in the future from the day and month string
        return_date = dateparser.parse(return_str,
                                       settings={"PREFER_DATES_FROM": "future"}) 

        if not return_date:
            raise ValueError("Failed to parse date from string '{}'"
                             .format(return_date))

        return_gameweek = get_gameweek_by_date(return_date,dbsession=dbsession)
        return return_gameweek
    return None


def calc_average_minutes(player_scores):
    """
    Simple average of minutes played for a list of PlayerScore objects.
    """
    total = 0.
    for ps in player_scores:
        total += ps.minutes
    return total / len(player_scores)


def estimate_minutes_from_prev_season(player, season=CURRENT_SEASON, dbsession=None):
    """
    take average of minutes from previous season if any, or else return [60]
    """
    if not dbsession:
        dbsession=session
    previous_season = get_previous_season(season)
    player_scores = dbsession.query(PlayerScore)\
                    .filter_by(player_id=player.player_id)\
                    .filter(PlayerScore.fixture.has(season=previous_season))\
                    .all()
    if len(player_scores) == 0:
        # Crude scaling based on player price vs teammates in his position
        teammates = list_players(position=player.position(season),
                                 team=player.team(season),
                                 season=season,
                                 dbsession=dbsession)

        team_prices = [pl.current_price(season) for pl in teammates]
        player_price = player.current_price(season)
        ratio = player_price/max(team_prices)

        return [60*(ratio**2)]
    else:
        average_mins = calc_average_minutes(player_scores)
        return [average_mins]


def get_recent_playerscore_rows(player,
                                num_match_to_use=3,
                                season=CURRENT_SEASON,
                                last_gw=None,
                                dbsession=None):
    """
    Query the playerscore table in the database to retrieve
    the last num_match_to_use rows for this player.
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
                  .filter(PlayerScore.fixture.has(Fixture.gameweek > first_gw))\
                  .filter(PlayerScore.fixture.has(Fixture.gameweek <= last_gw))\
                  .all()
    ## for speed, we use the fact that matches from this season
    ## are uploaded in order, so we can just take the last n
    ## rows, no need to look up dates and sort.
    return rows[-num_match_to_use:]


def get_recent_scores_for_player(player,
                                 num_match_to_use=3,
                                 season=CURRENT_SEASON,
                                 last_gw=None,
                                 dbsession=None):
    """
    Look num_match_to_use matches back, and return the
    FPL points for this player for each of these matches.
    Return a dict {gameweek: score, }
    """
    if not last_gw:
        last_gw = get_next_gameweek(season, dbsession=dbsession)
    first_gw = last_gw - num_match_to_use
    playerscores = get_recent_playerscore_rows(player,
                                               num_match_to_use,
                                               season,
                                               last_gw,
                                               dbsession)

    points = {}
    for i , ps in enumerate(playerscores):
        points[range(first_gw, last_gw)[i]] = ps.points
    return points


def get_recent_minutes_for_player(player,
                                  num_match_to_use=3,
                                  season=CURRENT_SEASON,
                                  last_gw=None,
                                  dbsession=None):

    """
    Look back num_match_to_use matches, and return an array
    containing minutes played in each.
    If current_gw is not given, we take it to be the most
    recent finished gameweek.
    """
    playerscores = get_recent_playerscore_rows(player,
                                               num_match_to_use,
                                               season,
                                               last_gw,
                                               dbsession)
    minutes = [r.minutes for r in playerscores]

    # if going back num_matches_to_use from last_gw takes us before the start
    # of the season, also include a minutes estimate using last season's data
    if not last_gw:
        last_gw = get_next_gameweek(season, dbsession=dbsession)
    first_gw = last_gw - num_match_to_use
    if first_gw < 0 or len(minutes) == 0:
        minutes = (minutes +
                   estimate_minutes_from_prev_season(player, season, dbsession))

    return minutes


def get_last_gameweek_in_db(season=CURRENT_SEASON, dbsession=None):
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


def get_latest_prediction_tag(season=CURRENT_SEASON,dbsession=None):
    """
    query the predicted_score table and get the method
    field for the last row.
    """
    if not dbsession:
        dbsession=session
    rows = dbsession.query(PlayerPrediction)\
                  .filter(PlayerPrediction.fixture.has(
                      Fixture.season==season
                  )).all()
    try:
        return rows[-1].tag
    except(IndexError):
        raise RuntimeError("No predicted points in database - has the database been filled?")


def get_latest_fixture_tag(season=CURRENT_SEASON,dbsession=None):
    """
    query the predicted_score table and get the method
    field for the last row.
    """
    if not dbsession:
        dbsession=session
    rows = dbsession.query(Fixture)\
                  .filter_by(season=season).all()
    return rows[-1].tag


def fixture_probabilities(gameweek, season=CURRENT_SEASON, dbsession=None):
    """
    Returns probabilities for all fixtures in a given gameweek and season, as a data frame with a row 
    for each fixture and columns being fixture_id, home_team, away_team, home_win_probability, 
    draw_probability, away_win_probability.
    """
    model_team = get_fitted_team_model(season, dbsession)
    fixture_probabilities_list = []
    fixture_id_list = []
    for fixture in get_fixtures_for_season():
        if fixture.gameweek == gameweek:
            probabilities = model_team.overall_probabilities(
                fixture.home_team, fixture.away_team)
            fixture_probabilities_list.append(
                [fixture.fixture_id, fixture.home_team, fixture.away_team, probabilities[0], probabilities[1], probabilities[2]])
            fixture_id_list.append(fixture.fixture_id)
    return pd.DataFrame(fixture_probabilities_list, columns=['fixture_id', 'home_team',
                                                          'away_team', 'home_win_probability', 'draw_probability', 'away_win_probability'], index=fixture_id_list)
