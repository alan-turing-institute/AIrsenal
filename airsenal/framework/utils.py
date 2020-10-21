"""
Useful commands to query the db
"""

from functools import lru_cache
from operator import itemgetter
from datetime import datetime, timezone
from typing import TypeVar
import pandas as pd
import dateparser
import re
from pickle import loads, dumps
from airsenal.framework.mappings import alternative_player_names

from airsenal.framework.data_fetcher import FPLDataFetcher
from airsenal.framework.schema import (
    Base,
    Player,
    PlayerAttributes,
    Result,
    Fixture,
    PlayerScore,
    PlayerPrediction,
    Transaction,
    Team,
    engine,
)
from airsenal.framework.season import CURRENT_SEASON
from airsenal.framework.bpl_interface import get_fitted_team_model

from sqlalchemy.orm import sessionmaker
from sqlalchemy import or_, case, func, desc

Base.metadata.bind = engine
DBSession = sessionmaker()
session = DBSession()

fetcher = FPLDataFetcher()  # in global scope so it can keep cached data


def get_max_gameweek(season=CURRENT_SEASON, dbsession=session):
    """
    Return the maximum gameweek number across all scheduled fixtures. This shuold
    generally be 38, but may be different in the case of major disruptino (e.g.
    Covid-19)
    """
    max_gw = (
        dbsession.query(func.max(Fixture.gameweek)).filter_by(season=season).first()[0]
    )
    if max_gw is None:
        # TODO Tests fail without this as tests don't populate fixture table in db
        max_gw = 100

    return max_gw


def get_next_gameweek(season=CURRENT_SEASON, dbsession=None):
    """
    Use the current time to figure out which gameweek we're in
    """
    if not dbsession:
        dbsession = session
    timenow = datetime.now(timezone.utc)
    fixtures = dbsession.query(Fixture).filter_by(season=season).all()
    earliest_future_gameweek = get_max_gameweek(season, dbsession) + 1

    if len(fixtures) > 0:
        for fixture in fixtures:
            try:
                fixture_date = dateparser.parse(fixture.date)
                fixture_date = fixture_date.replace(tzinfo=timezone.utc)
                if (
                        fixture_date > timenow
                        and fixture.gameweek < earliest_future_gameweek
                ):
                    earliest_future_gameweek = fixture.gameweek
            except (TypeError):  ## date could be null if fixture not scheduled
                continue
        ## now make sure we aren't in the middle of a gameweek
        for fixture in fixtures:
            try:
                if (
                        dateparser.parse(fixture.date).replace(tzinfo=timezone.utc)
                        < timenow
                        and fixture.gameweek == earliest_future_gameweek
                ):
                    earliest_future_gameweek += 1
            except (TypeError):
                continue

    else:
        # got no fixtures from database, maybe we're filling it for the first
        # time - get next gameweek from API instead
        fixture_data = fetcher.get_fixture_data()

        if len(fixture_data) == 0:
            # if no fixtures scheduled assume this is start of season before
            # fixtures have been announced
            return 1

        for fixture in fixture_data:
            if (
                    fixture["finished"] is False
                    and fixture["event"]
                    and fixture["event"] < earliest_future_gameweek
            ):
                earliest_future_gameweek = fixture["event"]
        # check whether we're mid-gameweek
        for fixture in fixture_data:
            if (
                    fixture["finished"] is True
                    and fixture["event"] == earliest_future_gameweek
            ):
                earliest_future_gameweek += 1
                break

    return earliest_future_gameweek


# make this a global variable in this module, import into other modules
NEXT_GAMEWEEK = get_next_gameweek()


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


def get_current_players(gameweek=None, season=None, dbsession=None):
    """
    Use the transactions table to find the team as of specified gameweek,
    then add up the values at that gameweek using the FPL API data.
    If gameweek is None, get team for next gameweek
    """
    if not season:
        season = CURRENT_SEASON
    if not dbsession:
        dbsession = session
    current_players = []
    transactions = (
        dbsession.query(Transaction)
            .filter_by(season=season)
            .order_by(Transaction.gameweek)
            .all()
    )
    if len(transactions) == 0:
        #  not updated the transactions table yet
        return []
    for t in transactions:
        if gameweek and t.gameweek > gameweek:
            break
        if t.bought_or_sold == 1:
            current_players.append(t.player_id)
        else:
            current_players.remove(t.player_id)
    assert len(current_players) == 15
    return current_players


def get_squad_value(squad, gameweek=NEXT_GAMEWEEK, season=CURRENT_SEASON, use_api=False):
    """
    Use the transactions table to find the squad as of specified gameweek,
    then add up the values at that gameweek (using the FPL API if set), plus the
    amount in the bank.
    If gameweek is None, get team for next gameweek
    """
    total_value = squad.budget  # initialise total to amount in the bank

    for p in squad.players:
        total_value += squad.get_sell_price_for_player(
            p, use_api=use_api, season=season, gameweek=gameweek
        )

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
    # to query the API we need to use the fpl_api_id for the player rather than player_id
    player_api_id = get_player(player_id).fpl_api_id

    pdata_bought = fetcher.get_gameweek_data_for_player(player_api_id, gw_bought)
    ## will be a list - can be more than one match in a gw - just use the 1st.
    price_bought = pdata_bought[0]["value"]

    if not gameweek:  # assume we want the current (i.e. next) gameweek
        price_now = fetcher.get_player_summary_data()[player_api_id]["now_cost"]
    else:
        pdata_now = fetcher.get_gameweek_data_for_player(player_api_id, gw_bought)
        price_now = pdata_now[0]["value"]
    ## take off our half of the profit - boo!
    if price_now > price_bought:
        value = (price_now + price_bought) // 2  # round down
    else:
        value = price_now
    return value


def get_bank(gameweek=None, fpl_team_id=None):
    """
    Find out how much this FPL team had in the bank before the specified gameweek.
    If gameweek is not provided, give the most recent value
    If fpl_team_id is not specified, will use the FPL_TEAM_ID environment var, or
    the contents of the file airsenal/data/FPL_TEAM_ID.
    """
    data = fetcher.get_fpl_team_history_data(fpl_team_id)
    if "current" in data.keys() and len(data["current"]) > 0:
        if gameweek and isinstance(gameweek, int):
            for gw in data["current"]:
                if gw["event"] == gameweek - 1: # value after previous gameweek
                    return gw["bank"]
        # otherwise, return the most recent value
        return data["current"][-1]["bank"]
    else:
        return 0


def get_free_transfers(gameweek=None, fpl_team_id=None):
    """
    Work out how many free transfers this FPL team should have before specified gameweek.
    If gameweek is not provided, give the most recent value
    If fpl_team_id is not specified, will use the FPL_TEAM_ID environment var, or
    the contents of the file airsenal/data/FPL_TEAM_ID.
    """
    data = fetcher.get_fpl_team_history_data(fpl_team_id)
    num_free_transfers = 1
    if "current" in data.keys() and len(data["current"]) > 0:
        for gw in data["current"]:
            if gw["event_transfers"] == 0 and num_free_transfers < 2:
                num_free_transfers += 1
            if gw["event_transfers"] == 2:
                num_free_transfers = 1
            # if gameweek was specified, and we reached the previous one, break out of loop.
            if gameweek and gw["event"] == gameweek - 1:
                break
    return num_free_transfers


def get_gameweek_by_date(date, dbsession=None):
    """
    Use the dates of the fixtures to find the gameweek.
    """
    # convert date to a datetime object if it isn't already one.
    if not dbsession:
        dbsession = session
    if not isinstance(date, datetime):
        date = dateparser.parse(date)
    fixtures = dbsession.query(Fixture).all()
    for fixture in fixtures:
        try:
            fixture_date = dateparser.parse(fixture.date)
            if fixture_date.date() == date.date():
                return fixture.gameweek
        except (TypeError):  # NULL date if fixture not scheduled
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
    query the player table by name or id, return the player object (or None).
    NOTE the player_id that can be passed as an argument here is NOT
    guaranteed to be the id for that player in the FPL API.  The one here
    is the entry (primary key) in our database.
    Use the function get_player_from_api_id() to find the player corresponding
    to the FPL API ID.
    """
    if not dbsession:
        dbsession = session  # use the one defined in this module

    # if an id has been passed as a string, convert it to an integer
    if isinstance(player_name_or_id, str) and player_name_or_id.isdigit():
        player_name_or_id = int(player_name_or_id)

    if isinstance(player_name_or_id, int):
        filter_attr = Player.player_id
    else:
        filter_attr = Player.name
    p = dbsession.query(Player).filter(filter_attr == player_name_or_id).first()
    if p:
        return p
    if isinstance(player_name_or_id, int):  # didn't find by id - return None
        return None
    # assume we have a name, now try alternative names
    for k, v in alternative_player_names.items():
        if player_name_or_id in v:
            p = dbsession.query(Player).filter_by(name=k).first()
            if p:
                return p
    # didn't find it - return None
    return None


def get_player_from_api_id(api_id, dbsession=None):
    """
    Query the database and return the player with the corresponding attribute fpl_api_id
    """
    if not dbsession:
        dbsession = session  # use the one defined in this module
    p = dbsession.query(Player).filter_by(fpl_api_id=api_id).first()
    if p:
        return p
    else:
        print("Unable to find player with fpl_api_id {}".format(api_id))
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


def list_teams(season=CURRENT_SEASON, dbsession=None):
    """
    Print all teams from current season.
    """
    if not dbsession:
        dbsession = session
    rows = dbsession.query(Team).filter_by(season=season).all()
    return [{"name": row.name, "full_name": row.full_name} for row in rows]


def list_players(
        position="all",
        team="all",
        order_by="price",
        season=CURRENT_SEASON,
        dbsession=None,
        verbose=False,
        gameweek=NEXT_GAMEWEEK,
):
    """
    print list of players, and
    return a list of player_ids
    """
    if not dbsession:
        dbsession = session

    # if trying to get players from the future, return current players
    if season == CURRENT_SEASON and gameweek > NEXT_GAMEWEEK:
        gameweek = NEXT_GAMEWEEK

    gameweeks = [gameweek]
    # check if the team (or all teams) play in the specified gameweek, if not
    # attributes might be missing
    fixtures = get_fixtures_for_gameweek(gameweek, season=season, dbsession=dbsession)
    teams_with_fixture = [t for fixture in fixtures for t in fixture]
    teams_with_fixture = set(teams_with_fixture)

    if (team == "all" and len(teams_with_fixture) < 20) or (
            team != "all" and team not in teams_with_fixture
    ):
        # check neighbouring gameweeks to get all 20 teams/specified team
        gws_to_try = [gameweek - 1, gameweek + 1, gameweek - 2, gameweek + 2]
        max_gw = get_max_gameweek(season, dbsession)
        gws_to_try = [gw for gw in gws_to_try if gw > 0 and gw <= max_gw]

        for gw in gws_to_try:
            fixtures = get_fixtures_for_gameweek(gw, season=season, dbsession=dbsession)
            new_teams = [t for fixture in fixtures for t in fixture]

            if team == "all" and any([t not in teams_with_fixture for t in new_teams]):
                # this gameweek has some teams we haven't seen before
                gameweeks.append(gw)
                [teams_with_fixture.add(t) for t in new_teams]
                if len(teams_with_fixture) == 20:
                    break

            elif team != "all" and team in new_teams:
                # this gameweek has the team we're looking for
                gameweeks.append(gw)
                break

    q = (
        dbsession.query(PlayerAttributes)
            .filter_by(season=season)
            .filter(PlayerAttributes.gameweek.in_(gameweeks))
    )
    if team != "all":
        q = q.filter_by(team=team)
    if position != "all":
        q = q.filter_by(position=position)
    if len(gameweeks) > 1:
        #  Sort query results by order of gameweeks - i.e. make sure the input
        # query gameweek comes first.
        _whens = {gw: i for i, gw in enumerate(gameweeks)}
        sort_order = case(value=PlayerAttributes.gameweek, whens=_whens)
        q = q.order_by(sort_order)
    if order_by == "price":
        q = q.order_by(PlayerAttributes.price.desc())
    players = []
    prices = []
    for p in q.all():
        if p.player not in players:
            # might have queried multiple gameweeks with same player returned
            #  multiple times - only add if it's a new player
            players.append(p.player)
            prices.append(p.price)
            if verbose and (len(gameweeks) == 1 or order_by != "price"):
                print(p.player.name, p.team, p.position, p.price)
    if len(gameweeks) > 1 and order_by == "price":
        # Query sorted by gameweek first, so need to do a final sort here to
        # get final price order if more than one gameweek queried.
        sort_players = sorted(zip(prices, players), reverse=True, key=lambda p: p[0])
        if verbose:
            for p in sort_players:
                print(p[1].name, p[0])
        players = [p for _, p in sort_players]
    return players


def get_max_matches_per_player(position="all", season=CURRENT_SEASON, dbsession=None):
    """
    can be used e.g. in bpl_interface.get_player_history_df
    to help avoid a ragged dataframe.
    """
    players = list_players(position=position, season=season, dbsession=dbsession)
    max_matches = 0
    for p in players:
        num_match = len(p.scores)
        if num_match > max_matches:
            max_matches = num_match
    return max_matches


def get_player_attributes(
        player_name_or_id, season=CURRENT_SEASON, gameweek=NEXT_GAMEWEEK, dbsession=None
):
    """Get a player's attributes for a given gameweek in a given season."""

    if not dbsession:
        dbsession = session

    if isinstance(player_name_or_id, str) and player_name_or_id.isdigit():
        player_id = int(player_name_or_id)
    elif isinstance(player_name_or_id, int):
        player_id = player_name_or_id
    elif isinstance(player_name_or_id, str):
        player = get_player(player_name_or_id)
        if player:
            player_id = player.player_id
        else:
            return None

    attr = (
        dbsession.query(PlayerAttributes)
            .filter_by(season=season)
            .filter_by(gameweek=gameweek)
            .filter_by(player_id=player_id)
            .first()
    )

    return attr


def get_fixtures_for_player(
        player, season=CURRENT_SEASON, gw_range=None, dbsession=None, verbose=False
):
    """
    search for upcoming fixtures for a player, specified either by id or name.
    If gw_range not specified:
       for current season: return fixtures from now to end of season
       for past seasons: return all fixtures in the season
    """
    if not dbsession:
        dbsession = session
    player_query = dbsession.query(Player)
    if isinstance(player, str):  # given a player name
        player_record = player_query.filter_by(name=player).first()
    elif isinstance(player, int):  # given a player id
        player_record = player_query.filter_by(player_id=player).first()
    else:  # given a player object
        player_record = player
    if not player_record:
        print("Couldn't find {} in database".format(player))
        return []
    team = player_record.team(season, gw_range[0])  # same team for whole gw_range
    tag = get_latest_fixture_tag(season, dbsession)
    fixture_rows = (
        dbsession.query(Fixture)
            .filter_by(season=season)
            .filter_by(tag=tag)
            .filter(or_(Fixture.home_team == team, Fixture.away_team == team))
            .order_by(Fixture.gameweek)
            .all()
    )
    fixtures = []
    for fixture in fixture_rows:
        if not fixture.gameweek:  # fixture not scheduled yet
            continue
        if gw_range:
            if fixture.gameweek in gw_range:
                fixtures.append(fixture)
        else:
            if season == CURRENT_SEASON and fixture.gameweek < NEXT_GAMEWEEK:
                continue
            if verbose:
                print(
                    "{} vs {} gameweek {}".format(
                        fixture.home_team, fixture.away_team, fixture.gameweek
                    )
                )
            fixtures.append(fixture)
    return fixtures


def get_next_fixture_for_player(
        player, season=CURRENT_SEASON, gameweek=NEXT_GAMEWEEK, dbsession=None
):
    """
    Get a players next fixture as a string, for easy displaying
    """
    if not dbsession:
        dbsession = session
    # given a player name or id, convert to player object
    if isinstance(player, str) or isinstance(player, int):
        player = get_player(player)
    team = player.team(season, gameweek)
    fixtures_for_player = get_fixtures_for_player(player, season, [gameweek], dbsession)
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
    fixtures = dbsession.query(Fixture).filter_by(season=season).all()
    return fixtures


def get_fixtures_for_gameweek(gameweek, season=CURRENT_SEASON, dbsession=session):
    """
    Get a list of fixtures for the specified gameweek
    """
    fixtures = (
        dbsession.query(Fixture)
            .filter_by(season=season)
            .filter_by(gameweek=gameweek)
            .all()
    )
    return [(fixture.home_team, fixture.away_team) for fixture in fixtures]


def get_result_for_fixture(fixture, dbsession=session):
    """Get result for a fixture."""
    result = session.query(Result).filter_by(fixture=fixture).all()
    return result


def get_player_scores_for_fixture(fixture, dbsession=session):
    """Get player scores for a fixture."""
    player_scores = session.query(PlayerScore).filter_by(fixture=fixture).all()
    return player_scores


def get_players_for_gameweek(gameweek):
    """
    Use FPL API to get the players for a given gameweek.
    """
    player_data = fetcher.get_fpl_team_data(gameweek)["picks"]
    player_api_id_list = [p["element"] for p in player_data]
    player_list = [
        get_player_from_api_id(api_id).player_id
        for api_id in player_api_id_list
        if get_player_from_api_id(api_id)
    ]
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
                .filter_by(player_id=player_id, fixture_id=fid[0])
                .all()
        )
        for s in scores:
            previous_points[fid[1]] = s.points

    return previous_points


@lru_cache(maxsize=4096)
def get_predicted_points_for_player(player, tag, season=CURRENT_SEASON, dbsession=None):
    """
    Query the player prediction table for a given player.
    Return a dict, keyed by gameweek.
    """
    if not dbsession:
        dbsession = session
    if isinstance(player, str) or isinstance(player, int):
        # we want the actual player object
        player = get_player(player, dbsession=dbsession)
    pps = (
        dbsession.query(PlayerPrediction)
            .filter(PlayerPrediction.fixture.has(Fixture.season == season))
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
    max_gw = get_max_gameweek(season, dbsession)
    for gw in range(1, max_gw + 1):
        if not gw in ppdict.keys():
            ppdict[gw] = 0.0
    return ppdict


def get_predicted_points(
        gameweek, tag, position="all", team="all", season=CURRENT_SEASON, dbsession=None
):
    """
    Query the player_prediction table with selections, return
    list of tuples (player_id, predicted_points) ordered by predicted_points
    "gameweek" argument can either be a single integer for one gameweek, or a
    list of gameweeks, in which case we will get the sum over all of them
    """
    players = list_players(position, team, season=season, dbsession=dbsession)

    if isinstance(gameweek, int):  # predictions for a single gameweek
        output_list = [
            (
                p,
                get_predicted_points_for_player(
                    p, tag=tag, season=season, dbsession=dbsession
                )[gameweek],
            )
            for p in players
        ]
    else:  # predictions for a list of gameweeks
        output_list = [
            (
                p,
                sum(
                    get_predicted_points_for_player(
                        p, tag=tag, season=season, dbsession=dbsession
                    )[gw]
                    for gw in gameweek
                ),
            )
            for p in players
        ]

    output_list.sort(key=itemgetter(1), reverse=True)
    return output_list


def get_top_predicted_points(
        gameweek=None,
        tag=None,
        position="all",
        team="all",
        n_players=10,
        per_position=False,
        max_price=None,
        season=CURRENT_SEASON,
        dbsession=None,
):
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
        gameweek = NEXT_GAMEWEEK

    if isinstance(gameweek, list) or isinstance(gameweek, tuple):
        # for determining position, team and price below
        first_gw = gameweek[0]
    else:
        first_gw = gameweek

    print("=" * 50)
    print("PREDICTED TOP {} PLAYERS FOR GAMEWEEK(S) {}:".format(n_players, gameweek))
    print("=" * 50)

    if not per_position:
        pts = get_predicted_points(
            gameweek,
            tag,
            position=position,
            team=team,
            season=season,
            dbsession=dbsession,
        )

        if max_price is not None:
            pts = [p for p in pts if p[0].price(season, first_gw) <= max_price]

        pts = sorted(pts, key=lambda x: x[1], reverse=True)

        for i, p in enumerate(pts[:n_players]):
            print(
                "{}. {}, {:.2f}pts (£{}m, {}, {})".format(
                    i + 1,
                    p[0].name,
                    p[1],
                    p[0].price(season, first_gw) / 10,
                    p[0].position(season),
                    p[0].team(season, first_gw),
                )
            )

    else:
        for position in ["GK", "DEF", "MID", "FWD"]:
            pts = get_predicted_points(
                gameweek,
                tag,
                position=position,
                team=team,
                season=season,
                dbsession=dbsession,
            )
            if max_price is not None:
                pts = [p for p in pts if p[0].price(season, first_gw) <= max_price]

            pts = sorted(pts, key=lambda x: x[1], reverse=True)
            print("{}:".format(position))

            for i, p in enumerate(pts[:n_players]):
                print(
                    "{}. {}, {:.2f}pts (£{}m, {})".format(
                        i + 1,
                        p[0].name,
                        p[1],
                        p[0].price(season, first_gw) / 10,
                        p[0].team(season, first_gw),
                    )
                )
            print("-" * 25)


def get_return_gameweek_for_player(player_api_id, dbsession=None):
    """
    If  a player is injured and there is 'news' about them on FPL,
    parse this string to get expected return date.
    """
    pdata = fetcher.get_player_summary_data()[player_api_id]
    rd_rex = "(Expected back|Suspended until)[\\s]+([\\d]+[\\s][\\w]{3})"
    if "news" in pdata.keys() and re.search(rd_rex, pdata["news"]):

        return_str = re.search(rd_rex, pdata["news"]).groups()[1]
        # return_str should be a day and month string (without year)

        # create a date in the future from the day and month string
        return_date = dateparser.parse(
            return_str, settings={"PREFER_DATES_FROM": "future"}
        )

        if not return_date:
            raise ValueError(
                "Failed to parse date from string '{}'".format(return_date)
            )

        return_gameweek = get_gameweek_by_date(return_date, dbsession=dbsession)
        return return_gameweek
    return None


def calc_average_minutes(player_scores):
    """
    Simple average of minutes played for a list of PlayerScore objects.
    """
    total = 0.0
    for ps in player_scores:
        total += ps.minutes
    return total / len(player_scores)


def estimate_minutes_from_prev_season(
        player,
        season=CURRENT_SEASON,
        dbsession=None,
        gameweek=NEXT_GAMEWEEK,
        n_games_to_use=10,
):
    """
    take average of minutes from previous season if any, or else return [60]
    """
    if not dbsession:
        dbsession = session
    previous_season = get_previous_season(season)

    # Only consider minutes the player played with his
    # current team in the previous season.
    current_team = player.team(season, gameweek)

    player_scores = (
        dbsession.query(PlayerScore)
            .filter_by(player_id=player.player_id)
            .filter(PlayerScore.fixture.has(season=previous_season))
            .filter_by(player_team=current_team)
            .join(Fixture, PlayerScore.fixture)
            .order_by(desc(Fixture.gameweek))
            .limit(n_games_to_use)
            .all()
    )

    if len(player_scores) == 0:
        # If this player didn't play for his current team last season, return 0 minutes
        return [0]
    else:
        average_mins = calc_average_minutes(player_scores)
        return [average_mins]


def get_recent_playerscore_rows(
        player, num_match_to_use=3, season=CURRENT_SEASON, last_gw=None, dbsession=None
):
    """
    Query the playerscore table in the database to retrieve
    the last num_match_to_use rows for this player.
    """
    if not dbsession:
        dbsession = session

    if not last_gw:
        last_gw = NEXT_GAMEWEEK
    # If asking for gameweeks without results in DB, revert to most recent results.
    last_available_gameweek = get_last_gameweek_in_db(
        season=season, dbsession=dbsession
    )
    if not last_available_gameweek:
        # e.g. before this season has started
        return None

    if last_gw > last_available_gameweek:
        last_gw = last_available_gameweek

    first_gw = last_gw - num_match_to_use
    ## get the playerscore rows from the db
    rows = (
        dbsession.query(PlayerScore)
            .filter(PlayerScore.fixture.has(season=season))
            .filter_by(player_id=player.player_id)
            .filter(PlayerScore.fixture.has(Fixture.gameweek > first_gw))
            .filter(PlayerScore.fixture.has(Fixture.gameweek <= last_gw))
            .all()
    )
    ## for speed, we use the fact that matches from this season
    ## are uploaded in order, so we can just take the last n
    ## rows, no need to look up dates and sort.
    return rows[-num_match_to_use:]


def get_recent_scores_for_player(
        player, num_match_to_use=3, season=CURRENT_SEASON, last_gw=None, dbsession=None
):
    """
    Look num_match_to_use matches back, and return the
    FPL points for this player for each of these matches.
    Return a dict {gameweek: score, }
    """
    if not last_gw:
        last_gw = NEXT_GAMEWEEK
    first_gw = last_gw - num_match_to_use

    playerscores = get_recent_playerscore_rows(
        player, num_match_to_use, season, last_gw, dbsession
    )
    if not playerscores:  # e.g. start of season
        return None

    points = {}
    for i, ps in enumerate(playerscores):
        points[range(first_gw, last_gw)[i]] = ps.points
    return points


def get_recent_minutes_for_player(
        player, num_match_to_use=3, season=CURRENT_SEASON, last_gw=None, dbsession=None
):
    """
    Look back num_match_to_use matches, and return an array
    containing minutes played in each.
    If current_gw is not given, we take it to be the most
    recent finished gameweek.
    """
    playerscores = get_recent_playerscore_rows(
        player, num_match_to_use, season, last_gw, dbsession
    )
    if playerscores:
        minutes = [r.minutes for r in playerscores]
    else:
        # got no playerscores, e.g. start of season
        minutes = []

    # if going back num_matches_to_use from last_gw takes us before the start
    # of the season, also include a minutes estimate using last season's data
    if not last_gw:
        last_gw = NEXT_GAMEWEEK
    first_gw = last_gw - num_match_to_use
    if first_gw < 0 or len(minutes) == 0:
        minutes = minutes + estimate_minutes_from_prev_season(player, season, dbsession)

    return minutes


def get_last_gameweek_in_db(season=CURRENT_SEASON, dbsession=None):
    """
    query the result table to see what was the last gameweek for which
    we have filled the data.
    """
    if not dbsession:
        dbsession = session
    last_result = (
        dbsession.query(Fixture)
            .filter_by(season=season)
            .filter(Fixture.result != None)
            .order_by(Fixture.gameweek.desc())
            .first()
    )
    if last_result:
        return last_result.gameweek
    else:
        return None


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


def get_latest_prediction_tag(season=CURRENT_SEASON, dbsession=None):
    """
    query the predicted_score table and get the method
    field for the last row.
    """
    if not dbsession:
        dbsession = session
    rows = (
        dbsession.query(PlayerPrediction)
            .filter(PlayerPrediction.fixture.has(Fixture.season == season))
            .all()
    )
    try:
        return rows[-1].tag
    except (IndexError):
        raise RuntimeError(
            "No predicted points in database - has the database been filled?\n"
            "To calculate points predictions (and fill the database) use "
            "'airsenal_run_prediction'. This should be done before using "
            "'airsenal_make_squad' or 'airsenal_run_optimization'."
        )


def get_latest_fixture_tag(season=CURRENT_SEASON, dbsession=None):
    """
    query the predicted_score table and get the method
    field for the last row.
    """
    if not dbsession:
        dbsession = session
    rows = dbsession.query(Fixture).filter_by(season=season).all()
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
    for fixture in get_fixtures_for_gameweek(
            gameweek, season=season, dbsession=dbsession
    ):
        probabilities = model_team.overall_probabilities(
            fixture.home_team, fixture.away_team
        )
        fixture_probabilities_list.append(
            [
                fixture.fixture_id,
                fixture.home_team,
                fixture.away_team,
                probabilities[0],
                probabilities[1],
                probabilities[2],
            ]
        )
        fixture_id_list.append(fixture.fixture_id)
    return pd.DataFrame(
        fixture_probabilities_list,
        columns=[
            "fixture_id",
            "home_team",
            "away_team",
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
        ],
        index=fixture_id_list,
    )


def find_fixture(
        gameweek,
        team,
        was_home=None,
        other_team=None,
        kickoff_time=None,
        season=CURRENT_SEASON,
        dbsession=session,
):
    """Get a fixture given a gameweek, team and optionally whether
    the team was at home or away, the kickoff time and the other team in the
    fixture.
    """
    fixture = None

    if not isinstance(team, str):
        team_name = get_team_name(team, season=season, dbsession=dbsession)
    else:
        team_name = team

    if not team_name:
        raise ValueError("No team with id {} in {} season".format(team, season))

    if other_team and not isinstance(other_team, str):
        other_team_name = get_team_name(other_team, season=season, dbsession=dbsession)
    else:
        other_team_name = other_team

    query = (
        dbsession.query(Fixture).filter_by(gameweek=gameweek).filter_by(season=season)
    )
    if was_home is True:
        query = query.filter_by(home_team=team_name)
    elif was_home is False:
        query = query.filter_by(away_team=team_name)
    elif was_home is None:
        query = query.filter(
            or_(Fixture.away_team == team_name, Fixture.home_team == team_name)
        )
    else:
        raise ValueError("was_home must be True, False or None")

    if other_team_name:
        if was_home is True:
            query = query.filter_by(away_team=other_team_name)
        elif was_home is False:
            query = query.filter_by(home_team=other_team_name)
        elif was_home is None:
            query = query.filter(
                or_(
                    Fixture.away_team == other_team_name,
                    Fixture.home_team == other_team_name,
                )
            )

    fixtures = query.all()

    if not fixtures or len(fixtures) == 0:
        raise ValueError(
            "No fixture with season={}, gw={}, team_name={}, was_home={}, other_team_name={}".format(
                season, gameweek, team_name, was_home, other_team_name
            )
        )

    if len(fixtures) == 1:
        fixture = fixtures[0]
    elif kickoff_time:
        # team played multiple games in the gameweek, determine the
        # fixture of interest using the kickoff time,
        kickoff_date = dateparser.parse(kickoff_time)
        kickoff_date = kickoff_date.replace(tzinfo=timezone.utc)
        kickoff_date = kickoff_date.date()

        for f in fixtures:
            f_date = dateparser.parse(f.date)
            f_date = f_date.replace(tzinfo=timezone.utc)
            f_date = f_date.date()
            if f_date == kickoff_date:
                fixture = f
                break

    if not fixture:
        raise ValueError(
            "No unique fixture with season={}, gw={}, team_name={}, was_home={}, kickoff_time={}".format(
                season, gameweek, team_name, was_home, kickoff_time
            )
        )

    return fixture


def get_player_team_from_fixture(
        gameweek,
        opponent,
        player_at_home=None,
        kickoff_time=None,
        season=CURRENT_SEASON,
        dbsession=session,
        return_fixture=False,
):
    """Get the team a player played for given the gameweek, opponent, time and
    whether they were home or away.

    If return_fixture is True, return a tuple of (team_name, fixture)
    """

    if player_at_home is True:
        opponent_was_home = False
    elif player_at_home is False:
        opponent_was_home = True
    elif player_at_home is None:
        opponent_was_home = None

    fixture = find_fixture(
        gameweek,
        opponent,
        was_home=opponent_was_home,
        kickoff_time=kickoff_time,
        season=season,
        dbsession=dbsession,
    )

    player_team = None

    if player_at_home is not None:
        if player_at_home:
            player_team = fixture.home_team
        else:
            player_team = fixture.away_team

    else:
        if not isinstance(opponent, str):
            opponent_name = get_team_name(opponent, season=season, dbsession=dbsession)

        if fixture.home_team == opponent_name:
            player_team = fixture.away_team
        elif fixture.away_team == opponent_name:
            player_team = fixture.home_team
        else:
            raise ValueError("Opponent {} not in fixture".format(opponent_name))

    if return_fixture:
        return (player_team, fixture)
    else:
        return player_team


T = TypeVar("T")


def fastcopy(obj: T) -> T:
    """ faster replacement for copy.deepcopy()"""
    return loads(dumps(obj, -1))
