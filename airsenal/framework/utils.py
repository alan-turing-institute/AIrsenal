"""
Useful commands to query the database.
"""

import warnings
from collections.abc import Iterable
from datetime import date, datetime, timezone
from functools import lru_cache
from operator import itemgetter
from pickle import dumps, loads
from typing import TypeVar

import dateparser
import pandas as pd
import regex as re
from bpl import ExtendedDixonColesMatchPredictor, NeutralDixonColesMatchPredictor
from curl_cffi import requests
from dateutil.parser import isoparse
from sqlalchemy import case, or_
from sqlalchemy.orm.session import Session

from airsenal.framework.data_fetcher import FPLDataFetcher
from airsenal.framework.random_team_model import RandomMatchPredictor
from airsenal.framework.schema import (
    Absence,
    Fixture,
    Player,
    PlayerAttributes,
    PlayerMapping,
    PlayerPrediction,
    PlayerScore,
    Team,
    Transaction,
    session,
)
from airsenal.framework.season import CURRENT_SEASON

fetcher = FPLDataFetcher()  # in global scope so it can keep cached data


@lru_cache(1)
def get_max_gameweek(season: str = CURRENT_SEASON, dbsession: Session = session) -> int:
    """
    Return the maximum gameweek number across all scheduled fixtures. This should
    generally be 38, but may be different with major disruptions (e.g. Covid-19).
    """
    max_gw_fixture = (
        dbsession.query(Fixture)
        .filter_by(season=season)
        .filter(Fixture.gameweek.isnot(None))
        .order_by(Fixture.gameweek.desc())
        .first()
    )
    return (
        38
        if max_gw_fixture is None or max_gw_fixture.gameweek is None
        else max_gw_fixture.gameweek
    )


def get_next_gameweek(
    season: str = CURRENT_SEASON, dbsession: Session = session
) -> int:
    """
    Use the current time to figure out which gameweek we are currently in.
    """
    timenow = datetime.now(timezone.utc)
    fixtures = dbsession.query(Fixture).filter_by(season=season).all()
    earliest_future_gameweek = get_max_gameweek(season, dbsession) + 1

    if len(fixtures) > 0:
        for fixture in fixtures:
            if fixture.date is None or fixture.gameweek is None:
                # date could be null if fixture not scheduled
                continue
            fixture_date = parse_datetime(fixture.date).replace(tzinfo=timezone.utc)
            if fixture_date > timenow and fixture.gameweek < earliest_future_gameweek:
                earliest_future_gameweek = fixture.gameweek

        # now make sure we aren't in the middle of a gameweek
        for fixture in fixtures:
            if not fixture.date:
                # date could be null if fixture not scheduled
                continue
            if (
                parse_datetime(fixture.date).replace(tzinfo=timezone.utc) < timenow
                and fixture.gameweek == earliest_future_gameweek
            ):
                earliest_future_gameweek += 1
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


@lru_cache(365)
def parse_datetime(check_date: datetime | str) -> datetime:
    if isinstance(check_date, datetime):
        return check_date
    try:
        dt: datetime | None = isoparse(check_date)
    except (ValueError, TypeError):
        dt = dateparser.parse(check_date)
    if dt is None:
        msg = f"Unable to parse date: {check_date}"
        raise ValueError(msg)
    return dt


@lru_cache(365)
def parse_date(check_date: date | datetime | str) -> date:
    return (
        check_date
        if isinstance(check_date, date)
        else parse_datetime(check_date).date()
    )


@lru_cache(365)
def get_return_gameweek_by_date(
    return_date: date | datetime | str,
    team: str,
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> int:
    """
    Use a date, or easily parse-able date string, and team name to determine the
    gameweek of the next match for that team on or after that date. If no match
    is found, return a placeholder gameweek after the end of the season.
    """
    if not dbsession:
        dbsession = session

    return_date = parse_date(return_date)

    fixtures = (
        dbsession.query(Fixture)
        .filter_by(season=season)
        .filter(or_(Fixture.away_team == team, Fixture.home_team == team))
        .filter(Fixture.date.isnot(None))
        .order_by(Fixture.date)
    ).all()

    # default return if no fixture found after the date
    end_season_gw = get_max_gameweek(season, dbsession) + 1

    if len(fixtures) == 0:
        return end_season_gw

    for fixture in fixtures:
        if fixture.date is None or fixture.gameweek is None:
            # should be filtered out by query, but to keep mypy happy
            continue
        fixture_date = parse_date(fixture.date)
        if fixture_date >= return_date:
            return fixture.gameweek

    return end_season_gw


def get_gameweeks_array(
    weeks_ahead: int | None = None,
    gameweek_start: int | None = None,
    gameweek_end: int | None = None,
    season: str = CURRENT_SEASON,
    dbsession: Session = session,
) -> list[int]:
    """
    Returns the array containing only the valid (< max_gameweek) game-weeks
    or raise an exception if no game-weeks remaining.
    """
    # Check arguments are valid
    if gameweek_end is not None and weeks_ahead is not None:
        msg = "Only one of gameweek_end and weeks_ahead should be defined"
        raise RuntimeError(msg)
    if gameweek_start is None and season != CURRENT_SEASON:
        msg = "gameweek_start must be defined if using previous seasons"
        raise RuntimeError(msg)

    # Set defaults for undefined arguments
    if weeks_ahead is None:
        weeks_ahead = 3
    if gameweek_start is None:
        gameweek_start = NEXT_GAMEWEEK
    if gameweek_end is None:
        gameweek_end = gameweek_start + weeks_ahead

    gw_range = list(range(gameweek_start, gameweek_end))
    max_gameweek = get_max_gameweek(season=season, dbsession=dbsession)
    gw_range = list(filter(lambda x: x <= max_gameweek, gw_range))

    if len(gw_range) == 0:
        msg = "No gameweeks in specified range"
        raise ValueError(msg)
    if max(gw_range) < gameweek_end - 1:
        print(
            f"WARN: Last gameweek set to {max(gw_range)} ({len(gw_range)} weeks ahead)"
        )

    return gw_range


# make this a global variable in this module, import into other modules
NEXT_GAMEWEEK = get_next_gameweek()


def get_next_season(season: str) -> str:
    """
    Convert string e.g. '1819' into one for next season, i.e. '1920'.
    """
    start_year = int(season[:2])
    end_year = int(season[2:])
    next_start_year = (
        f"0{start_year + 1}" if start_year + 1 < 10 else str(start_year + 1)
    )
    next_end_year = f"0{end_year + 1}" if end_year + 1 < 10 else str(end_year + 1)
    return f"{next_start_year}{next_end_year}"


def get_start_end_dates_of_season(season: str) -> list[pd.Timestamp]:
    """
    Obtains rough start and end dates for the season.
    Takes into account the shorter and longer seasons in 19/20 and 20/21.
    """
    start_year = int(f"20{season[:2]}")
    end_year = int(f"20{season[2:]}")
    if season == "1920":
        # regular start, late end to season
        return [pd.Timestamp(2019, 7, 1), pd.Timestamp(2020, 7, 31)]
    if season == "2021":
        # late start to season, regular end
        return [pd.Timestamp(2020, 8, 1), pd.Timestamp(2021, 6, 30)]
    # regular season
    return [pd.Timestamp(start_year, 7, 1), pd.Timestamp(end_year, 6, 30)]


def get_previous_season(season: str) -> str:
    """
    Convert string e.g. '1819' into one for previous season, i.e. '1718'
    """
    start_year = int(season[:2])
    end_year = int(season[2:])
    prev_start_year = start_year - 1
    prev_end_year = end_year - 1
    return f"{prev_start_year}{prev_end_year}"


def get_past_seasons(num_seasons: int) -> list[str]:
    """
    Go back num_seasons from the current one.
    """
    season = CURRENT_SEASON
    seasons = []
    for _ in range(num_seasons):
        season = get_previous_season(season)
        seasons.append(season)
    return seasons


def get_current_players(
    gameweek: int | None = None,
    season: str | None = None,
    fpl_team_id: int | None = None,
    dbsession: Session | None = None,
) -> list[int]:
    """
    Use the transactions table to find the team as of specified gameweek,
    then add up the values at that gameweek using the FPL API data.
    If gameweek is None, get team for next gameweek.
    """
    if not fpl_team_id:
        fpl_team_id = fetcher.FPL_TEAM_ID
    if not season:
        season = CURRENT_SEASON
    if not dbsession:
        dbsession = session
    current_players = []
    transactions = (
        dbsession.query(Transaction)
        .order_by(Transaction.gameweek, Transaction.id)
        .filter_by(fpl_team_id=fpl_team_id)
        .filter_by(free_hit=0)  # free_hit players shouldn't be considered part of squad
        .filter_by(season=season)
        .all()
    )

    if len(transactions) == 0:
        # not updated the transactions table yet
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


def get_bank(
    fpl_team_id: int | None = None,
    gameweek: int | None = None,
    season: str = CURRENT_SEASON,
    apifetcher: FPLDataFetcher = fetcher,
) -> float:
    """
    Find out how much this FPL team had in the bank before the specified gameweek.
    If gameweek is not provided, give the most recent value.
    If fpl_team_id is not specified, will use the FPL_TEAM_ID environment var, or
    the contents of the file airsenal/data/FPL_TEAM_ID.
    """
    if season != CURRENT_SEASON:
        msg = "Calculating the bank for past seasons not yet implemented"
        raise RuntimeError(msg)

    if not fpl_team_id:
        fpl_team_id = fetcher.FPL_TEAM_ID
    # check if we're logged in, which will let us get the most up-to-date info
    try:
        return apifetcher.get_current_bank(fpl_team_id)
    except requests.exceptions.RequestException as e:
        warnings.warn(
            f"Failed to get actual bank from a logged in API:\n{e}\n"
            "Will try to estimate it from the API without logging in, which will "
            "not include any transfers made in the current gameweek.",
            stacklevel=2,
        )
        data = apifetcher.get_fpl_team_history_data(fpl_team_id)
        if "current" not in data or len(data["current"]) <= 0:
            return 0

        if gameweek and isinstance(gameweek, int):
            for gw in data["current"]:
                if gw["event"] == gameweek - 1:  # value after previous gameweek
                    return gw["bank"]
        # otherwise, return the most recent value
        return data["current"][-1]["bank"]


def get_entry_start_gameweek(
    fpl_team_id: int, apifetcher: FPLDataFetcher = fetcher
) -> int:
    """
    Find the gameweek an FPL team ID was entered in by searching for the first gameweek
    the API has 'picks' for.
    """
    starting_gw = 1
    while starting_gw < NEXT_GAMEWEEK:
        try:
            if get_players_for_gameweek(
                starting_gw, fpl_team_id, apifetcher=apifetcher
            ):
                return starting_gw
            starting_gw += 1
        except requests.exceptions.HTTPError:
            starting_gw += 1
        except requests.exceptions.ConnectionError as e:
            warnings.warn(
                f"Failed to connect to the API:\n{e}\n. Assuming team {fpl_team_id}"
                " was entered in GW1 which may be incorrect.",
                stacklevel=2,
            )
            return 1

    # if we failed to find picks in any gameweek, or we're before the start of the
    # season, assume this team ID was entered in NEXT_GAMEWEEK
    return NEXT_GAMEWEEK


def get_free_transfers(
    fpl_team_id: int | None = None,
    gameweek: int | None = None,
    season: str = CURRENT_SEASON,
    dbsession: Session = session,
    apifetcher: FPLDataFetcher = fetcher,
    is_replay: bool = False,
) -> int:
    """
    Work out how many free transfers FPL team should have before specified gameweek.
    If gameweek is not provided, give the most recent value.
    If fpl_team_id is not specified, will use the FPL_TEAM_ID environment var, or
    the contents of the file airsenal/data/FPL_TEAM_ID.
    """
    if season == CURRENT_SEASON and not is_replay:
        # we will use the API to estimate num transfers
        if not fpl_team_id:
            fpl_team_id = apifetcher.FPL_TEAM_ID

        # try to get the most up-to-date info from logged in api
        try:
            return apifetcher.get_num_free_transfers(fpl_team_id)
        except requests.exceptions.RequestException as e:
            warnings.warn(
                f"Failed to get actual free transfers from a logged in API:\n{e}\n"
                "Will try to estimate it from the API without logging in, which will "
                "not include any transfers used in the current gameweek.",
                stacklevel=2,
            )
        # try to calculate free transfers based on previous transfer history in API
        try:
            data = apifetcher.get_fpl_team_history_data(fpl_team_id)
            num_free_transfers = 1
            if "current" in data and len(data["current"]) > 0:
                starting_gw = get_entry_start_gameweek(
                    fpl_team_id, apifetcher=apifetcher
                )
                for gw in data["current"]:
                    if gw["event"] <= starting_gw:
                        continue
                    if gw["event_transfers"] == 0 and num_free_transfers < 2:
                        num_free_transfers += 1
                    elif gw["event_transfers"] >= 2:
                        num_free_transfers = 1
                    # if gameweek was specified, and we reached the previous one,
                    # break out of loop.
                    if gameweek and gw["event"] == gameweek - 1:
                        break
            return num_free_transfers
        except requests.exceptions.RequestException as e:
            warnings.warn(
                f"Failed to estimate free transfers from the API:\n{e}\n"
                "Will estimate from the DB instead, which may be out of date.",
                stacklevel=2,
            )

    # historical/simulated data or API failed - fetch from database
    transactions = (
        dbsession.query(Transaction)
        .order_by(Transaction.gameweek, Transaction.id)
        .filter_by(fpl_team_id=fpl_team_id)
        .filter_by(bought_or_sold=1)
        .all()
    )
    if len(transactions) == 0:
        return 1
    starting_gw = transactions[0].gameweek
    gw_transactions = {}
    for t in transactions:
        if t.gameweek not in gw_transactions:
            gw_transactions[t.gameweek] = 0
        gw_transactions[t.gameweek] += 1
    num_free_transfers = 1
    if gameweek is None and (season != CURRENT_SEASON or is_replay):
        msg = "Gameweek must be specified for historical data"
        raise ValueError(msg)
    gameweek = gameweek or NEXT_GAMEWEEK
    for prev_gw in range(starting_gw + 1, gameweek):
        if prev_gw not in gw_transactions:
            num_free_transfers = 2
        elif gw_transactions[prev_gw] >= 2:
            num_free_transfers = 1

    return num_free_transfers


@lru_cache(maxsize=365)
def get_gameweek_by_date(
    check_date: date | datetime,
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> int | None:
    """
    Gameweek of the next fixture on or after the specified date.
    """
    # convert date to a datetime object if it isn't already one.
    if not dbsession:
        dbsession = session
    check_date = parse_date(check_date)

    fixtures = (
        dbsession.query(Fixture)
        .filter_by(season=season)
        .filter(Fixture.date.isnot(None))
        .order_by(Fixture.date)
    ).all()

    for fixture in fixtures:
        if not fixture.date:
            # to keep mypy happy
            continue
        fixture_date = parse_date(fixture.date)
        if fixture_date >= check_date:
            return fixture.gameweek
    return None


def get_team_name(
    team_id: int, season: str = CURRENT_SEASON, dbsession: Session | None = None
) -> str | None:
    """
    Return 3-letter team name given a numerical id.
    These ids are based on alphabetical order of all teams in that season,
    so can vary from season to season.
    """
    if not dbsession:
        dbsession = session
    team = dbsession.query(Team).filter_by(season=season, team_id=team_id).first()
    if team:
        return team.name
    print(f"Unknown team_id {team_id} for {season} season")
    return None


def get_player(
    player_name_or_id: str | int,
    dbsession: Session | None = None,
) -> Player | None:
    """
    Query the player table by name or id, return the player object (or None).
    NOTE the player_id that can be passed as an argument here is NOT
    guaranteed to be the id for that player in the FPL API. The one here
    is the entry (primary key) in our database.
    Use the function get_player_from_api_id() to find the player corresponding
    to the FPL API ID.
    """
    if not dbsession:
        dbsession = session

    # ID field match
    if isinstance(player_name_or_id, str) and player_name_or_id.isdigit():
        player_name_or_id = int(player_name_or_id)

    if isinstance(player_name_or_id, int):
        if p := dbsession.query(Player).filter_by(player_id=player_name_or_id).first():
            return p
        # failed to find player by ID
        return None

    # String field matches
    if p := dbsession.query(Player).filter_by(name=player_name_or_id).first():
        return p

    if (
        mapping := dbsession.query(PlayerMapping)
        .filter_by(alt_name=player_name_or_id)
        .first()
    ):
        return dbsession.query(Player).filter_by(player_id=mapping.player_id).first()

    if p := dbsession.query(Player).filter_by(display_name=player_name_or_id).first():
        return p

    # No match found
    return None


def get_player_from_api_id(
    api_id: int, dbsession: Session | None = None
) -> Player | None:
    """
    Query the database and return the player with corresponding attribute fpl_api_id.
    """
    if not dbsession:
        dbsession = session
    if p := dbsession.query(Player).filter_by(fpl_api_id=api_id).first():
        return p
    print(f"Unable to find player with fpl_api_id {api_id}")
    return None


def get_player_name(player_id: int, dbsession: Session = session) -> str | None:
    """
    Lookup player name, for human readability.
    """
    if p := get_player(player_id, dbsession):
        return str(p)
    print(f"Unknown player_id {player_id}")
    return None


def get_player_id(player_name: str, dbsession: Session = session) -> int | None:
    if p := get_player(player_name, dbsession):
        return p.player_id
    print(f"Unknown player_name {player_name}")
    return None


def list_teams(
    season: str = CURRENT_SEASON, dbsession: Session = session
) -> list[dict[str, str]]:
    """
    Print all teams from current season.
    """
    rows = dbsession.query(Team).filter_by(season=season).all()
    return [{"name": row.name, "full_name": row.full_name} for row in rows]


def list_players(
    position: str = "all",
    team: str = "all",
    order_by: str = "price",
    season: str = CURRENT_SEASON,
    gameweek: int = NEXT_GAMEWEEK,
    dbsession: Session | None = None,
    verbose: bool = False,
) -> list[Player]:
    """
    Print list of players and return a list of player_ids.
    """
    if not dbsession:
        dbsession = session
    # if trying to get players from after DB has filled, return most recent players
    if season == CURRENT_SEASON:
        last_pa = (
            dbsession.query(PlayerAttributes)
            .filter_by(season=season)
            .order_by(PlayerAttributes.gameweek.desc())
            .first()
        )
        if last_pa and gameweek > last_pa.gameweek:
            if verbose:
                print(
                    f"WARNING: Incomplete data in DB for GW{gameweek}, "
                    f"returning players from GW{last_pa.gameweek}."
                )
            gameweek = last_pa.gameweek

    gameweeks = [gameweek]
    # check if the team (or all teams) play in the specified gameweek, if not
    # attributes might be missing
    fixtures = get_fixture_teams(
        get_fixtures_for_gameweek(gameweek, season=season, dbsession=dbsession)
    )
    teams_with_fixture = {t for fixture in fixtures for t in fixture}

    if (team == "all" and len(teams_with_fixture) < 20) or (
        team != "all" and team not in teams_with_fixture
    ):
        # check neighbouring gameweeks to get all 20 teams/specified team
        gws_to_try = [gameweek - 1, gameweek + 1, gameweek - 2, gameweek + 2]
        max_gw = get_max_gameweek(season, dbsession)
        gws_to_try = [gw for gw in gws_to_try if gw > 0 and gw <= max_gw]

        for gw in gws_to_try:
            fixtures = get_fixture_teams(
                get_fixtures_for_gameweek(gw, season=season, dbsession=dbsession)
            )
            new_teams = [t for fixture in fixtures for t in fixture]

            if team == "all" and any(t not in teams_with_fixture for t in new_teams):
                # this gameweek has some teams we haven't seen before
                gameweeks.append(gw)
                for t in new_teams:
                    teams_with_fixture.add(t)
                if len(teams_with_fixture) == 20:
                    break

            elif team != "all" and team in new_teams:
                # this gameweek has the team we're looking for
                gameweeks.append(gw)
                break

    query = (
        dbsession.query(PlayerAttributes)
        .filter_by(season=season)
        .filter(PlayerAttributes.gameweek.in_(gameweeks))
    )
    if team != "all":
        query = query.filter_by(team=team)
    if position != "all":
        query = query.filter_by(position=position)
    else:
        # exclude managers
        query = query.filter(PlayerAttributes.position != "MNG")
    if len(gameweeks) > 1:
        # Sort query results by order of gameweeks - i.e. make sure the input
        # query gameweek comes first.
        _whens = {gw: i for i, gw in enumerate(gameweeks)}
        sort_order = case(_whens, value=PlayerAttributes.gameweek)
        query = query.order_by(sort_order)
    if order_by == "price":
        query = query.order_by(PlayerAttributes.price.desc())
    players = []
    prices = []
    for pa in query.all():
        if pa.player not in players:
            # might have queried multiple gameweeks with same player returned
            # multiple times - only add if it's a new player
            players.append(pa.player)
            prices.append(pa.price)
            if verbose and (len(gameweeks) == 1 or order_by != "price"):
                print(pa.player, pa.team, pa.position, pa.price)
    if len(gameweeks) > 1 and order_by == "price":
        # Query sorted by gameweek first, so need to do a final sort here to
        # get final price order if more than one gameweek queried.
        sort_players = sorted(
            zip(prices, players, strict=False), reverse=True, key=lambda p: p[0]
        )
        if verbose:
            for pa in sort_players:
                print(pa[1], pa[0])
        players = [p for _, p in sort_players]
    return players


def is_future_gameweek(
    season: str,
    gameweek: int | None,
    current_season: str = CURRENT_SEASON,
    next_gameweek: int = NEXT_GAMEWEEK,
) -> bool:
    """
    Return True is season and gameweek refers to a gameweek that is after
    (or the same) as current_season and next_gameweek.
    """
    return (
        season == current_season and (gameweek is None or gameweek >= next_gameweek)
    ) or (season != current_season and int(season) > int(current_season))


def get_max_matches_per_player(
    position: str = "all",
    season: str = CURRENT_SEASON,
    gameweek: int = NEXT_GAMEWEEK,
    dbsession: Session | None = None,
) -> int:
    """
    Can be used e.g. in bpl_interface.get_player_history_df
    to help avoid a ragged dataframe.
    """
    players = list_players(
        position=position, season=season, gameweek=gameweek, dbsession=dbsession
    )
    max_matches = 0
    for p in players:
        num_match = sum(
            not is_future_gameweek(
                score.fixture.season,
                score.fixture.gameweek,
                current_season=season,
                next_gameweek=gameweek,
            )
            for score in p.scores
        )
        if num_match > max_matches:
            max_matches = num_match
    return max_matches


def get_player_attributes(
    player_name_or_id: str | int,
    season: str = CURRENT_SEASON,
    gameweek: int = NEXT_GAMEWEEK,
    dbsession: Session | None = None,
) -> PlayerAttributes | None:
    """
    Get a player's attributes for a given gameweek in a given season.
    """
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
    return (
        dbsession.query(PlayerAttributes)
        .filter_by(season=season)
        .filter_by(gameweek=gameweek)
        .filter_by(player_id=player_id)
        .first()
    )


def get_fixtures_for_player(
    player: Player | str | int,
    season: str = CURRENT_SEASON,
    gw_range: list[int] | None = None,
    dbsession: Session | None = None,
    verbose: bool = False,
) -> list[Fixture]:
    """
    Search for upcoming fixtures for a player, specified either by id or name.
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
        print(f"Couldn't find {player} in database")
        return []
    if not gw_range and season != CURRENT_SEASON:
        msg = "Gameweek range must be specified for past seasons"
        raise ValueError(msg)
    if not gw_range:
        team = player_record.team(season, NEXT_GAMEWEEK)
    else:
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
                print(fixture)
            fixtures.append(fixture)
    return fixtures


def get_next_fixture_for_player(
    player: Player | str | int,
    season: str = CURRENT_SEASON,
    gameweek: int = NEXT_GAMEWEEK,
    dbsession: Session | None = None,
) -> str:
    """
    Get a players next fixture as a string, for easy displaying.
    """
    if not dbsession:
        dbsession = session
    # given a player name or id, convert to player object
    if isinstance(player, str | int):
        maybe_player = get_player(player, dbsession)
        if not maybe_player:
            print(f"Couldn't find player {player} in database")
            return ""
        player = maybe_player
    team = player.team(season, gameweek)
    fixtures_for_player = get_fixtures_for_player(player, season, [gameweek], dbsession)
    output_string = ""
    for fixture in fixtures_for_player:
        if fixture.home_team == team:
            output_string += fixture.away_team + " (h)"
        else:
            output_string += fixture.home_team + " (a)"
        output_string += ", "
    return output_string[:-2]


def get_fixtures_for_season(
    season: str = CURRENT_SEASON, dbsession: Session = session
) -> list[Fixture]:
    """
    Return all fixtures for a season.
    """
    return dbsession.query(Fixture).filter_by(season=season).all()


def get_fixtures_for_gameweek(
    gameweek: list[int] | int,
    season: str = CURRENT_SEASON,
    dbsession: Session = session,
) -> list[Fixture]:
    """
    Get a list of fixtures for the specified gameweek(s).
    """
    if isinstance(gameweek, int):
        gameweek = [gameweek]
    return (
        dbsession.query(Fixture)
        .filter_by(season=season)
        .filter(Fixture.gameweek.in_(gameweek))
        .all()
    )


def get_fixture_teams(fixtures: Iterable[Fixture]) -> list[tuple[str, str]]:
    """
    Get (home_team, away_team) tuples for each fixture in a list of fixtures.
    """
    return [(fixture.home_team, fixture.away_team) for fixture in fixtures]


def get_player_scores(
    fixture: Fixture | None = None,
    player: Player | None = None,
    dbsession: Session = session,
) -> list[PlayerScore] | PlayerScore | None:
    """
    Get player scores for a fixture.
    """
    if fixture is None and player is None:
        msg = "At least one of fixture and player must be defined"
        raise ValueError(msg)

    query = dbsession.query(PlayerScore)
    if fixture is not None:
        query = query.filter(PlayerScore.fixture == fixture)
    if player is not None:
        query = query.filter(PlayerScore.player == player)

    player_scores = query.all()
    if not player_scores:
        return None

    if fixture is not None and player is not None:
        if len(player_scores) > 1:
            msg = f"More than one score found for player {player} in fixture {fixture}"
            raise ValueError(msg)
        return player_scores[0]
    return player_scores


def get_players_for_gameweek(
    gameweek: int,
    fpl_team_id: int | None = None,
    apifetcher: FPLDataFetcher = fetcher,
) -> list[Player]:
    """
    Use FPL API to get the players for a given gameweek.
    """
    if not fpl_team_id:
        fpl_team_id = apifetcher.FPL_TEAM_ID

    player_data = apifetcher.get_fpl_team_data(gameweek, fpl_team_id)["picks"]
    player_api_id_list = [p["element"] for p in player_data]
    players: list[Player] = []
    for api_id in player_api_id_list:
        player = get_player_from_api_id(api_id)
        if player is None:
            print(f"Unable to find player with fpl_api_id {api_id}")
            continue
        players.append(player)
    return players


def get_previous_points_for_same_fixture(
    player: str | int, fixture_id: int, dbsession: Session = session
) -> dict[str, int]:
    """
    Search the past matches for same fixture in past seasons,
    and how many points the player got.
    """
    if isinstance(player, str):
        player_record = dbsession.query(Player).filter_by(name=player).first()
        if not player_record:
            print(f"Can't find player {player}")
            return {}
        player_id = player_record.player_id
    else:
        player_id = player
    fixture = dbsession.query(Fixture).filter_by(fixture_id=fixture_id).first()
    if not fixture:
        print(f"Couldn't find fixture_id {fixture_id}")
        return {}
    home_team = fixture.home_team
    away_team = fixture.away_team

    previous_matches = (
        dbsession.query(Fixture)
        .filter_by(home_team=home_team)
        .filter_by(away_team=away_team)
        .order_by(Fixture.season)
        .all()
    )
    fixture_ids = [(f.fixture_id, f.season) for f in previous_matches]
    previous_points = {}
    for fid in fixture_ids:
        scores = (
            dbsession.query(PlayerScore)
            .filter_by(player_id=player_id, fixture_id=fid[0])
            .all()
        )
        for s in scores:
            previous_points[fid[1]] = s.points

    return previous_points


@lru_cache(maxsize=4096)
def get_predicted_points_for_player(
    player: Player | str | int,
    tag: str,
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> dict[int, float]:
    """
    Query the player prediction table for a given player.
    Return a dict, keyed by gameweek.
    """
    if not dbsession:
        dbsession = session
    if isinstance(player, str | int):
        maybe_player = get_player(player, dbsession=dbsession)
        if maybe_player is None:
            msg = f"Couldn't find player {player} in database"
            raise ValueError(msg)
        player = maybe_player

    pps = (
        dbsession.query(PlayerPrediction)
        .filter(PlayerPrediction.fixture.has(Fixture.season == season))
        .filter_by(player_id=player.player_id, tag=tag)
        .all()
    )
    ppdict = {}
    for prediction in pps:
        # there is one prediction per fixture.
        # for double gameweeks, we need to add the two together
        gameweek = prediction.fixture.gameweek
        if gameweek is None:
            print(f"Player {player} has no gameweek for fixture {prediction.fixture}")
            continue
        if gameweek not in ppdict:
            ppdict[gameweek] = 0.0
        ppdict[gameweek] += prediction.predicted_points
    # we still need to fill in zero for gameweeks that they're not playing.
    max_gw = get_max_gameweek(season, dbsession)
    for gw in range(1, max_gw + 1):
        if gw not in ppdict:
            ppdict[gw] = 0.0
    return ppdict


def get_predicted_points(
    gameweek: int | list[int],
    tag: str,
    position: str = "all",
    team: str = "all",
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> list[tuple[Player, float]]:
    """
    Query the player_prediction table with selections, return
    list of tuples (player_id, predicted_points) ordered by predicted_points
    "gameweek" argument can either be a single integer for one gameweek, or a
    list of gameweeks, in which case we will get the sum over all of them.
    """
    if isinstance(gameweek, int):  # predictions for a single gameweek
        players = list_players(
            position, team, season=season, gameweek=gameweek, dbsession=dbsession
        )
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
        players = list_players(
            position, team, season=season, gameweek=gameweek[0], dbsession=dbsession
        )
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
    gameweek: int | list[int] | None = None,
    tag: str | None = None,
    position: str = "all",
    team: str = "all",
    n_players: int = 10,
    per_position: bool = False,
    max_price: float | None = None,
    season: str = CURRENT_SEASON,
    dbsession: Session = session,
) -> None:
    """
    Print players with the top predicted points.

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
    discord_webhook = fetcher.DISCORD_WEBHOOK
    if not tag:
        tag = get_latest_prediction_tag()
    if not gameweek:
        gameweek = NEXT_GAMEWEEK

    discord_embed = {
        "title": "AIrsenal webhook",
        "description": f"PREDICTED TOP {n_players} PLAYERS FOR GAMEWEEK(S) {gameweek}:",
        "color": 0x35A800,
        "fields": [],
    }

    first_gw = gameweek[0] if isinstance(gameweek, list) else gameweek
    print("=" * 50)
    print(f"PREDICTED TOP {n_players} PLAYERS FOR GAMEWEEK(S) {gameweek}:")
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
            for p in pts:
                price = p[0].price(season, first_gw)
                if price is not None and price > max_price:
                    pts.remove(p)

        pts = sorted(pts, key=lambda x: x[1], reverse=True)

        for i, p in enumerate(pts[:n_players]):
            price = p[0].price(season, first_gw)
            price_str = str(price / 10) if price is not None else "UNKNOWN_PRICE"
            print(
                f"{i + 1}. {p[0]}, {p[1]:.2f}pts "
                f"(£{price_str}m, {p[0].position(season)}, "
                f"{p[0].team(season, first_gw)})"
            )

        # If a valid discord webhook URL has been stored
        # in env variables, send a webhook message
        if discord_webhook:
            # Use regex to check the discord webhook url is correctly formatted
            if re.match(
                r"^.*(discord|discordapp)\.com\/api"
                r"\/webhooks\/([\d]+)\/([a-zA-Z0-9_-]+)$",
                discord_webhook,
            ):
                # Maximum fields on a discord embed is 25, so limit this to n_players=8
                payload = predicted_points_discord_payload(
                    discord_embed=discord_embed,
                    position=position,
                    pts=pts[: min(n_players, 8)],
                    season=season,
                    first_gw=first_gw,
                )
                result = requests.post(discord_webhook, json=payload)
                if 200 <= result.status_code < 300:
                    print(f"Discord webhook sent, status code: {result.status_code}")
                else:
                    print(
                        f"Not sent with {result.status_code},"
                        "response:\n{result.json()}"
                    )
            else:
                print("Warning: Discord webhook url is malformed!\n", discord_webhook)
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
                for p in pts:
                    maybe_price = p[0].price(season, first_gw)
                    if maybe_price is not None and maybe_price > max_price:
                        pts.remove(p)

            pts = sorted(pts, key=lambda x: x[1], reverse=True)
            print(f"{position}:")

            for i, p in enumerate(pts[:n_players]):
                maybe_price = p[0].price(season, first_gw)
                price_str = (
                    str(maybe_price / 10)
                    if maybe_price is not None
                    else "UNKNOWN_PRICE"
                )
                print(
                    f"{i + 1}. {p[0]}, {p[1]:.2f}pts "
                    f"(£{price_str}m, "
                    f"{p[0].team(season, first_gw)})"
                )
            print("-" * 25)

            discord_embed["fields"] = []
            # If a valid discord webhook URL has been stored
            # in env variables, send a webhook message
            if discord_webhook is not None:
                # Use regex to check the discord webhook url is correctly formatted
                if re.match(
                    r"^.*(discord|discordapp)\.com\/api"
                    r"\/webhooks\/([\d]+)\/([a-zA-Z0-9_-]+)$",
                    discord_webhook,
                ):
                    # create a formatted team lineup message for the discord webhook
                    # Maximum fields on a discord embed is 25
                    # limit this to n_players=8
                    payload = predicted_points_discord_payload(
                        discord_embed=discord_embed,
                        position=position,
                        pts=pts[: min(n_players, 8)],
                        season=season,
                        first_gw=first_gw,
                    )
                    result = requests.post(discord_webhook, json=payload)
                    if 200 <= result.status_code < 300:
                        print(
                            f"Discord webhook sent, status code: {result.status_code}"
                        )
                    else:
                        print(
                            f"Not sent with {result.status_code}, "
                            f"response:\n{result.json()}"
                        )
                else:
                    print(
                        "Warning: Discord webhook url is malformed!\n", discord_webhook
                    )


def predicted_points_discord_payload(
    discord_embed: dict,
    position: str,
    pts: list[tuple[Player, float]],
    season: str,
    first_gw: int,
) -> dict:
    """
    json formated discord webhook contentent.
    """
    discord_embed["fields"].append(
        {"name": "Position", "value": str(position), "inline": False}
    )
    for i, p in enumerate(pts):
        price = p[0].price(season, first_gw)
        price_str = str(price / 10) if price is not None else "UNKNOWN_PRICE"
        discord_embed["fields"].extend(
            [
                {
                    "name": "Player",
                    "value": f"{i + 1}. {p[0]}",
                    "inline": True,
                },
                {
                    "name": "Predicted points",
                    "value": f"{p[1]:.2f}pts",
                    "inline": True,
                },
                {
                    "name": "Attributes",
                    "value": (
                        f"£{price_str}m, "
                        f"{p[0].position(season)}, {p[0].team(season, first_gw)}"
                    ),
                    "inline": True,
                },
            ]
        )
    return {
        "content": "",
        "username": "AIrsenal",
        "embeds": [discord_embed],
    }


def get_return_gameweek_from_news(
    news: str, team: str, season: str = CURRENT_SEASON, dbsession: Session = session
) -> int | None:
    """
    Parse news strings from the FPL API for the return date of injured or
    suspended players. If a date is found, determine and return the gameweek it
    corresponds to.
    """
    rd_rex = "(Expected back|Suspended until)[\\s]+([\\d]+[\\s][\\w]{3})"
    search_results = re.search(rd_rex, news)
    if not search_results:
        return None

    return_str = search_results.groups()[1]
    # return_str should be a day and month string (without year)

    # create a date in the future from the day and month string
    return_date = dateparser.parse(return_str, settings={"PREFER_DATES_FROM": "future"})
    if not return_date:
        msg = f"Failed to parse date from string '{return_date}'"
        raise ValueError(msg)

    return get_return_gameweek_by_date(
        return_date.date(), team=team, season=season, dbsession=dbsession
    )


def calc_average_minutes(player_scores: list[PlayerScore]) -> float:
    """
    Simple average of minutes played for a list of PlayerScore objects.
    """
    total = 0.0
    for ps in player_scores:
        total += ps.minutes
    return total / len(player_scores)


def estimate_minutes_from_prev_season(
    player: Player,
    season: str = CURRENT_SEASON,
    gameweek: int = NEXT_GAMEWEEK,
    n_games_to_use: int = 10,
    exclude_unavailable: bool = True,
    current_team_only: bool = True,
    dbsession: Session | None = None,
) -> list[float]:
    """
    Take average of minutes from previous season if any, or else return [0]
    """
    if not dbsession:
        dbsession = session
    previous_season = get_previous_season(season)

    # Only consider minutes the player played with his current team
    current_team = player.team(season, gameweek)
    query = (
        dbsession.query(PlayerScore)
        .filter_by(player_id=player.player_id)
        .filter(PlayerScore.fixture.has(season=previous_season))
        .join(Fixture, PlayerScore.fixture)
    )

    if current_team_only:
        current_team = player.team(season, gameweek)
        query = query.filter(PlayerScore.player_team == current_team)

    if exclude_unavailable:
        query = query.filter(
            or_(
                PlayerScore.minutes >= 60,
                PlayerScore.chance_of_playing == 100,
                PlayerScore.chance_of_playing.is_(None),  # for backwards compatibility
            )
        )

    player_scores = query.order_by(Fixture.gameweek.desc()).limit(n_games_to_use).all()

    if len(player_scores) == 0:
        # no FPL history / didn't play for current team last season
        return [0]

    # Return average minutes. A weakness of this is increased rotation at the end of the
    # season when teams don't have anything to play for.
    return [calc_average_minutes(player_scores)]


def get_recent_playerscore_rows(
    player: Player,
    num_match_to_use: int = 3,
    season: str = CURRENT_SEASON,
    last_gw: int | None = None,
    exclude_unavailable: bool = False,
    current_team_only: bool = False,
    dbsession: Session | None = None,
) -> list[PlayerScore]:
    """
    Query the playerscore table in the database to retrieve
    the last num_match_to_use rows for this player.
    """
    if not dbsession:
        dbsession = session
    # If asking for gameweeks without results in DB, revert to most recent results.
    last_available_gameweek = get_last_complete_gameweek_in_db(
        season=season, dbsession=dbsession
    )
    if not last_available_gameweek:
        # e.g. before this season has started
        return []

    if last_gw is None and season != CURRENT_SEASON:
        msg = "last_gw must be specified is running on previous seasons"
        raise ValueError(msg)

    if last_gw is None or last_gw > last_available_gameweek:
        last_gw = last_available_gameweek

    # get the playerscore rows from the db
    query = (
        dbsession.query(PlayerScore)
        .join(Fixture, PlayerScore.fixture_id == Fixture.fixture_id)
        .filter(Fixture.season == season)
        .filter(PlayerScore.player_id == player.player_id)
        .filter(PlayerScore.fixture.has(Fixture.gameweek <= last_gw))
    )
    if exclude_unavailable:
        # minutes at least 60 or no flag status (100% chance of playing)
        query = query.filter(
            or_(
                PlayerScore.minutes >= 60,
                PlayerScore.chance_of_playing == 100,
                PlayerScore.chance_of_playing.is_(None),  # for backwards compatibility
            )
        )
    if current_team_only:
        team = player.team(season, last_gw)
        query = query.filter(PlayerScore.player_team == team)

    return query.order_by(Fixture.gameweek.desc()).limit(num_match_to_use).all()


def get_playerscores_for_player_gameweek(
    player: Player,
    gameweek: int,
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> list[PlayerScore]:
    """
    FPL points for this player for selected match.
    Returns a PlayerScore object.
    """
    if not dbsession:
        dbsession = session
    return (
        dbsession.query(PlayerScore)
        .filter(PlayerScore.fixture.has(season=season))
        .filter_by(player_id=player.player_id)
        .filter(PlayerScore.fixture.has(Fixture.gameweek == gameweek))
        .all()
    )


def get_recent_scores_for_player(
    player: Player,
    num_match_to_use: int = 3,
    season: str = CURRENT_SEASON,
    last_gw: int | None = None,
    exclude_unavailable: bool = False,
    current_team_only: bool = False,
    dbsession: Session | None = None,
) -> dict[int, int]:
    """
    Look num_match_to_use matches back, and return the
    FPL points for this player for each of these matches.
    Return a dict {gameweek: score, }
    """
    if last_gw is None:
        if season != CURRENT_SEASON:
            msg = "last_gw must be specified if running on previous seasons"
            raise ValueError(msg)
        last_gw = NEXT_GAMEWEEK
    first_gw = last_gw - num_match_to_use

    playerscores = get_recent_playerscore_rows(
        player,
        num_match_to_use,
        season,
        last_gw,
        exclude_unavailable,
        current_team_only,
        dbsession,
    )
    if not playerscores:  # e.g. start of season
        return {}

    return {range(first_gw, last_gw)[i]: ps.points for i, ps in enumerate(playerscores)}


def get_recent_minutes_for_player(
    player: Player,
    num_match_to_use: int = 3,
    season: str = CURRENT_SEASON,
    last_gw: int | None = None,
    exclude_unavailable: bool = True,
    current_team_only: bool = True,
    dbsession: Session = session,
) -> list[float]:
    """
    Look back num_match_to_use matches, and return an array
    containing minutes played in each.
    If current_gw is not given, we take it to be the most
    recent finished gameweek.
    """
    if last_gw is None:
        if season != CURRENT_SEASON:
            msg = "last_gw must be defined if running on previous seasons"
            raise ValueError(msg)
        last_gw = NEXT_GAMEWEEK

    playerscores = (
        get_recent_playerscore_rows(
            player,
            num_match_to_use,
            season,
            last_gw,
            exclude_unavailable,
            current_team_only,
            dbsession,
        )
        or []
    )

    minutes = [float(r.minutes) for r in playerscores]

    if len(minutes) < num_match_to_use:
        minutes += estimate_minutes_from_prev_season(
            player, season, gameweek=last_gw, dbsession=dbsession
        )
    return minutes or [0.0]


def was_historic_absence(
    player: Player, gameweek: int, season: str, dbsession: Session | None = None
) -> bool:
    """
    For past seasons, query the Absence table for a given player and season,
    and see if the gameweek is within the period of the absence.

    Returns: bool, True if player was absent (injured or suspended), False otherwise.
    """
    if season == CURRENT_SEASON:
        # we only consider past seasons here
        return False
    if not dbsession:
        dbsession = session
    absence = (
        dbsession.query(Absence)
        .filter_by(season=season)
        .filter_by(player=player)
        .filter(Absence.gw_from < gameweek)
        .filter(Absence.gw_until > gameweek)
        .first()
    )
    return bool(absence)


def get_last_complete_gameweek_in_db(
    season: str = CURRENT_SEASON, dbsession: Session | None = None
) -> int | None:
    """
    Query the result table to see what was the last gameweek for which
    we have filled the data.
    """
    if not dbsession:
        dbsession = session
    first_missing = (
        dbsession.query(Fixture)
        .filter_by(season=season)
        .filter(Fixture.result == None)  # noqa: E711
        .filter(Fixture.gameweek != None)  # noqa: E711
        .order_by(Fixture.gameweek)
        .first()
    )
    if first_missing is not None and first_missing.gameweek is not None:
        return first_missing.gameweek - 1
    if season == CURRENT_SEASON:
        return None
    return get_max_gameweek(season=season, dbsession=dbsession)


def get_last_finished_gameweek() -> int:
    """
    Query the API to see what the last gameweek marked as 'finished' is.
    """
    event_data = fetcher.get_event_data()
    last_finished = 0
    for gw in sorted(event_data.keys()):
        if event_data[gw]["is_finished"]:
            last_finished = gw
        else:
            return last_finished
    return last_finished


def get_latest_prediction_tag(
    season: str = CURRENT_SEASON,
    tag_prefix: str = "",
    dbsession: Session | None = None,
) -> str:
    """
    Query the predicted_score table and get the tag field for the last row.
    """
    if not dbsession:
        dbsession = session
    rows = (
        dbsession.query(PlayerPrediction)
        .filter(PlayerPrediction.fixture.has(Fixture.season == season))
        .all()
    )
    if len(rows) == 0:
        msg = (
            "No predicted points in database - has the database been filled?\n"
            "To calculate points predictions (and fill the database) use "
            "'airsenal_run_prediction'. This should be done before using "
            "'airsenal_make_squad' or 'airsenal_run_optimization'."
        )
        raise RuntimeError(msg)
    if tag_prefix:
        rows = [r for r in rows if r.tag.startswith(tag_prefix)]
    return rows[-1].tag


def get_latest_fixture_tag(
    season: str = CURRENT_SEASON, dbsession: Session | None = None
) -> str:
    """
    Query the predicted_score table and get the tag field for the last row.
    """
    if not dbsession:
        dbsession = session
    rows = dbsession.query(Fixture).filter_by(season=season).all()
    return rows[-1].tag


def find_fixture(
    team: str | int,
    was_home: bool | None = None,
    other_team: str | int | None = None,
    gameweek: int | None = None,
    season: str = CURRENT_SEASON,
    kickoff_time: date | datetime | str | None = None,
    dbsession: Session = session,
) -> Fixture | None:
    """
    Get a fixture given a team and optionally whether the team was at home or away,
    the season, kickoff time and the other team in the fixture. Only returns the fixture
    if exactly one match is found, otherwise raises a ValueError.
    """
    if not isinstance(team, str):
        team_name = get_team_name(team, season=season, dbsession=dbsession)
    else:
        team_name = team

    if not team_name:
        msg = f"No team with id {team} in {season} season"
        raise ValueError(msg)

    if isinstance(other_team, int):
        other_team_name = get_team_name(other_team, season=season, dbsession=dbsession)
    else:
        other_team_name = other_team

    query = dbsession.query(Fixture).filter_by(season=season)
    if gameweek:
        query = query.filter_by(gameweek=gameweek)
    if was_home is True:
        query = query.filter_by(home_team=team_name)
    elif was_home is False:
        query = query.filter_by(away_team=team_name)
    else:
        query = query.filter(
            or_(Fixture.away_team == team_name, Fixture.home_team == team_name)
        )

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
        print(
            f"No fixture with season={season}, gw={gameweek}, "
            f"team_name={team_name}, was_home={was_home}, "
            f"other_team_name={other_team_name}, kickoff_time={kickoff_time}"
        )
        return None

    if len(fixtures) == 1:
        return fixtures[0]
    if kickoff_time:
        # team played multiple games in the gameweek, determine the
        # fixture of interest using the kickoff time,
        kickoff_date = parse_date(kickoff_time)

        for f in fixtures:
            f_date = parse_date(f.date)
            if f_date == kickoff_date:
                return f

    print(
        f"No unique fixture with season={season}, gw={gameweek}, "
        f"team_name={team_name}, was_home={was_home}, "
        f"kickoff_time={kickoff_time}"
    )
    return None


def get_player_team_from_fixture(
    fixture: Fixture,
    opponent: str | int | None = None,
    player_at_home: bool | None = None,
    season: str = CURRENT_SEASON,
    dbsession: Session = session,
) -> str:
    """
    Get the team a player played for given the gameweek, opponent, time and
    whether they were home or away.
    If return_fixture is True, return a tuple of (team_name, fixture).
    """
    if opponent is None and player_at_home is None:
        msg = "Either opponent or player_at_home must be specified"
        raise ValueError(msg)

    if player_at_home is not None:
        return fixture.home_team if player_at_home else fixture.away_team

    if isinstance(opponent, int):
        opponent_name = get_team_name(opponent, season=season, dbsession=dbsession)
    else:
        opponent_name = opponent

    if fixture.home_team == opponent_name:
        return fixture.away_team
    if fixture.away_team == opponent_name:
        return fixture.home_team

    msg = f"Opponent {opponent_name} not in fixture"
    raise ValueError(msg)


def is_transfer_deadline_today() -> bool:
    """
    Return True if there is a transfer deadline later today.
    """
    deadlines = fetcher.get_transfer_deadlines()
    for deadline in deadlines:
        deadline_datetime = datetime.strptime(deadline, "%Y-%m-%dT%H:%M:%SZ")
        if (deadline_datetime - datetime.now()).days == 0:
            return True
    return False


T = TypeVar("T")


def fastcopy(obj: T) -> T:
    """
    Faster replacement for copy.deepcopy().
    """
    return loads(dumps(obj, -1))


def parse_team_model_from_str(
    team_model: str,
) -> (
    RandomMatchPredictor
    | ExtendedDixonColesMatchPredictor
    | NeutralDixonColesMatchPredictor
):
    """
    Returns the team model class corresponding to the given string.
    """
    if team_model == "random":
        return RandomMatchPredictor()
    if team_model == "extended":
        return ExtendedDixonColesMatchPredictor()
    if team_model == "neutral":
        return NeutralDixonColesMatchPredictor()
    msg = "Unknown team model"
    raise ValueError(msg)
