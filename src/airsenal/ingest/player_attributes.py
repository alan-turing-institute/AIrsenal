"""Fill the "player_attributes" table from this season's API and past seasons' files."""

import json
from typing import Any

import dateparser
import regex as re
from sqlalchemy.orm.session import Session

from airsenal.core.console import track
from airsenal.core.data_files import data_file
from airsenal.core.logging import get_logger
from airsenal.db.models import PlayerAttributes
from airsenal.db.queries.fixtures import find_fixture, get_player_team_from_fixture
from airsenal.db.queries.gameweeks import (
    get_return_gameweek_by_date,
    next_gameweek,
)
from airsenal.db.queries.players import (
    get_player,
    get_player_attributes,
    get_player_from_api_id,
)
from airsenal.db.queries.teams import get_team_name
from airsenal.db.session import get_session
from airsenal.game.mappings import positions
from airsenal.game.season import CURRENT_SEASON, get_past_seasons, sort_seasons
from airsenal.remote.fpl_api import get_fetcher

logger = get_logger(__name__)


def get_return_gameweek_from_news(
    news: str, team: str, season: str = CURRENT_SEASON, dbsession: Session | None = None
) -> int | None:
    """
    The gameweek a player flagged in the FPL API's news text is due back for.

    None if the news carries no parseable return date.
    """
    dbsession = dbsession if dbsession is not None else get_session()
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


def fill_attributes_table_from_file(
    detail_data: dict[str, Any], season: str, dbsession: Session | None = None
) -> None:
    """Fill the attributes table for a past season, from its player detail JSON."""
    dbsession = dbsession if dbsession is not None else get_session()
    for player_name_or_id, player_data in track(
        detail_data.items(), description=f"PLAYER ATTRIBUTES {season}"
    ):
        # find the player id in the player table.  If they're not
        # there, then we don't care (probably not a current player).
        player = get_player(player_name_or_id, dbsession=dbsession)
        if not player:
            logger.warning("Couldn't find player %s", player_name_or_id)
            continue

        # now loop through all the fixtures that player played in
        # Only one attributes row per gameweek - create list of gameweeks
        # encountered so can ignore duplicates (e.g. from double gameweeks).
        previous_gameweeks = []
        for fixture_data in player_data:
            gameweek = int(fixture_data["gameweek"])
            if gameweek in previous_gameweeks:
                # already done this gameweek
                continue
            previous_gameweeks.append(gameweek)

            pa = PlayerAttributes()
            pa.player = player
            pa.player_id = player.player_id
            pa.season = season
            pa.gameweek = gameweek
            # obtain attributes if available, otherwise set to None or default value
            pa.price = int(fixture_data.get("value", 0))
            pa.team = fixture_data.get("played_for", "")
            pa.position = fixture_data.get("position", "")
            pa.transfers_balance = (
                int(fixture_data.get("transfers_balance"))
                if fixture_data.get("transfers_balance") is not None
                else None
            )
            pa.selected = (
                int(fixture_data.get("selected"))
                if fixture_data.get("selected") is not None
                else None
            )
            pa.transfers_in = (
                int(fixture_data.get("transfers_in"))
                if fixture_data.get("transfers_in") is not None
                else None
            )
            pa.transfers_out = (
                int(fixture_data.get("transfers_out"))
                if fixture_data.get("transfers_out") is not None
                else None
            )
            dbsession.add(pa)
    dbsession.commit()


def fill_attributes_table_from_api(
    season: str, gw_start: int = 1, dbsession: Session | None = None
) -> None:
    """Fill the attributes table for the current season, from the FPL API."""
    dbsession = dbsession if dbsession is not None else get_session()
    fetcher = get_fetcher()
    next_gw = next_gameweek()

    # needed for selected by calculation from percentage below
    n_players = fetcher.get_current_summary_data()["total_players"]

    input_data = fetcher.get_player_summary_data()

    for player_api_id in track(input_data, description=f"PLAYER ATTRIBUTES {season}"):
        # find the player in the player table
        player = get_player_from_api_id(player_api_id, dbsession=dbsession)
        if not player:
            logger.warning(
                "ATTRIBUTES %s No player found with id %s", season, player_api_id
            )
            continue

        # First update the current gameweek using the summary data
        p_summary = input_data[player_api_id]

        if player.opta_code is None and "opta_code" in p_summary:
            player.opta_code = p_summary["opta_code"]

        position = positions[p_summary["element_type"]]

        pa = get_player_attributes(
            player.player_id, gameweek=next_gw, season=season, dbsession=dbsession
        )

        if pa:
            # found pre-existing attributes for this gameweek
            update = True
        else:
            # no attributes for this gameweek for this player yet
            pa = PlayerAttributes()
            update = False

        pa.player = player
        pa.player_id = player.player_id
        pa.season = season
        pa.gameweek = next_gw
        pa.price = int(p_summary["now_cost"])
        team = get_team_name(p_summary["team"], season=season, dbsession=dbsession)
        if team is None:
            logger.warning(
                "Couldn't find team %s for player %s", p_summary["team"], player
            )
            continue
        pa.team = team
        pa.position = positions[p_summary["element_type"]]
        pa.selected = int(float(p_summary["selected_by_percent"]) * n_players / 100)
        transfers_in = int(p_summary["transfers_in"])
        transfers_out = int(p_summary["transfers_out"])
        pa.transfers_in = transfers_in
        pa.transfers_out = transfers_out
        pa.transfers_balance = transfers_in - transfers_out
        pa.news = p_summary["news"]
        chance_of_playing_next_round = p_summary["chance_of_playing_next_round"]
        pa.chance_of_playing_next_round = chance_of_playing_next_round
        if (
            chance_of_playing_next_round is not None
            and chance_of_playing_next_round <= 50
        ):
            pa.return_gameweek = get_return_gameweek_from_news(
                p_summary["news"],
                team=team,
                season=season,
                dbsession=dbsession,
            )

        if not update:
            # only need to add to the dbsession for new entries, if we're doing
            # an update the final dbsession.commit() is enough
            dbsession.add(pa)

        # now get data for previous gameweeks
        if next_gw > 1:
            player_data = fetcher.get_gameweek_data_for_player(player_api_id)
            if not player_data:
                logger.warning("Failed to get data for %s", player)
                continue
            for gameweek, data in player_data.items():
                if gameweek < gw_start:
                    continue

                for result in data:
                    # check whether there are pre-existing attributes to update
                    pa = get_player_attributes(
                        player.player_id,
                        season=season,
                        gameweek=gameweek,
                        dbsession=dbsession,
                    )
                    if pa:
                        update = True
                    else:
                        pa = PlayerAttributes()
                        update = False

                    # determine the team the player played for in this fixture
                    opponent_id = result["opponent_team"]
                    was_home = result["was_home"]
                    kickoff_time = result["kickoff_time"]
                    fixture = find_fixture(
                        opponent_id,
                        was_home=not was_home,
                        gameweek=gameweek,
                        season=season,
                        kickoff_time=kickoff_time,
                        dbsession=dbsession,
                    )
                    if fixture is None:
                        logger.warning(
                            "Couldn't find fixture for %s vs %s in gameweek %s",
                            player,
                            opponent_id,
                            gameweek,
                        )
                        continue
                    team = get_player_team_from_fixture(
                        fixture,
                        opponent_id,
                        player_at_home=was_home,
                        season=season,
                        dbsession=dbsession,
                    )

                    pa.player = player
                    pa.player_id = player.player_id
                    pa.season = season
                    pa.gameweek = gameweek
                    pa.price = int(result["value"])
                    pa.team = team
                    pa.position = position  # does not change during season
                    pa.transfers_balance = int(result["transfers_balance"])
                    pa.selected = int(result["selected"])
                    pa.transfers_in = int(result["transfers_in"])
                    pa.transfers_out = int(result["transfers_out"])

                    if not update:
                        # don't need to add to dbsession if updating pre-existing row
                        dbsession.add(pa)

                    break  # done this gameweek now
    dbsession.commit()


def make_attributes_table(
    seasons: list[str] | None = None, dbsession: Session | None = None
) -> None:
    """Fill the attributes table: past seasons from JSON, this one from the API."""
    dbsession = dbsession if dbsession is not None else get_session()
    if seasons is None:
        seasons = []
    if not seasons:
        seasons = [CURRENT_SEASON]
        seasons += get_past_seasons(3)
    for season in sort_seasons(seasons):
        if season == CURRENT_SEASON:
            # current season - use API
            fill_attributes_table_from_api(season=CURRENT_SEASON, dbsession=dbsession)
        else:
            with data_file(f"player_details_{season}.json").open() as f:
                input_data = json.load(f)

            fill_attributes_table_from_file(
                detail_data=input_data, season=season, dbsession=dbsession
            )
    dbsession.commit()
