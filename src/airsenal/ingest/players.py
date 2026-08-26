"""Fill the "player" table from this season's FPL API and past seasons' files."""

import json

from sqlalchemy import select
from sqlalchemy.orm.session import Session

from airsenal.core.console import track
from airsenal.core.data_files import FilePath, data_file
from airsenal.db.models import Player, PlayerMapping
from airsenal.db.session import get_session
from airsenal.game.season import CURRENT_SEASON, get_past_seasons, sort_seasons
from airsenal.ingest.player_mappings import (
    add_mappings,
    make_player_mappings_table,
)
from airsenal.remote.fpl_api import FPLDataFetcher


def find_player_in_table(
    name: str, dbsession: Session, opta_code: str | None = None
) -> Player | None:
    """Find a player already in the table by opta code, name, or a known alias."""
    # look for an opta code match
    if opta_code and (
        player := dbsession.scalars(
            select(Player).where(Player.opta_code == opta_code).limit(1)
        ).first()
    ):
        return player

    # look for an exact name match
    if player := dbsession.scalars(
        select(Player).where(Player.name == name).limit(1)
    ).first():
        return player

    # look for an alternative name
    mapping = dbsession.scalars(
        select(PlayerMapping).where(PlayerMapping.alt_name == name).limit(1)
    ).first()
    if mapping:
        return dbsession.scalars(
            select(Player).where(Player.player_id == mapping.player_id).limit(1)
        ).first()

    return None


def fill_player_table_from_file(
    filename: FilePath, season: str, dbsession: Session
) -> None:
    """Add a season's players from its packaged JSON file."""
    with open(filename) as f:
        jplayers = json.load(f)
    for jp in track(jplayers, description=f"PLAYERS {season}"):
        new_entry = False
        name = jp["name"]
        opta_code = jp.get("opta_code")
        p = find_player_in_table(name, dbsession, opta_code=opta_code)
        if not p:
            new_entry = True
            p = Player()
            p.name = name
            p.opta_code = opta_code
        if new_entry:
            dbsession.add(p)
            dbsession.commit()
            add_mappings(p, dbsession=dbsession)
    dbsession.commit()


def fill_player_table_from_api(season: str, dbsession: Session) -> None:
    """Add the current season's players from the FPL API."""
    df = FPLDataFetcher()
    pd = df.get_player_summary_data()

    for k, v in track(pd.items(), description=f"PLAYERS {season}"):
        p = Player()
        p.fpl_api_id = k
        first_name = v["first_name"]  # .encode("utf-8")
        second_name = v["second_name"]  # .encode("utf-8")
        name = f"{first_name} {second_name}"
        display_name = v.get("web_name")

        p.name = name
        p.display_name = display_name
        p.opta_code = v["opta_code"]
        dbsession.add(p)
    dbsession.commit()


def make_init_player_table(season: str, dbsession: Session | None = None) -> None:
    """
    Add the most recent season's players.

    Done on its own, before the older seasons, so that a player's canonical row
    is the one with their current name and the rest map onto it.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if season == CURRENT_SEASON:
        # current season - use API
        fill_player_table_from_api(CURRENT_SEASON, dbsession)
    else:
        fill_player_table_from_file(
            data_file(f"player_summary_{season}.json"), season, dbsession
        )


def make_remaining_player_table(
    seasons: list[str] | None = None, dbsession: Session | None = None
) -> None:
    """Add players from older seasons who are not in the most recent one already."""
    dbsession = dbsession if dbsession is not None else get_session()
    if seasons is None:
        seasons = []
    for season in seasons:
        fill_player_table_from_file(
            data_file(f"player_summary_{season}.json"), season, dbsession
        )


def make_player_table(
    seasons: list[str] | None = None, dbsession: Session | None = None
) -> None:
    dbsession = dbsession if dbsession is not None else get_session()
    if seasons is None:
        seasons = []
    if not seasons:
        seasons = [CURRENT_SEASON]
        seasons += get_past_seasons(3)
    seasons = sort_seasons(seasons)
    make_init_player_table(season=seasons[0], dbsession=dbsession)
    make_player_mappings_table(dbsession=dbsession)
    make_remaining_player_table(seasons=seasons[1:], dbsession=dbsession)
