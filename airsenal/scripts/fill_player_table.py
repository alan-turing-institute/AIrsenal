#!/usr/bin/env python

"""
Fill the "Player" table with info from this and past seasonss FPL
"""
import json
import os
from typing import List, Optional

from sqlalchemy.orm.session import Session

from airsenal.framework.data_fetcher import FPLDataFetcher
from airsenal.framework.schema import Player, PlayerMapping, session, session_scope
from airsenal.framework.season import CURRENT_SEASON, sort_seasons
from airsenal.framework.utils import get_past_seasons
from airsenal.scripts.fill_player_mappings_table import (
    add_mappings,
    make_player_mappings_table,
)


def find_player_in_table(name: str, dbsession: Session) -> Optional[Player]:
    """
    see if we already have the player
    """
    player = dbsession.query(Player).filter_by(name=name).first()
    if not player:
        # see if we have an alternative name for this player
        mapping = dbsession.query(PlayerMapping).filter_by(alt_name=name).first()
        if mapping:
            player = (
                dbsession.query(Player).filter_by(player_id=mapping.player_id).first()
            )

    return player or None


def num_players_in_table(dbsession: Session) -> int:
    """
    how many players already in player table
    """
    players = dbsession.query(Player).all()
    return len(players)


def fill_player_table_from_file(filename: str, season: str, dbsession: Session) -> None:
    """
    use json file
    """
    jplayers = json.load(open(filename))
    for jp in jplayers:
        new_entry = False
        name = jp["name"]
        print(f"PLAYER {season} {name}")
        p = find_player_in_table(name, dbsession)
        if not p:
            new_entry = True
            p = Player()
            p.name = name
        if new_entry:
            dbsession.add(p)
            dbsession.commit()
            add_mappings(p, dbsession=dbsession)
    dbsession.commit()


def fill_player_table_from_api(season: str, dbsession: Session) -> None:
    """
    use the FPL API
    """
    df = FPLDataFetcher()
    pd = df.get_player_summary_data()

    for k, v in pd.items():
        p = Player()
        p.fpl_api_id = k
        first_name = v["first_name"]  # .encode("utf-8")
        second_name = v["second_name"]  # .encode("utf-8")
        name = f"{first_name} {second_name}"

        print(f"PLAYER {season} {name}")
        p.name = name
        dbsession.add(p)
    dbsession.commit()


def make_init_player_table(season: str, dbsession: Session = session) -> None:
    """
    Fill the player table with the latest season of data (only, as then need to do
    mappings)
    """
    if season == CURRENT_SEASON:
        # current season - use API
        fill_player_table_from_api(CURRENT_SEASON, dbsession)
    else:
        filename = os.path.join(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                f"player_summary_{season}.json",
            )
        )
        fill_player_table_from_file(filename, season, dbsession)


def make_remaining_player_table(
    seasons: Optional[List[str]] = [], dbsession: Session = session
) -> None:
    """
    Fill remaining players for subsequent seasons (AFTER players from the most recent
    season)
    """
    for season in seasons:
        filename = os.path.join(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                f"player_summary_{season}.json",
            )
        )
        fill_player_table_from_file(filename, season, dbsession)


def make_player_table(
    seasons: Optional[List[str]] = [], dbsession: Session = session
) -> None:
    if not seasons:
        seasons = [CURRENT_SEASON]
        seasons += get_past_seasons(3)
    seasons = sort_seasons(seasons)
    make_init_player_table(season=seasons[0], dbsession=dbsession)
    make_player_mappings_table(dbsession=dbsession)
    make_remaining_player_table(seasons=seasons[1:], dbsession=dbsession)


if __name__ == "__main__":
    with session_scope() as session:
        make_player_table(dbsession=session)
