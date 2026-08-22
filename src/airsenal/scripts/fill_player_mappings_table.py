"""
Fill the "PlayerMapping" table with alternative names for players
"""

import csv

from sqlalchemy import select
from sqlalchemy.orm.session import Session

from airsenal.core.resources import resource
from airsenal.db.models import Player, PlayerMapping


def load_mappings_data() -> list[list[str]]:
    with resource("alternative_player_names.csv").open(encoding="UTF-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        return list(reader)


mappings_data = load_mappings_data()


def add_mappings(player: Player, dbsession: Session) -> None:
    for row in mappings_data:
        if player.name in row:
            # add mappings from row
            for alt_name in row:
                if alt_name != player.name:
                    mapping = PlayerMapping()
                    mapping.player_id = player.player_id
                    mapping.alt_name = alt_name
                    dbsession.add(mapping)
            dbsession.commit()
            break


def make_player_mappings_table(dbsession: Session) -> None:
    players = dbsession.scalars(select(Player)).all()
    for p in players:
        add_mappings(p, dbsession)
