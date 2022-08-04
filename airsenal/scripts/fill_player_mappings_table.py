"""
Fill the "PlayerMapping" table with alternative names for players
"""
import csv
import os

from airsenal.framework.schema import Player, PlayerMapping


def load_mappings_data():
    filename = os.path.join(
        os.path.join(os.path.dirname(__file__), "..", "data", "player_mappings.csv")
    )
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        mappings_data = list(reader)
    return mappings_data


mappings_data = load_mappings_data()


def add_mappings(player, dbsession):
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


def make_player_mappings_table(dbsession):
    players = dbsession.query(Player).all()
    for p in players:
        add_mappings(p, dbsession)
