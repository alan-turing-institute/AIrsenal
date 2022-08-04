"""
Fill the "PlayerMappings" table with alternative names for players
"""


def make_player_mappings_table(dbsession):
    # load data file
    # for each row in data file
    #    check whether name in any column matches a name in Player table
    #       if yes add to mappings table (with current player ID of player)
    #       if not try some fancy string matching (optionally?)
    #          if good potential match, prompt user to verify
    #             if yes match add to mappings table (with current player ID)
    #             and update data file (or save to list of rows to add later)
    raise NotImplementedError()
