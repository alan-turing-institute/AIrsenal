#!/usr/bin/env python

"""
query the transfer_suggestion table.  Each row of the table
will be in individual player in-or-out in a gameweek - we
therefore need to group together all the rows that correspond
to the same transfer strategy.  We do this using the "timestamp".
"""

import sys


from ..framework.schema import TransferSuggestion
from ..framework.utils import session, get_player_name

if __name__ == "__main__":
    all_rows = session.query(TransferSuggestion).all()
    last_timestamp = all_rows[-1].timestamp
    rows = (
        session.query(TransferSuggestion)
        .filter_by(timestamp=last_timestamp)
        .order_by(TransferSuggestion.gameweek)
        .all()
    )
    output_string = "Suggested transfer strategy: \n"
    current_gw = 0
    for row in rows:
        if row.gameweek != current_gw:
            output_string += " gameweek {}: ".format(row.gameweek)
            current_gw = row.gameweek
        if row.in_or_out < 0:
            output_string += " sell "
        else:
            output_string += " buy "
        output_string += get_player_name(row.player_id) + ","
    output_string += " for a total gain of {} points.".format(rows[0].points_gain)
    print(output_string)
