#!/usr/bin/env python
"""
query the transfer suggestion table and print the suggested strategy

"""


from airsenal.framework.schema import TransferSuggestion
from airsenal.framework.utils import session, get_player_name

def get_transfer_suggestions(currentsession, Suggestions):
    """
    query the transfer_suggestion table.  Each row of the table
    will be in individual player in-or-out in a gameweek - we
    therefore need to group together all the rows that correspond
    to the same transfer strategy.  We do this using the "timestamp".
    """
    all_rows = currentsession.query(Suggestions).all()
    last_timestamp = all_rows[-1].timestamp
    rows = (
        session.query(TransferSuggestion)
        .filter_by(timestamp=last_timestamp)
        .order_by(TransferSuggestion.gameweek)
        .all()
    )
    return rows

def build_strategy_string(rows):
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
    return output_string

if __name__ == "__main__":
    rows = get_transfer_suggestions(session, TransferSuggestion)
    output_string = build_strategy_string(rows)
    print(output_string)
