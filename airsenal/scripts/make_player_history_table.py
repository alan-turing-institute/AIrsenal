#!/usr/bin/env python

"""
get values from player_score and match tables to use as input to
Empirical Bayes model.
"""
from airsenal.framework.schema import PlayerScore
from airsenal.framework.utils import list_players, get_player_name, session


def get_player_history_table(position="all"):
    """
    Query the player_score table.
    """
    with open("player_history_{}.csv".format(position), "w") as output_file:
        output_file.write(
            "player_id,player_name,match_id,goals,assists,minutes,team_goals\n"
        )
        player_ids = list_players(position)
        for pid in player_ids:
            player_name = get_player_name(pid)
            results = session.query(PlayerScore).filter_by(player_id=pid).all()
            row_count = 0
            for row in results:
                minutes = row.minutes
                match_id = row.match_id
                goals = row.goals
                assists = row.assists
                # find the match, in order to get team goals
                Match = None  # TODO: Placeholder for missing (deprecated?) Match class
                match = session.query(Match).filter_by(match_id=row.match_id).first()
                if match.home_team == row.opponent:
                    team_goals = match.away_score
                elif match.away_team == row.opponent:
                    team_goals = match.home_score
                else:
                    print("Unknown opponent!")
                    team_goals = -1
                output_file.write(
                    "{},{},{},{},{},{},{}\n".format(
                        pid, player_name, match_id, goals, assists, minutes, team_goals
                    )
                )
                row_count += 1
            if row_count < 38 * 3:
                for _ in range(row_count, 38 * 3):
                    output_file.write("{},{},0,0,0,0,0\n".format(pid, player_name))


if __name__ == "__main__":

    for position in ["GK", "DEF", "MID", "FWD"]:
        get_player_history_table(position)
