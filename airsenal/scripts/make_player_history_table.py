"""
get values from player_score and match tables to use as input to Empirical Bayes model.
"""

from airsenal.framework.schema import PlayerScore
from airsenal.framework.utils import list_players, session


def get_player_history_table(position: str = "all") -> None:
    """
    Query the player_score table.
    """
    with open(f"player_history_{position}.csv", "w") as output_file:
        output_file.write(
            "player_id,player_name,match_id,goals,assists,minutes,team_goals\n"
        )
        players = list_players(position)
        for player in players:
            player_name = player.name
            results = (
                session.query(PlayerScore).filter_by(player_id=player.player_id).all()
            )
            row_count = 0
            for player_score in results:
                minutes = player_score.minutes
                fixture_id = player_score.fixture_id
                goals = player_score.goals
                assists = player_score.assists
                if player_score.fixture.home_team == player_score.opponent:
                    team_goals = player_score.result.away_score
                elif player_score.fixture.away_team == player_score.opponent:
                    team_goals = player_score.result.home_score
                else:
                    print("Unknown opponent!")
                    team_goals = -1
                output_file.write(
                    f"{player.player_id},{player_name},{fixture_id},{goals},{assists},"
                    f"{minutes},{team_goals}\n"
                )
                row_count += 1
            if row_count < 38 * 3:
                for _ in range(row_count, 38 * 3):
                    output_file.write(f"{player.player_id},{player_name},0,0,0,0,0\n")


if __name__ == "__main__":
    for position in ["GK", "DEF", "MID", "FWD"]:
        get_player_history_table(position)
