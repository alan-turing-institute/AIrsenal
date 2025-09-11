import csv
import os
import re
from datetime import date, datetime

import dateparser

from airsenal.framework.data_fetcher import FPLDataFetcher
from airsenal.framework.mappings import positions
from airsenal.framework.season import CURRENT_SEASON
from airsenal.framework.utils import (
    NEXT_GAMEWEEK,
    parse_date,
)


def get_return_gameweek_by_date(
    check_date: datetime | None, ordered_deadlines: list[tuple[int, date]]
) -> int | None:
    """
    Use a date, or easily parse-able date string to figure out which gameweek its in.
    """
    if check_date is None:
        return None

    for gw, deadline in ordered_deadlines:
        if deadline >= check_date.date():
            return gw

    return None


def save_attributes_from_api(
    file_path: str, season: str = CURRENT_SEASON, gameweek: int = NEXT_GAMEWEEK
) -> None:
    """
    use the FPL API to get player attributes info for the current season
    """

    if not os.path.isfile(file_path):
        with open(file_path, "w") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(
                [
                    "timestamp",
                    "season",
                    "gameweek",
                    "team",
                    "position",
                    "player_id",
                    "opta_code",
                    "player",
                    "price",
                    "selected",
                    "transfers_in",
                    "transfers_out",
                    "transfers_balance",
                    "news",
                    "chance_of_playing_next_round",
                    "return_gameweek",
                ]
            )

    fetcher = FPLDataFetcher()
    timestamp = datetime.isoformat(datetime.now())
    # needed for selected by calculation from percentage below
    summary_data = fetcher.get_current_summary_data()
    n_players = summary_data["total_players"]
    teams = {team["id"]: team["short_name"] for team in summary_data["teams"]}
    deadlines = sorted(
        [
            (int(gw["id"]), parse_date(gw["deadline_time"]))
            for gw in summary_data["events"]
        ]
    )

    input_data = fetcher.get_player_summary_data()

    with open(file_path, "a") as f:
        writer = csv.writer(f, delimiter=",")
        for player_api_id, player_data in input_data.items():
            name = f"{player_data['first_name']} {player_data['second_name']}"
            print(name)
            opta_code = player_data["opta_code"]
            position = positions[player_data["element_type"]]
            price = int(player_data["now_cost"])
            team = teams[player_data["team"]]
            selected = int(float(player_data["selected_by_percent"]) * n_players / 100)
            transfers_in = int(player_data["transfers_in"])
            transfers_out = int(player_data["transfers_out"])
            transfers_balance = transfers_in - transfers_out
            news = player_data["news"]
            chance_of_playing_next_round = player_data["chance_of_playing_next_round"]
            if (
                chance_of_playing_next_round is not None
                and chance_of_playing_next_round <= 50
            ):
                rd_rex = "(Expected back|Suspended until)[\\s]+([\\d]+[\\s][\\w]{3})"
                search_results = re.search(rd_rex, news)
                if search_results:
                    return_str = search_results.groups()[1]
                    # return_str should be a day and month string (without year)
                    # create a date in the future from the day and month string
                    return_date = dateparser.parse(
                        return_str, settings={"PREFER_DATES_FROM": "future"}
                    )
                    return_gameweek = get_return_gameweek_by_date(
                        return_date, deadlines
                    )
                else:
                    return_gameweek = None
            else:
                return_gameweek = None

            writer.writerow(
                [
                    timestamp,
                    season,
                    gameweek,
                    team,
                    position,
                    player_api_id,
                    opta_code,
                    name,
                    price,
                    selected,
                    transfers_in,
                    transfers_out,
                    transfers_balance,
                    news,
                    chance_of_playing_next_round,
                    return_gameweek,
                ]
            )


if __name__ == "__main__":
    file_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        f"player_attributes_history_{CURRENT_SEASON}.csv",
    )
    save_attributes_from_api(file_path)
