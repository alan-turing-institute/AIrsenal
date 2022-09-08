import os
from datetime import datetime

import pandas as pd

from airsenal.framework.schema import Absence, session
from airsenal.framework.utils import get_gameweek_by_date, get_past_seasons, get_player


def load_injuries(season, dbsession):
    reason = "injury"
    path = os.path.join(
        os.path.dirname(__file__), "..", "data", f"injuries_{season}.csv"
    )
    injuries = pd.read_csv(path)
    for _, row in injuries.iterrows():
        p = get_player(row["player"])
        if not p:
            print(f"Couldn't find player {row['player']}")
            continue
        date_from = row["from"]
        date_until = row["until"]
        gw_from = get_gameweek_by_date(date_from, season, dbsession)
        gw_until = get_gameweek_by_date(date_until, season, dbsession)
        details = row["injury"]
        url = row["url"]
        timestamp = datetime.now().isoformat()
        absence = Absence(
            player=p,
            season=season,
            reason=reason,
            details=details,
            date_from=date_from,
            date_until=date_until,
            gw_from=gw_from,
            gw_until=gw_until,
            url=url,
            timestamp=timestamp,
        )
        dbsession.add(absence)
    dbsession.commit()


def load_suspensions(season, dbsession):
    path = os.path.join(
        os.path.dirname(__file__), "..", "data", f"suspensions_{season}.csv"
    )
    _ = pd.read_csv(path)


def main(seasons=get_past_seasons(3), dbsession=session):
    for season in seasons:
        load_injuries(season, dbsession)
        load_suspensions(season, dbsession)


if __name__ == "__main__":
    load_injuries("1920", session)
