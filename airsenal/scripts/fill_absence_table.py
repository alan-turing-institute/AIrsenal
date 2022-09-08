import os
from datetime import datetime
import numpy as np
import pandas as pd

from airsenal.framework.schema import Absence, session
from airsenal.framework.utils import get_gameweek_for_date, get_past_seasons, get_player


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
        # date_until (and therefore gw_until) can be null
        date_until = None if np.isnan(row["until"]) else row["until"]
        gw_from = get_gameweek_for_date(date_from, season, dbsession)
        gw_until = None if not date_until else \
            get_gameweek_for_date(date_until, season, dbsession)
        print(f"Dates {gw_from} {gw_until}")
        details = row["injury"]
        url = row["url"]
        timestamp = datetime.now().isoformat()
        absence = Absence(
            player=p,
            player_id=p.player_id,
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
        print(absence)
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
