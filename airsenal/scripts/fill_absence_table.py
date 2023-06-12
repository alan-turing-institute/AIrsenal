import os
from datetime import datetime
from typing import List, Optional

import pandas as pd
from sqlalchemy.orm.session import Session
from tqdm import tqdm

from airsenal.framework.schema import Absence, session
from airsenal.framework.season import CURRENT_SEASON, sort_seasons
from airsenal.framework.utils import (
    get_next_gameweek_by_date,
    get_past_seasons,
    get_player,
)


def load_absences(season: str, dbsession: Session) -> None:
    print(f"ABSENCES {season}")
    path = os.path.join(
        os.path.dirname(__file__), "..", "data", f"absences_{season}.csv"
    )
    absences = pd.read_csv(
        path, parse_dates=["from", "until"], infer_datetime_format=True
    )

    for _, row in tqdm(absences.iterrows(), total=absences.shape[0]):
        p = get_player(row["player"], dbsession=dbsession)
        if not p:
            print(f"Couldn't find player {row['player']}")
            continue
        date_from = row["from"].date()
        if date_from is pd.NaT:
            print(f"{row['player']} {row['details']} has no from date")
            continue
        date_until = None if row["until"] is pd.NaT else row["until"].date()

        gw_from = get_next_gameweek_by_date(date_from, season, dbsession)
        gw_until = (
            get_next_gameweek_by_date(date_until, season, dbsession)
            if date_until
            else None
        )

        url = row["url"]
        timestamp = datetime.now().isoformat()
        absence = Absence(
            player=p,
            player_id=p.player_id,
            season=season,
            reason=row["reason"],
            details=row["details"],
            date_from=date_from,
            date_until=date_until,
            gw_from=gw_from,
            gw_until=gw_until,
            url=url,
            timestamp=timestamp,
        )
        dbsession.add(absence)
    dbsession.commit()


def make_absence_table(
    seasons: Optional[List[str]] = [], dbsession: Session = session
) -> None:
    if not seasons:
        seasons = [CURRENT_SEASON]
        seasons += get_past_seasons(3)
    for season in sort_seasons(seasons):
        if season == CURRENT_SEASON:
            continue
        load_absences(season, dbsession)
