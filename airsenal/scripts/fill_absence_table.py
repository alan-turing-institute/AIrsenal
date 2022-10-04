import os
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from airsenal.framework.schema import Absence, session
from airsenal.framework.utils import (
    get_next_gameweek_by_date,
    get_past_seasons,
    get_player,
)


def load_injuries(season, dbsession):
    print(f"INJURIES {season}")
    reason = "injury"
    path = os.path.join(
        os.path.dirname(__file__), "..", "data", f"injuries_{season}.csv"
    )
    injuries = pd.read_csv(
        path, parse_dates=["from", "until"], infer_datetime_format=True
    )

    for _, row in tqdm(injuries.iterrows(), total=injuries.shape[0]):
        p = get_player(row["player"], dbsession=dbsession)
        if not p:
            print(f"Couldn't find player {row['player']}")
            continue
        date_from = row["from"].date()
        if date_from is pd.NaT:
            print(f"{row['player']} {row['injury']} has no from date")
            continue
        date_until = None if row["until"] is pd.NaT else row["until"].date()
        gw_from = get_next_gameweek_by_date(date_from, season, dbsession)
        gw_until = (
            get_next_gameweek_by_date(date_until, season, dbsession)
            if date_until
            else None
        )
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
        dbsession.add(absence)
    dbsession.commit()


def get_reason(details):
    """get suspension/absence reason category (not for injuries)"""
    return "suspension" if "suspen" in details.lower() else "absence"


def load_suspensions(season, dbsession):
    print(f"SUSPENSIONS {season}")
    path = os.path.join(
        os.path.dirname(__file__), "..", "data", f"suspensions_{season}.csv"
    )
    suspensions = pd.read_csv(
        path, parse_dates=["from", "until"], infer_datetime_format=True
    )

    for _, row in tqdm(suspensions.iterrows(), total=suspensions.shape[0]):
        if (
            isinstance(row["competition"], str)
            and row["competition"] != "Premier League"
        ):
            continue
        p = get_player(row["player"], dbsession=dbsession)
        if not p:
            print(f"Couldn't find player {row['player']}")
            continue
        date_from = row["from"].date()
        if date_from is pd.NaT:
            print(f"{row['player']} {row['absence/suspension']} has no from date")
            continue
        date_until = None if row["until"] is pd.NaT else row["until"].date()

        gw_from = get_next_gameweek_by_date(date_from, season, dbsession)
        gw_until = (
            get_next_gameweek_by_date(date_until, season, dbsession)
            if date_until
            else None
        )

        details = row["absence/suspension"]
        reason = get_reason(details)
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

        dbsession.add(absence)
    dbsession.commit()


def make_absence_table(seasons=get_past_seasons(3), dbsession=session):
    for season in seasons:
        load_injuries(season, dbsession)
        load_suspensions(season, dbsession)
