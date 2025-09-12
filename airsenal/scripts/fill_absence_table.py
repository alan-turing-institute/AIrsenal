import os
from datetime import datetime

import pandas as pd
from sqlalchemy.orm.session import Session
from tqdm import tqdm

from airsenal.framework.schema import Absence, session
from airsenal.framework.season import CURRENT_SEASON, sort_seasons
from airsenal.framework.utils import (
    get_gameweek_by_date,
    get_past_seasons,
    get_player,
    get_return_gameweek_by_date,
)


def load_absences(season: str, dbsession: Session) -> None:
    print(f"ABSENCES {season}")
    path = os.path.join(
        os.path.dirname(__file__), "..", "data", f"absences_{season}.csv"
    )
    absences = pd.read_csv(path, parse_dates=["from", "until"])

    for _, row in tqdm(absences.iterrows(), total=absences.shape[0]):
        p = get_player(row["player"], dbsession=dbsession)
        if not p:
            print(f"Couldn't find player {row['player']}")
            continue

        date_from = row["from"].date()
        if date_from is pd.NaT:
            print(f"{row['player']} {row['details']} has no from date")
            continue

        # first check approx gameweek to determine player's team at that time
        gw_date = get_gameweek_by_date(
            check_date=date_from, season=season, dbsession=dbsession
        )
        if gw_date is None:
            print(f"Couldn't find gameweek for {row['player']} from date {date_from}")
            continue
        team_from = p.team(season, gw_date)
        # then get actual return gameweek using the player's team
        gw_from = get_return_gameweek_by_date(date_from, team_from, season, dbsession)

        date_until = None if row["until"] is pd.NaT else row["until"].date()
        if date_until is not None and (
            gw_date := get_gameweek_by_date(
                check_date=date_until, season=season, dbsession=dbsession
            )
        ):
            team_until = p.team(season, gw_date)
            gw_until = get_return_gameweek_by_date(
                date_until, team_until, season, dbsession
            )
        else:
            gw_until = None

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
    seasons: list[str] | None = None, dbsession: Session = session
) -> None:
    if seasons is None:
        seasons = []
    if not seasons:
        seasons = [CURRENT_SEASON]
        seasons += get_past_seasons(3)
    for season in sort_seasons(seasons):
        if season == CURRENT_SEASON:
            continue
        load_absences(season, dbsession)
