"""
Fill the "fifa_ratings" table with info from fifa_team_ratings CSV files.
"""

from sqlalchemy.orm.session import Session

from airsenal.core.console import track
from airsenal.core.data_files import data_file
from airsenal.core.logging import get_logger
from airsenal.db.models import FifaTeamRating
from airsenal.db.session import get_session
from airsenal.game.mappings import alternative_team_names
from airsenal.game.season import CURRENT_SEASON, get_past_seasons, sort_seasons

logger = get_logger(__name__)


def make_fifa_ratings_table(
    seasons: list[str] | None = None, dbsession: Session | None = None
) -> None:
    # make the fifa ratings table
    # TODO: scrape the data first rather than committing file to repo

    dbsession = dbsession if dbsession is not None else get_session()
    if seasons is None:
        seasons = []
    if not seasons:
        seasons = [CURRENT_SEASON]
        seasons += get_past_seasons(3)
    for season in track(sort_seasons(seasons), description="FIFA RATINGS"):
        input_path = data_file(f"fifa_team_ratings_{season}.csv")
        if not input_path.exists():
            logger.warning("No FIFA ratings file found for %s", season)
            continue

        with open(input_path) as input_file:
            for line in input_file.readlines()[1:]:
                team, att, mid, defn, ovr = line.strip().split(",")
                r = FifaTeamRating()
                r.season = season
                r.team = team
                r.att = int(att)
                r.defn = int(defn)
                r.mid = int(mid)
                r.ovr = int(ovr)
                team_is_known = False
                for k, v in alternative_team_names.items():
                    if team in v:
                        r.team = k
                        team_is_known = True
                    elif team == k:
                        team_is_known = True
                if not team_is_known:
                    msg = f"Unknown team {team}."
                    raise ValueError(msg)
                dbsession.add(r)
    dbsession.commit()
