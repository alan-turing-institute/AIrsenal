"""Fill the "fifa_rating" table from the fifa_team_ratings CSV files."""

from sqlalchemy.orm.session import Session

from airsenal.core.console import track
from airsenal.core.data_files import data_file
from airsenal.core.logging import get_logger
from airsenal.db.models import FifaTeamRating
from airsenal.db.session import get_session
from airsenal.game.mappings import canonical_team_name
from airsenal.game.season import default_seasons, sort_seasons

logger = get_logger(__name__)


def make_fifa_ratings_table(
    seasons: list[str] | None = None, dbsession: Session | None = None
) -> None:
    dbsession = get_session(dbsession)
    if not seasons:
        seasons = default_seasons()
    for season in track(sort_seasons(seasons), description="FIFA RATINGS"):
        input_path = data_file(f"fifa_team_ratings_{season}.csv")
        if not input_path.exists():
            logger.warning("No FIFA ratings file found for %s", season)
            continue

        with open(input_path) as input_file:
            for line in input_file.readlines()[1:]:
                team, att, mid, defn, ovr = line.strip().split(",")
                code = canonical_team_name(team)
                if code is None:
                    msg = f"Unknown team {team}."
                    raise ValueError(msg)
                r = FifaTeamRating()
                r.season = season
                r.team = code
                r.att = int(att)
                r.defn = int(defn)
                r.mid = int(mid)
                r.ovr = int(ovr)
                dbsession.add(r)
    dbsession.commit()
