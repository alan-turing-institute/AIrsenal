"""Prediction and fixture tags, which group a run's rows together."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from airsenal.core.season import CURRENT_SEASON
from airsenal.db.models import Fixture, PlayerPrediction
from airsenal.db.session import get_session


def get_latest_prediction_tag(
    season: str = CURRENT_SEASON,
    tag_prefix: str = "",
    dbsession: Session | None = None,
) -> str:
    """
    Query the predicted_score table and get the tag field for the last row.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    query = select(PlayerPrediction).where(
        PlayerPrediction.fixture.has(Fixture.season == season)
    )
    if tag_prefix:
        query = query.where(PlayerPrediction.tag.startswith(tag_prefix))

    latest_prediction = dbsession.scalars(
        query.order_by(PlayerPrediction.id.desc()).limit(1)
    ).first()
    if latest_prediction is None:
        msg = (
            "No predicted points in database - has the database been filled?\n"
            "To calculate points predictions (and fill the database) use "
            "'airsenal_run_prediction'. This should be done before using "
            "'airsenal_make_squad' or 'airsenal_run_optimization'."
        )
        raise RuntimeError(msg)
    return latest_prediction.tag


def get_latest_fixture_tag(
    season: str = CURRENT_SEASON, dbsession: Session | None = None
) -> str:
    """
    Query the predicted_score table and get the tag field for the last row.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    latest_fixture = dbsession.scalars(
        select(Fixture)
        .where(Fixture.season == season)
        .order_by(Fixture.fixture_id.desc())
        .limit(1)
    ).first()
    if latest_fixture is None:
        msg = f"No fixtures found in database for season {season}"
        raise RuntimeError(msg)
    return latest_fixture.tag


def check_tag_valid(
    prediction_tag: str,
    gameweeks: list[int],
    season: str = CURRENT_SEASON,
    dbsession: Session | None = None,
) -> bool:
    """Check a prediction tag contains predictions for all the specified gameweeks."""
    # get unique gameweek and season values associated with prediction_tag
    dbsession = dbsession if dbsession is not None else get_session()
    fixtures = dbsession.execute(
        select(Fixture.season, Fixture.gameweek)
        .join(PlayerPrediction)
        .where(PlayerPrediction.tag == prediction_tag)
        .distinct()
    ).all()
    pred_seasons = [f[0] for f in fixtures]
    pred_gws = [f[1] for f in fixtures]

    season_ok = all(s == season for s in pred_seasons)
    gws_ok = all(gw in pred_gws for gw in gameweeks)

    return season_ok and gws_ok
