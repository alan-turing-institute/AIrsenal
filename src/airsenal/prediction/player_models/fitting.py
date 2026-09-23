"""Fitting a player model to the assembled training data."""

import pandas as pd
from sqlalchemy.orm import Session

from airsenal.core.copy import fastcopy
from airsenal.core.logging import get_logger
from airsenal.db.session import get_session
from airsenal.game.enums import Position
from airsenal.prediction.features import process_player_data
from airsenal.prediction.player_models import build_player_model
from airsenal.prediction.protocols import PlayerModel

logger = get_logger(__name__)


def fit_player_data(
    position: str,
    gameweek: int,
    season: str,
    model: PlayerModel | None = None,
    dbsession: Session | None = None,
) -> pd.DataFrame:
    """
    Fit the player model for a given position and return calculated probabilities.

    Hyperparameters live on the model: pass one constructed with the config you
    want, e.g. `XGPlayerModel(XGPlayerConfig(...))`. `None` means the model
    `DEFAULT_PLAYER_MODEL` names.
    """
    dbsession = get_session(dbsession)
    if model is None:
        model = build_player_model()

    data = process_player_data(position, gameweek, season, dbsession)
    logger.info("Fitting player model for %s...", position)
    model = fastcopy(model)
    fitted_model = model.fit(data)
    df = pd.DataFrame(fitted_model.predict_involvement().as_dict())

    df["pos"] = position
    return (
        df.rename(columns={"index": "player_id"})
        .sort_values("player_id")
        .set_index("player_id")
    )


def get_all_fitted_player_data(
    gameweek: int,
    season: str,
    model: PlayerModel | None = None,
    dbsession: Session | None = None,
) -> dict[str, pd.DataFrame]:
    """Fit player models for all positions (GK, DEF, MID, FWD)."""
    dbsession = get_session(dbsession)
    return {
        pos: fit_player_data(pos, gameweek, season, model, dbsession)
        for pos in list(Position.back_to_front())
    }
