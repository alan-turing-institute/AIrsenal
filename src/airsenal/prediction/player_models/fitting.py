"""
Fitting a player model to the assembled training data.

The player-side twin of `team_models/fitting.py`: model-agnostic, so it takes a
`PlayerModel` rather than naming one. It is not re-exported from the package
`__init__`, because `features.py` imports the models and this imports
`features.py`.
"""

import pandas as pd
from sqlalchemy.orm import Session

from airsenal.core.copy import fastcopy
from airsenal.core.logging import get_logger
from airsenal.db.session import get_session
from airsenal.game.enums import Position
from airsenal.prediction.features import process_player_data
from airsenal.prediction.player_models import ConjugatePlayerModel
from airsenal.prediction.protocols import PlayerModel

logger = get_logger(__name__)


def fit_player_data(
    position: str,
    season: str,
    gameweek: int,
    model: PlayerModel | None = None,
    dbsession: Session | None = None,
) -> pd.DataFrame:
    """
    Fit the player model for a given position and return calculated probabilities.

    Hyperparameters live on the model, not here: pass a model constructed with
    the config you want, e.g. `ConjugatePlayerModel(ConjugatePlayerConfig(...))`.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if model is None:
        model = ConjugatePlayerModel()

    data = process_player_data(position, season, gameweek, dbsession)
    logger.info("Fitting player model for %s...", position)
    model = fastcopy(model)
    fitted_model = model.fit(data)
    df = pd.DataFrame(fitted_model.get_probs())

    df["pos"] = position
    return (
        df.rename(columns={"index": "player_id"})
        .sort_values("player_id")
        .set_index("player_id")
    )


def get_all_fitted_player_data(
    season: str,
    gameweek: int,
    model: PlayerModel | None = None,
    dbsession: Session | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Fit player models for all positions (GK, DEF, MID, FWD).
    """
    dbsession = dbsession if dbsession is not None else get_session()
    return {
        pos: fit_player_data(pos, season, gameweek, model, dbsession)
        for pos in list(Position.back_to_front())
    }
