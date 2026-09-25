"""Points for being on the pitch at all."""

from sqlalchemy.orm import Session

from airsenal.game.scoring import get_appearance_points
from airsenal.prediction.protocols import ComponentRequest


class AppearanceComponent:
    """A point for playing, and another for playing an hour."""

    name = "appearance"

    def fit(
        self, gameweek: int, season: str, dbsession: Session
    ) -> "AppearanceComponent":
        """Nothing to fit: appearance points are a rule, not an average."""
        del gameweek, season, dbsession
        return self

    def expected_points(self, request: ComponentRequest) -> float:
        return get_appearance_points(request.minutes)
