"""SQLAlchemy models. Import them from here rather than the individual modules."""

from airsenal.db.models.base import Base
from airsenal.db.models.match import FifaTeamRating, Fixture, Result, Team
from airsenal.db.models.performance import PlayerPrediction, PlayerScore
from airsenal.db.models.player import (
    Absence,
    Player,
    PlayerAttributes,
    PlayerMapping,
)
from airsenal.db.models.squad import (
    SessionBudget,
    SessionSquad,
    Transaction,
    TransferSuggestion,
)

__all__ = [
    "Absence",
    "Base",
    "FifaTeamRating",
    "Fixture",
    "Player",
    "PlayerAttributes",
    "PlayerMapping",
    "PlayerPrediction",
    "PlayerScore",
    "Result",
    "SessionBudget",
    "SessionSquad",
    "Team",
    "Transaction",
    "TransferSuggestion",
]
