"""Recorded and predicted player performance in a fixture."""

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from airsenal.db.models.base import Base, intpk, str100, str100_optional

if TYPE_CHECKING:
    from airsenal.db.models.match import Fixture, Result
    from airsenal.db.models.player import Player


class PlayerScore(Base):
    __tablename__ = "player_score"
    __table_args__ = (
        Index("ix_player_score_fixture_id", "fixture_id"),
        Index("ix_player_score_player_fixture", "player_id", "fixture_id"),
    )
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    player_team: Mapped[str100]
    opponent: Mapped[str100]
    points: Mapped[int]
    goals: Mapped[int]
    assists: Mapped[int]
    bonus: Mapped[int]
    conceded: Mapped[int]
    minutes: Mapped[int]
    player: Mapped["Player"] = relationship(back_populates="scores")
    player_id: Mapped[int] = mapped_column(
        ForeignKey("player.player_id"), nullable=False
    )
    result: Mapped["Result"] = relationship()
    result_id: Mapped[int] = mapped_column(
        ForeignKey("result.result_id"), nullable=False
    )
    fixture: Mapped["Fixture"] = relationship()
    fixture_id: Mapped[int] = mapped_column(
        ForeignKey("fixture.fixture_id"), nullable=False
    )

    # extended features
    clean_sheets: Mapped[int | None]
    own_goals: Mapped[int | None]
    penalties_saved: Mapped[int | None]
    penalties_missed: Mapped[int | None]
    yellow_cards: Mapped[int | None]
    red_cards: Mapped[int | None]
    saves: Mapped[int | None]
    bps: Mapped[int | None]
    influence: Mapped[float | None]
    creativity: Mapped[float | None]
    threat: Mapped[float | None]
    ict_index: Mapped[float | None]
    expected_goals: Mapped[float | None]
    expected_assists: Mapped[float | None]
    expected_goal_involvements: Mapped[float | None]
    expected_goals_conceded: Mapped[float | None]
    defensive_contribution: Mapped[int | None]
    clearances_blocks_interceptions: Mapped[int | None]
    tackles: Mapped[int | None]
    recoveries: Mapped[int | None]

    # populated from PlayerAttributes history from the morning of the match
    chance_of_playing: Mapped[int | None]
    news: Mapped[str100_optional]

    def __repr__(self):
        return f"{self.player} ({self.result}): {self.points} pts, {self.minutes} mins"


class PlayerPrediction(Base):
    __tablename__ = "player_prediction"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    fixture: Mapped["Fixture"] = relationship()
    fixture_id: Mapped[int] = mapped_column(
        ForeignKey("fixture.fixture_id"), nullable=False
    )
    predicted_points: Mapped[float]
    tag: Mapped[str100]
    player: Mapped["Player"] = relationship(back_populates="predictions")
    player_id: Mapped[int] = mapped_column(
        ForeignKey("player.player_id"), nullable=False
    )

    def __repr__(self):
        return f"{self.player}: Predict {self.predicted_points} pts in {self.fixture}"
