"""Teams, fixtures, results and team ratings."""

from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from airsenal.db.models.base import Base, intpk, str3, str4, str100


class Result(Base):
    __tablename__ = "result"
    result_id: Mapped[intpk] = mapped_column(autoincrement=True)
    fixture: Mapped["Fixture"] = relationship(back_populates="result")
    fixture_id: Mapped[int] = mapped_column(
        ForeignKey("fixture.fixture_id"), nullable=False
    )
    home_score: Mapped[int]
    away_score: Mapped[int]

    def __repr__(self):
        return (
            f"{self.fixture.season} GW{self.fixture.gameweek} "
            f"{self.fixture.home_team} {self.home_score} - "
            f"{self.away_score} {self.fixture.away_team}"
        )


class Fixture(Base):
    __tablename__ = "fixture"
    __table_args__ = (Index("ix_fixture_season_gameweek", "season", "gameweek"),)
    fixture_id: Mapped[intpk] = mapped_column(autoincrement=True)
    date: Mapped[str | None] = mapped_column(
        String(100)
    )  # In case fixture not yet scheduled!
    gameweek: Mapped[int | None]  # In case fixture not yet scheduled!
    home_team: Mapped[str100]
    away_team: Mapped[str100]
    season: Mapped[str100]
    tag: Mapped[str100]
    result: Mapped["Result | None"] = relationship(back_populates="fixture")

    def __repr__(self):
        return f"{self.season} GW{self.gameweek} {self.home_team} vs. {self.away_team}"


class FifaTeamRating(Base):
    __tablename__ = "fifa_rating"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    season: Mapped[str4]
    team: Mapped[str100]
    att: Mapped[int]
    defn: Mapped[int]
    mid: Mapped[int]
    ovr: Mapped[int]

    def __repr__(self):
        return (
            f"{self.team} {self.season} FIFA rating: "
            f"ovr {self.ovr}, def {self.defn}, mid {self.mid}, att {self.att}"
        )


class Team(Base):
    __tablename__ = "team"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    name: Mapped[str3]
    full_name: Mapped[str100]
    season: Mapped[str4]
    team_id: Mapped[int]  # the season-dependent team ID (from alphabetical order)

    def __repr__(self):
        return f"{self.full_name} ({self.name})"
