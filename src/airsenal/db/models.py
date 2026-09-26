"""The SQLAlchemy models: every table in the AIrsenal database."""

from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Annotated

from sqlalchemy import ForeignKey, Index, String, UniqueConstraint, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from airsenal.core.logging import get_logger

logger = get_logger(__name__)

intpk = Annotated[int, mapped_column(primary_key=True)]
str100 = Annotated[str, mapped_column(String(100))]
str4 = Annotated[str, mapped_column(String(4))]
str3 = Annotated[str, mapped_column(String(3))]
str100_optional = Annotated[str | None, mapped_column(String(100))]


class Base(DeclarativeBase):
    pass


# --- Teams, fixtures, results and team ratings ---


class Result(Base):
    __tablename__ = "result"
    result_id: Mapped[intpk] = mapped_column(autoincrement=True)
    fixture: Mapped["Fixture"] = relationship(back_populates="result")
    fixture_id: Mapped[int] = mapped_column(
        ForeignKey("fixture.fixture_id"), nullable=False
    )
    home_score: Mapped[int]
    away_score: Mapped[int]

    def __repr__(self) -> str:
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

    def __repr__(self) -> str:
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

    def __repr__(self) -> str:
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

    def __repr__(self) -> str:
        return f"{self.full_name} ({self.name})"


# --- Players and the per-gameweek attributes attached to them ---


class Player(Base):
    __tablename__ = "player"
    player_id: Mapped[intpk] = mapped_column(autoincrement=True)
    fpl_api_id: Mapped[int | None]
    name: Mapped[str100]
    display_name: Mapped[str100 | None]
    opta_code: Mapped[str | None]
    attributes: Mapped[list["PlayerAttributes"]] = relationship(
        back_populates="player",
        order_by="(PlayerAttributes.season.desc(), PlayerAttributes.gameweek.desc())",
    )

    predictions: Mapped[list["PlayerPrediction"]] = relationship(
        back_populates="player"
    )
    scores: Mapped[list["PlayerScore"]] = relationship(back_populates="player")

    def team(self, gameweek: int, season: str) -> str | None:
        """
        This player's team in a gameweek, or from the nearest one we have.

        None, with a warning, if there are no attributes at all for `season`.
        """
        attr = self._nearest_attributes(gameweek, season)
        if attr is not None:
            return attr.team
        logger.warning("No team found for %s in %s season.", self, season)
        return None

    def price(self, gameweek: int, season: str) -> int | None:
        """
        This player's price in a gameweek, interpolated if we have no exact match.

        None, with a warning, if there are no attributes at all for `season`.
        """
        attr = self.get_gameweek_attributes(gameweek, season, before_and_after=True)
        if attr is not None:
            return self._calculate_price(attr, gameweek)
        logger.warning("No price found for %s in %s season.", self, season)
        return None

    def _calculate_price(
        self,
        attr: "PlayerAttributes | tuple[PlayerAttributes, PlayerAttributes]",
        gameweek: int,
    ) -> int:
        """
        The price for one gameweek, straight from `attr` or interpolated.

        A pair of attributes means no exact match: the price is interpolated
        linearly between the gameweek before and the gameweek after.
        """
        if not isinstance(attr, tuple):
            return attr.price
        # interpolate price between nearest available gameweeks
        gameweek_before = attr[0].gameweek
        price_before = attr[0].price
        gameweek_after = attr[1].gameweek
        price_after = attr[1].price

        gradient = (price_after - price_before) / (gameweek_after - gameweek_before)
        intercept = price_before - gradient * gameweek_before
        price = gradient * gameweek + intercept
        return round(price)

    def position(self, season: str) -> str | None:
        """This player's position in `season`, or None if we have no attributes."""
        attr = self._nearest_attributes(None, season)
        if attr is not None:
            return attr.position
        logger.warning("No position found for %s in %s season.", self, season)
        return None

    def is_injured_or_suspended(
        self, season: str, current_gameweek: int, fixture_gameweek: int
    ) -> bool:
        """
        Whether a player is injured or suspended (<=50% chance of playing).

        The two gameweeks are different points in time: `current_gameweek` is when we
        are asking, and `fixture_gameweek` is the future fixture we are asking about.
        So this answers "as of `current_gameweek`, did we expect this player to still
        be out by `fixture_gameweek`?".
        """
        attr = self._nearest_attributes(current_gameweek, season)
        if attr is not None:
            return (
                attr.chance_of_playing_next_round is not None
                and attr.chance_of_playing_next_round <= 50
            ) and (
                attr.return_gameweek is None or attr.return_gameweek > fixture_gameweek
            )
        return False

    def _nearest_attributes(
        self, gameweek: int | None, season: str
    ) -> "PlayerAttributes | None":
        """As `get_gameweek_attributes`, for the one gameweek nearest `gameweek`."""
        attr = self.get_gameweek_attributes(gameweek, season)
        # a pair only comes back with before_and_after, so this never sees one
        return None if isinstance(attr, tuple) else attr

    def _season_attributes(self, season: str) -> "_SeasonAttributes | None":
        """
        This player's attributes for `season`, indexed by gameweek.

        Built from `attributes` on first use and kept on the instance, and built
        again if a row has been added or removed since. A row changed in place
        needs nothing: the index holds the rows themselves.
        """
        attributes = self.attributes
        cached = self.__dict__.get("_attributes_by_season")
        if cached is None or cached[0] != len(attributes):
            by_season: dict[str, _SeasonAttributes] = {}
            for attr in attributes:
                if attr.season not in by_season:
                    by_season[attr.season] = _SeasonAttributes(first=attr)
                # the first row for a gameweek wins, as the scan it replaced did
                by_season[attr.season].by_gameweek.setdefault(attr.gameweek, attr)
            for season_attributes in by_season.values():
                season_attributes.gameweeks = sorted(season_attributes.by_gameweek)
            cached = (len(attributes), by_season)
            self.__dict__["_attributes_by_season"] = cached
        return cached[1].get(season)

    def get_gameweek_attributes(
        self, gameweek: int | None, season: str, before_and_after: bool = False
    ) -> "PlayerAttributes | tuple[PlayerAttributes, PlayerAttributes] | None":
        """
        This player's attributes for a gameweek, or from the nearest one we have.

        Returns None in all cases if there are no attributes at all for `season`.
        With `before_and_after` True and no exact match, returns both the nearest
        gameweek before and the nearest after, as a tuple.
        """
        season_attributes = self._season_attributes(season)
        if season_attributes is None:
            return None
        if gameweek is None:
            # trying to match season only
            return season_attributes.first
        exact = season_attributes.by_gameweek.get(gameweek)
        if exact is not None:
            return exact

        return season_attributes.nearest(gameweek, before_and_after)

    def __repr__(self) -> str:
        return self.display_name or self.name


@dataclass
class _SeasonAttributes:
    """One player's attribute rows for one season, indexed by gameweek."""

    # the first row in `Player.attributes` order, which loads latest gameweek first
    first: "PlayerAttributes"
    by_gameweek: dict[int, "PlayerAttributes"] = field(default_factory=dict)
    gameweeks: list[int] = field(default_factory=list)

    def nearest(
        self, gameweek: int, before_and_after: bool
    ) -> "PlayerAttributes | tuple[PlayerAttributes, PlayerAttributes] | None":
        """
        The row nearest a gameweek with no row of its own.

        Only gameweeks from 1 to 99 count: one at 0 or from 100 on is found by an
        exact match alone. A tie goes to the gameweek before, and with
        `before_and_after` both neighbours come back when there are two.
        """
        i = bisect_left(self.gameweeks, gameweek)
        gameweek_before, attr_before = 0, None
        if i > 0 and self.gameweeks[i - 1] > 0:
            gameweek_before = self.gameweeks[i - 1]
            attr_before = self.by_gameweek[gameweek_before]
        gameweek_after, attr_after = 100, None
        if i < len(self.gameweeks) and self.gameweeks[i] < 100:
            gameweek_after = self.gameweeks[i]
            attr_after = self.by_gameweek[gameweek_after]

        if attr_before is None or attr_after is None:
            return attr_before if attr_after is None else attr_after
        if before_and_after:
            return (attr_before, attr_after)
        if (gameweek_after - gameweek) >= (gameweek - gameweek_before):
            return attr_before
        return attr_after


class PlayerMapping(Base):
    # alternative names for players
    __tablename__ = "player_mapping"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("player.player_id"))
    alt_name: Mapped[str100]


class PlayerAttributes(Base):
    __tablename__ = "player_attributes"
    __table_args__ = (
        UniqueConstraint(
            "player_id",
            "season",
            "gameweek",
            name="uq_player_attributes_player_season_gw",
        ),
        Index(
            "ix_player_attributes_player_season_gw",
            "player_id",
            "season",
            "gameweek",
        ),
    )
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    player: Mapped["Player"] = relationship(back_populates="attributes")
    player_id: Mapped[int] = mapped_column(
        ForeignKey("player.player_id"), nullable=False
    )
    season: Mapped[str100]
    gameweek: Mapped[int]
    price: Mapped[int]
    team: Mapped[str100]
    position: Mapped[str100]

    chance_of_playing_next_round: Mapped[int | None]
    news: Mapped[str100_optional]
    return_gameweek: Mapped[int | None]
    transfers_balance: Mapped[int | None]
    selected: Mapped[int | None]
    transfers_in: Mapped[int | None]
    transfers_out: Mapped[int | None]

    def __repr__(self) -> str:
        return (
            f"{self.player} ({self.season} GW{self.gameweek}): "
            f"£{self.price / 10}, {self.team}, {self.position}"
        )


# --- Recorded and predicted player performance in a fixture ---


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

    # what was known about the player's availability for this match
    chance_of_playing: Mapped[int | None]
    news: Mapped[str100_optional]

    def __repr__(self) -> str:
        return f"{self.player} ({self.result}): {self.points} pts, {self.minutes} mins"


class PlayerPrediction(Base):
    __tablename__ = "player_prediction"
    __table_args__ = (Index("ix_player_prediction_tag_player", "tag", "player_id"),)
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

    def __repr__(self) -> str:
        return f"{self.player}: Predict {self.predicted_points} pts in {self.fixture}"


# --- The user's squad: transactions, suggestions and session state ---


class Transaction(Base):
    __tablename__ = "transaction"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    player_id: Mapped[int]
    gameweek: Mapped[int]
    bought_or_sold: Mapped[int]  # +1 for bought, -1 for sold
    season: Mapped[str100]
    time: Mapped[str100]
    tag: Mapped[str100]
    price: Mapped[int]
    free_hit: Mapped[int]  # 1 if transfer on Free Hit, 0 otherwise
    # 1 if this came out of the entry's free-transfer quota, 0 otherwise. A
    # wildcard or free hit makes transfers unlimited, and the fifteen players an
    # entry starts with were never transfers at all, so neither is charged for.
    # Separate from free_hit, which says the change lasts a single gameweek.
    counts_as_transfer: Mapped[int] = mapped_column(default=1, server_default=text("1"))
    fpl_team_id: Mapped[int]

    def __repr__(self) -> str:
        trans_str = f"{self.season} GW{self.gameweek}: Team {self.fpl_team_id} "
        if self.bought_or_sold == 1:
            trans_str += f"bought player {self.player_id}"
        else:
            trans_str += f"sold player {self.player_id}"
        if self.free_hit:
            trans_str += " (FREE HIT)"
        elif not self.counts_as_transfer:
            trans_str += " (FREE)"
        return trans_str


class TransferSuggestion(Base):
    __tablename__ = "transfer_suggestion"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    player_id: Mapped[int]
    in_or_out: Mapped[int]  # +1 for buy, -1 for sell
    gameweek: Mapped[int]
    points_gain: Mapped[float]
    timestamp: Mapped[str100]  # use this to group suggestions
    season: Mapped[str100]
    fpl_team_id: Mapped[int]  # to identify team to apply transfers.
    chip_played: Mapped[str100_optional]

    def __repr__(self) -> str:
        sugg_str = f"{self.season} GW{self.gameweek}: Suggest "
        if self.in_or_out == 1:
            sugg_str += f"buying {self.player_id} to gain {self.points_gain:.2f} pts"
        else:
            sugg_str += f"selling {self.player_id} to gain {self.points_gain:.2f} pts"
        return sugg_str
