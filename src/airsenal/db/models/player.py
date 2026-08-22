"""Players and the per-gameweek attributes attached to them."""

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from airsenal.core.logging import get_logger
from airsenal.db.models.base import Base, intpk, str100, str100_optional

if TYPE_CHECKING:
    from airsenal.db.models.performance import PlayerPrediction, PlayerScore


logger = get_logger(__name__)


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

    absences: Mapped[list["Absence"]] = relationship(back_populates="player")
    predictions: Mapped[list["PlayerPrediction"]] = relationship(
        back_populates="player"
    )
    scores: Mapped[list["PlayerScore"]] = relationship(back_populates="player")

    def team(self, season: str, gameweek: int) -> str | None:
        """
        Get player's team for given season and gameweek.
        If data not available for specified gameweek but data is available for
        at least one gameweek in specified season, return a best guess value
        based on data nearest to specified gameweek.
        """
        attr = self.get_gameweek_attributes(season, gameweek)
        if attr is not None and not isinstance(attr, tuple):
            return attr.team
        logger.warning("No team found for %s in %s season.", self, season)
        return None

    def price(self, season: str, gameweek: int) -> int | None:
        """
        get player's price for given season and gameweek
        If data not available for specified gameweek but data is available for
        at least one gameweek in specified season, return a best guess value
        based on data nearest to specified gameweek.
        """
        attr = self.get_gameweek_attributes(season, gameweek, before_and_after=True)
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
        Either return price available for specified gameweek or interpolate based
        on nearest available price.
        """
        if not isinstance(attr, tuple):
            return attr.price
        # interpolate price between nearest available gameweeks
        gw_before = attr[0].gameweek
        price_before = attr[0].price
        gw_after = attr[1].gameweek
        price_after = attr[1].price

        gradient = (price_after - price_before) / (gw_after - gw_before)
        intercept = price_before - gradient * gw_before
        price = gradient * gameweek + intercept
        return round(price)

    def position(self, season: str) -> str | None:
        """
        get player's position for given season
        """
        attr = self.get_gameweek_attributes(season, None)
        if attr is not None and not isinstance(attr, tuple):
            return attr.position
        logger.warning("No position found for %s in %s season.", self, season)
        return None

    def is_injured_or_suspended(
        self, season: str, current_gw: int, fixture_gw: int
    ) -> bool:
        """Check whether a player is injured or suspended (<=50% chance of playing).
        current_gw - The current gameweek, i.e. the gameweek when we are querying the
        player's status.
        fixture_gw - The gameweek of the fixture we want to check whether the player
        is available for, i.e. we are checking whether the player is availiable in the
        future week "fixture_gw" at the previous point in time "current_gw".
        """
        attr = self.get_gameweek_attributes(season, current_gw)
        if attr is not None and not isinstance(attr, tuple):
            return (
                attr.chance_of_playing_next_round is not None
                and attr.chance_of_playing_next_round <= 50
            ) and (attr.return_gameweek is None or attr.return_gameweek > fixture_gw)
        return False

    def get_gameweek_attributes(
        self, season: str, gameweek: int | None, before_and_after: bool = False
    ) -> "PlayerAttributes | tuple[PlayerAttributes, PlayerAttributes] | None":
        """Get the PlayerAttributes object for this player in the given gameweek and
        season, or the nearest available gameweek(s) if the exact gameweek is not
        available.
        If no attributes available in the specified season, return None in all cases.
        If before_and_after is True and an exact gameweek & season match is not found,
        return both the nearest gameweek before and after the specified gameweek.
        """
        gw_before = 0
        gw_after = 100
        attr_before = None
        attr_after = None

        for attr in self.attributes:
            if attr.season != season:
                continue

            if gameweek is None:
                # trying to match season only
                return attr
            if attr.gameweek == gameweek:
                return attr
            if (attr.gameweek < gameweek) and (attr.gameweek > gw_before):
                # update last available attr before specified gameweek
                gw_before = attr.gameweek
                attr_before = attr
            elif (attr.gameweek > gameweek) and (attr.gameweek < gw_after):
                # update next available attr after specified gameweek
                gw_after = attr.gameweek
                attr_after = attr

        # ran through all attributes without finding exact gameweek and season match
        if attr_before is None and attr_after is None:
            # no attributes for this player in this season
            return None
        if not attr_after:
            return attr_before
        if not attr_before:
            return attr_after
        if before_and_after:
            return (attr_before, attr_after)
        # return attributes at gameweeek nearest to input gameweek
        if gameweek is not None and (gw_after - gameweek) >= (gameweek - gw_before):
            return attr_before
        return attr_after

    def __repr__(self):
        return self.display_name or self.name


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

    def __repr__(self):
        return (
            f"{self.player} ({self.season} GW{self.gameweek}): "
            f"£{self.price / 10}, {self.team}, {self.position}"
        )


class Absence(Base):
    __tablename__ = "absence"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    player: Mapped["Player"] = relationship(back_populates="absences")
    player_id: Mapped[int | None] = mapped_column(ForeignKey("player.player_id"))
    season: Mapped[str100]
    reason: Mapped[str100]  # high-level, e.g. injury/suspension
    details: Mapped[str100_optional]
    # ISO-8601 dates ("2025-08-16") held as text, not DATE columns. Changing the
    # column type would need a migration, and this repo has no Alembic: create_all
    # does not alter existing tables, so sqlite and postgres users alike would keep
    # the VARCHAR columns their database already has.
    date_from: Mapped[str100]
    date_until: Mapped[str100_optional]
    gw_from: Mapped[int]
    gw_until: Mapped[int | None]
    url: Mapped[str100_optional]
    timestamp: Mapped[str100]

    def __repr__(self):
        return (
            f"Absence(\n"
            f"  player='{self.player}',\n"
            f"  player_id='{self.player_id}',\n"
            f"  season='{self.season}',\n"
            f"  reason='{self.reason}',\n"
            f"  details='{self.details}',\n"
            f"  date_from='{self.date_from}',\n"
            f"  date_until='{self.date_until}',\n"
            f"  gw_from='{self.gw_from}',\n"
            f"  gw_until='{self.gw_until}',\n"
            f"  url='{self.url}',\n"
            f"  timestamp='{self.timestamp}'\n"
            ")"
        )
