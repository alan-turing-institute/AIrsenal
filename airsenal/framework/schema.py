"""
Interface to the SQL database.
Use SQLAlchemy to convert between DB tables and python objects.
"""

from contextlib import contextmanager
from typing import Annotated

from sqlalchemy import ForeignKey, String, create_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

from airsenal.framework.env import (
    AIRSENAL_DB_FILE,
    AIRSENAL_DB_PASSWORD,
    AIRSENAL_DB_URI,
    AIRSENAL_DB_USER,
    AIRSENAL_HOME,
    save_env,
)

# Common type annotations using PEP 593 Annotated
intpk = Annotated[int, mapped_column(primary_key=True)]
str100 = Annotated[str, mapped_column(String(100))]
str4 = Annotated[str, mapped_column(String(4))]
str3 = Annotated[str, mapped_column(String(3))]
str100_optional = Annotated[str | None, mapped_column(String(100))]


class Base(DeclarativeBase):
    pass


class Player(Base):
    __tablename__ = "player"
    player_id: Mapped[intpk] = mapped_column(autoincrement=True)
    fpl_api_id: Mapped[int | None]
    name: Mapped[str100]
    attributes: Mapped[list["PlayerAttributes"]] = relationship(back_populates="player")
    absences: Mapped[list["Absence"]] = relationship(back_populates="player")
    results: Mapped[list["Result"]] = relationship(back_populates="player")
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
        print("No team found for", self.name, "in", season, "season.")
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
        print("No price found for", self.name, "in", season, "season.")
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
        print("No position found for", self.name, "in", season, "season.")
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

    def __str__(self):
        return self.name


class PlayerMapping(Base):
    # alternative names for players
    __tablename__ = "player_mapping"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("player.player_id"))
    alt_name: Mapped[str100]


class PlayerAttributes(Base):
    __tablename__ = "player_attributes"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    player: Mapped["Player"] = relationship(back_populates="attributes")
    player_id: Mapped[int | None] = mapped_column(ForeignKey("player.player_id"))
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

    def __str__(self):
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
    date_from: Mapped[str100]
    date_until: Mapped[str100_optional]
    gw_from: Mapped[int]
    gw_until: Mapped[int | None]
    url: Mapped[str100_optional]
    timestamp: Mapped[str100]

    def __str__(self):
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


class Result(Base):
    __tablename__ = "result"
    result_id: Mapped[intpk] = mapped_column(autoincrement=True)
    fixture: Mapped["Fixture"] = relationship(back_populates="result")
    fixture_id: Mapped[int | None] = mapped_column(ForeignKey("fixture.fixture_id"))
    home_score: Mapped[int]
    away_score: Mapped[int]
    player: Mapped["Player"] = relationship(back_populates="results")
    player_id: Mapped[int | None] = mapped_column(ForeignKey("player.player_id"))

    def __str__(self):
        return (
            f"{self.fixture.season} GW{self.fixture.gameweek} "
            f"{self.fixture.home_team} {self.home_score} - "
            f"{self.away_score} {self.fixture.away_team}"
        )


class Fixture(Base):
    __tablename__ = "fixture"
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

    def __str__(self):
        return f"{self.season} GW{self.gameweek} {self.home_team} vs. {self.away_team}"


class PlayerScore(Base):
    __tablename__ = "player_score"

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
    player_id: Mapped[int | None] = mapped_column(ForeignKey("player.player_id"))
    result: Mapped["Result"] = relationship()
    result_id: Mapped[int | None] = mapped_column(ForeignKey("result.result_id"))
    fixture: Mapped["Fixture"] = relationship()
    fixture_id: Mapped[int | None] = mapped_column(ForeignKey("fixture.fixture_id"))

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

    def __str__(self):
        return f"{self.player} ({self.result}): {self.points} pts, {self.minutes} mins"


class PlayerPrediction(Base):
    __tablename__ = "player_prediction"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    fixture: Mapped["Fixture"] = relationship()
    fixture_id: Mapped[int | None] = mapped_column(ForeignKey("fixture.fixture_id"))
    predicted_points: Mapped[float]
    tag: Mapped[str100]
    player: Mapped["Player"] = relationship(back_populates="predictions")
    player_id: Mapped[int | None] = mapped_column(ForeignKey("player.player_id"))

    def __str__(self):
        return f"{self.player}: Predict {self.predicted_points} pts in {self.fixture}"


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
    fpl_team_id: Mapped[int]

    def __str__(self):
        trans_str = f"{self.season} GW{self.gameweek}: Team {self.fpl_team_id} "
        if self.bought_or_sold == 1:
            trans_str += f"bought player {self.player_id}"
        else:
            trans_str += f"sold player {self.player_id}"
        if self.free_hit:
            trans_str += " (FREE HIT)"
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

    def __str__(self):
        sugg_str = f"{self.season} GW{self.gameweek}: Suggest "
        if self.in_or_out == 1:
            sugg_str += f"buying {self.player_id} to gain {self.points_gain:.2f} pts"
        else:
            sugg_str += f"selling {self.player_id} to gain {self.points_gain:.2f} pts"
        return sugg_str


class FifaTeamRating(Base):
    __tablename__ = "fifa_rating"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    season: Mapped[str4]
    team: Mapped[str100]
    att: Mapped[int]
    defn: Mapped[int]
    mid: Mapped[int]
    ovr: Mapped[int]

    def __str__(self):
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

    def __str__(self):
        return f"{self.full_name} ({self.name})"


class SessionSquad(Base):
    __tablename__ = "sessionteam"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    session_id: Mapped[str100]
    player_id: Mapped[int]


class SessionBudget(Base):
    __tablename__ = "sessionbudget"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    session_id: Mapped[str100]
    budget: Mapped[int]


def get_connection_string() -> str:
    if AIRSENAL_DB_FILE and AIRSENAL_DB_URI:
        msg = "Please choose only ONE of AIRSENAL_DB_FILE and AIRSENAL_DB_URI"
        raise RuntimeError(msg)

    # postgres database specified by: AIRSENAL_DB{_URI, _USER, _PASSWORD}
    if AIRSENAL_DB_URI:
        if AIRSENAL_DB_PASSWORD is None:
            msg = "AIRSENAL_DB_PASSWORD must be defined when using a postgres database"
            raise KeyError(msg)
        if AIRSENAL_DB_USER is None:
            msg = "AIRSENAL_DB_USER must be defined when using a postgres database"
            raise KeyError(msg)

        return (
            f"postgresql://{AIRSENAL_DB_USER}:"
            f"{AIRSENAL_DB_PASSWORD}@{AIRSENAL_DB_URI}/airsenal"
        )

    # sqlite database in a local file with path specified by AIRSENAL_DB_FILE,
    # or AIRSENAL_HOME / data.db by default
    if not AIRSENAL_DB_FILE:
        db_file = str(AIRSENAL_HOME / "data.db")
        save_env("AIRSENAL_DB_FILE", db_file)
        return f"sqlite:///{db_file}"
    return f"sqlite:///{AIRSENAL_DB_FILE}"


def get_session():
    conn_str = get_connection_string()
    engine = create_engine(conn_str)

    Base.metadata.create_all(engine)
    # Bind the engine to the metadata of the Base class so that the
    # declaratives can be accessed through a DBSession instance
    # Note: Base.metadata.bind is deprecated in SQLAlchemy 2.0

    DBSession = sessionmaker(bind=engine, autoflush=False)
    return DBSession()


# global database session used by default throughout the package
session = get_session()


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def clean_database():
    """
    Clean up database
    """
    engine = create_engine(get_connection_string())
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def database_is_empty(dbsession):
    """
    Basic check to determine whether the database is empty
    """
    return dbsession.query(Team).first() is None
