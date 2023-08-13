"""
Interface to the SQL database.
Use SQLAlchemy to convert between DB tables and python objects.
"""
from contextlib import contextmanager

from sqlalchemy import Column, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from airsenal.framework.env import AIRSENAL_HOME, get_env

Base = declarative_base()


class Player(Base):
    __tablename__ = "player"
    player_id = Column(Integer, primary_key=True, nullable=False, autoincrement=True)
    fpl_api_id = Column(Integer, nullable=True)
    name = Column(String(100), nullable=False)
    attributes = relationship("PlayerAttributes", uselist=True, back_populates="player")
    absences = relationship("Absence", uselist=True, back_populates="player")
    results = relationship("Result", uselist=True, back_populates="player")
    fixtures = relationship("Fixture", uselist=True, back_populates="player")
    predictions = relationship(
        "PlayerPrediction", uselist=True, back_populates="player"
    )
    scores = relationship("PlayerScore", uselist=True, back_populates="player")

    def team(self, season, gameweek):
        """
        Get player's team for given season and gameweek.
        If data not available for specified gameweek but data is available for
        at least one gameweek in specified season, return a best guess value
        based on data nearest to specified gameweek.
        """
        attr = self.get_gameweek_attributes(season, gameweek)
        if attr is not None:
            return attr.team
        print("No team found for", self.name, "in", season, "season.")
        return None

    def price(self, season, gameweek):
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

    def _calculate_price(self, attr, gameweek):
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

    def position(self, season):
        """
        get player's position for given season
        """
        attr = self.get_gameweek_attributes(season, None)
        if attr is not None:
            return attr.position
        print("No position found for", self.name, "in", season, "season.")
        return None

    def is_injured_or_suspended(self, season, current_gw, fixture_gw):
        """Check whether a player is injured or suspended (<=50% chance of playing).
        current_gw - The current gameweek, i.e. the gameweek when we are querying the
        player's status.
        fixture_gw - The gameweek of the fixture we want to check whether the player
        is available for, i.e. we are checking whether the player is availiable in the
        future week "fixture_gw" at the previous point in time "current_gw".
        """
        attr = self.get_gameweek_attributes(season, current_gw)
        if attr is not None:
            return (
                attr.chance_of_playing_next_round is not None
                and attr.chance_of_playing_next_round <= 50
            ) and (attr.return_gameweek is None or attr.return_gameweek > fixture_gw)
        else:
            return False

    def get_gameweek_attributes(self, season, gameweek, before_and_after=False):
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
            elif attr.gameweek == gameweek:
                return attr
            elif (attr.gameweek < gameweek) and (attr.gameweek > gw_before):
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
        elif not attr_after:
            return attr_before
        elif not attr_before:
            return attr_after
        elif before_and_after:
            return (attr_before, attr_after)
        else:
            # return attributes at gameweeek nearest to input gameweek
            if (gw_after - gameweek) >= (gameweek - gw_before):
                return attr_before
            else:
                return attr_after

    def __str__(self):
        return self.name


class PlayerMapping(Base):
    # alternative names for players
    __tablename__ = "player_mapping"
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("player.player_id"), nullable=False)
    alt_name = Column(String(100), nullable=False)


class PlayerAttributes(Base):
    __tablename__ = "player_attributes"
    id = Column(Integer, primary_key=True, autoincrement=True)
    player = relationship("Player", back_populates="attributes")
    player_id = Column(Integer, ForeignKey("player.player_id"))
    season = Column(String(100), nullable=False)
    gameweek = Column(Integer, nullable=False)
    price = Column(Integer, nullable=False)
    team = Column(String(100), nullable=False)
    position = Column(String(100), nullable=False)

    chance_of_playing_next_round = Column(Integer, nullable=True)
    news = Column(String(100), nullable=True)
    return_gameweek = Column(Integer, nullable=True)
    transfers_balance = Column(Integer, nullable=True)
    selected = Column(Integer, nullable=True)
    transfers_in = Column(Integer, nullable=True)
    transfers_out = Column(Integer, nullable=True)

    def __str__(self):
        return (
            f"{self.player} ({self.season} GW{self.gameweek}): "
            f"£{self.price / 10}, {self.team}, {self.position}"
        )


class Absence(Base):
    __tablename__ = "absence"
    id = Column(Integer, primary_key=True, autoincrement=True)
    player = relationship("Player", back_populates="absences")
    player_id = Column(Integer, ForeignKey("player.player_id"))
    season = Column(String(100), nullable=False)
    reason = Column(String(100), nullable=False)  # high-level, e.g. injury/suspension
    details = Column(String(100), nullable=True)
    date_from = Column(String(100), nullable=False)
    date_until = Column(String(100), nullable=True)
    gw_from = Column(Integer, nullable=False)
    gw_until = Column(Integer, nullable=True)
    url = Column(String(100), nullable=True)
    timestamp = Column(String(100), nullable=False)

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
    result_id = Column(Integer, primary_key=True, autoincrement=True)
    fixture = relationship("Fixture", uselist=False, back_populates="result")
    fixture_id = Column(Integer, ForeignKey("fixture.fixture_id"))
    home_score = Column(Integer, nullable=False)
    away_score = Column(Integer, nullable=False)
    player = relationship("Player", back_populates="results")
    player_id = Column(Integer, ForeignKey("player.player_id"))

    def __str__(self):
        return (
            f"{self.fixture.season} GW{self.fixture.gameweek} "
            f"{self.fixture.home_team} {self.home_score} - "
            f"{self.away_score} {self.fixture.away_team}"
        )


class Fixture(Base):
    __tablename__ = "fixture"
    fixture_id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(100), nullable=True)  # In case fixture not yet scheduled!
    gameweek = Column(Integer, nullable=True)  # In case fixture not yet scheduled!
    home_team = Column(String(100), nullable=False)
    away_team = Column(String(100), nullable=False)
    season = Column(String(100), nullable=False)
    tag = Column(String(100), nullable=False)
    result = relationship("Result", uselist=False, back_populates="fixture")
    player = relationship("Player", back_populates="fixtures")
    player_id = Column(Integer, ForeignKey("player.player_id"))

    def __str__(self):
        return (
            f"{self.season} GW{self.gameweek} " f"{self.home_team} vs. {self.away_team}"
        )


class PlayerScore(Base):
    __tablename__ = "player_score"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_team = Column(String(100), nullable=False)
    opponent = Column(String(100), nullable=False)
    points = Column(Integer, nullable=False)
    goals = Column(Integer, nullable=False)
    assists = Column(Integer, nullable=False)
    bonus = Column(Integer, nullable=False)
    conceded = Column(Integer, nullable=False)
    minutes = Column(Integer, nullable=False)
    player = relationship("Player", back_populates="scores")
    player_id = Column(Integer, ForeignKey("player.player_id"))
    result = relationship("Result", uselist=False)
    result_id = Column(Integer, ForeignKey("result.result_id"))
    fixture = relationship("Fixture", uselist=False)
    fixture_id = Column(Integer, ForeignKey("fixture.fixture_id"))

    # extended features
    clean_sheets = Column(Integer, nullable=True)
    own_goals = Column(Integer, nullable=True)
    penalties_saved = Column(Integer, nullable=True)
    penalties_missed = Column(Integer, nullable=True)
    yellow_cards = Column(Integer, nullable=True)
    red_cards = Column(Integer, nullable=True)
    saves = Column(Integer, nullable=True)
    bps = Column(Integer, nullable=True)
    influence = Column(Float, nullable=True)
    creativity = Column(Float, nullable=True)
    threat = Column(Float, nullable=True)
    ict_index = Column(Float, nullable=True)
    expected_goals = Column(Float, nullable=True)
    expected_assists = Column(Float, nullable=True)
    expected_goal_involvements = Column(Float, nullable=True)
    expected_goals_conceded = Column(Float, nullable=True)

    def __str__(self):
        return (
            f"{self.player} ({self.result}): " f"{self.points} pts, {self.minutes} mins"
        )


class PlayerPrediction(Base):
    __tablename__ = "player_prediction"
    id = Column(Integer, primary_key=True, autoincrement=True)
    fixture = relationship("Fixture", uselist=False)
    fixture_id = Column(Integer, ForeignKey("fixture.fixture_id"))
    predicted_points = Column(Float, nullable=False)
    tag = Column(String(100), nullable=False)
    player = relationship("Player", back_populates="predictions")
    player_id = Column(Integer, ForeignKey("player.player_id"))

    def __str__(self):
        return f"{self.player}: Predict {self.predicted_points} pts in {self.fixture}"


class Transaction(Base):
    __tablename__ = "transaction"
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, nullable=False)
    gameweek = Column(Integer, nullable=False)
    bought_or_sold = Column(Integer, nullable=False)  # +1 for bought, -1 for sold
    season = Column(String(100), nullable=False)
    time = Column(String(100), nullable=False)
    tag = Column(String(100), nullable=False)
    price = Column(Integer, nullable=False)
    free_hit = Column(Integer, nullable=False)  # 1 if transfer on Free Hit, 0 otherwise
    fpl_team_id = Column(Integer, nullable=False)

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
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, nullable=False)
    in_or_out = Column(Integer, nullable=False)  # +1 for buy, -1 for sell
    gameweek = Column(Integer, nullable=False)
    points_gain = Column(Float, nullable=False)
    timestamp = Column(String(100), nullable=False)  # use this to group suggestions
    season = Column(String(100), nullable=False)
    fpl_team_id = Column(
        Integer, nullable=False
    )  # to identify team to apply transfers.
    chip_played = Column(String(100), nullable=True)

    def __str__(self):
        sugg_str = f"{self.season} GW{self.gameweek}: Suggest "
        if self.in_or_out == 1:
            sugg_str += f"buying {self.player_id} to gain {self.points_gain:.2f} pts"
        else:
            sugg_str += f"selling {self.player_id} to gain {self.points_gain:.2f} pts"
        return sugg_str


class FifaTeamRating(Base):
    __tablename__ = "fifa_rating"
    id = Column(Integer, primary_key=True, autoincrement=True)
    season = Column(String(4), nullable=False)
    team = Column(String(100), nullable=False)
    att = Column(Integer, nullable=False)
    defn = Column(Integer, nullable=False)
    mid = Column(Integer, nullable=False)
    ovr = Column(Integer, nullable=False)

    def __str__(self):
        return (
            f"{self.team} {self.season} FIFA rating: "
            f"ovr {self.ovr}, def {self.defn}, mid {self.mid}, att {self.att}"
        )


class Team(Base):
    __tablename__ = "team"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(3), nullable=False)
    full_name = Column(String(100), nullable=False)
    season = Column(String(4), nullable=False)
    team_id = Column(
        Integer, nullable=False
    )  # the season-dependent team ID (from alphabetical order)

    def __str__(self):
        return f"{self.full_name} ({self.name})"


class SessionSquad(Base):
    __tablename__ = "sessionteam"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False)
    player_id = Column(Integer, nullable=False)


class SessionBudget(Base):
    __tablename__ = "sessionbudget"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False)
    budget = Column(Integer, nullable=False)


def get_connection_string():
    if get_env("AIRSENAL_DB_FILE") and get_env("AIRSENAL_DB_URI"):
        raise RuntimeError(
            "Please choose only ONE of AIRSENAL_DB_FILE and AIRSENAL_DB_URI"
        )

    # postgres database specified by: AIRSENAL_DB{_URI, _USER, _PASSWORD}
    if get_env("AIRSENAL_DB_URI"):
        keys = ["AIRSENAL_DB_URI", "AIRSENAL_DB_USER", "AIRSENAL_DB_PASSWORD"]
        params = {}
        for k in keys:
            if value := get_env(k):
                params[k] = value
            else:
                raise KeyError(f"{k} must be defined when using a postgres database")

        return (
            f"postgresql://{params['AIRSENAL_DB_USER']}:"
            f"{params['AIRSENAL_DB_PASSWORD']}@{params['AIRSENAL_DB_URI']}/airsenal"
        )

    # sqlite database in a local file with path specified by AIRSENAL_DB_FILE,
    # or AIRSENAL_HOME / data.db by default
    return f"sqlite:///{get_env('AIRSENAL_DB_FILE', default=AIRSENAL_HOME / 'data.db')}"


def get_session():
    engine = create_engine(get_connection_string())

    Base.metadata.create_all(engine)
    # Bind the engine to the metadata of the Base class so that the
    # declaratives can be accessed through a DBSession instance
    Base.metadata.bind = engine

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
