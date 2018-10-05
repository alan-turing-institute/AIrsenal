"""
Interface to the SQLite db.
Use SQLAlchemy to convert between DB tables and python objects.
"""
## location of sqlite file - default is /tmp/data.db, unless
## overridden by an env var

db_location = "/tmp/data.db"
import os

if "AIrsenalDB" in os.environ.keys():
    db_location = os.environ["AIrsenalDB"]


from sqlalchemy import Column, ForeignKey, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm import sessionmaker

from sqlalchemy import create_engine

from contextlib import contextmanager

Base = declarative_base()


class Player(Base):
    __tablename__ = "player"
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, nullable=False)
    name = Column(String(100), nullable=False)
    team = Column(String(100), nullable=False)
    position = Column(String(100), nullable=False)
    current_price = Column(Integer, nullable=True)
    purchased_price = Column(Integer, nullable=True)
    season = Column(String(100), nullable=False)

#    scores = relationship("PlayerScore")
#    fixtures = relationship("Fixture")


class Result(Base):
    __tablename__ = "result"
    result_id = Column(Integer, primary_key=True, autoincrement=True)
    fixture = relationship("Fixture", uselist=False, back_populates="result")
    fixture_id = Column(Integer, ForeignKey('fixture.fixture_id'))
    home_score = Column(Integer, nullable=False)
    away_score = Column(Integer, nullable=False)


class Fixture(Base):
    __tablename__ = "fixture"
    fixture_id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(100), nullable=False)
    gameweek = Column(Integer, nullable=False)
    home_team = Column(String(100), nullable=False)
    away_team = Column(String(100), nullable=False)
    season = Column(String(100), nullable=False)
    tag = Column(String(100), nullable=False)
    result = relationship("Result", uselist=False, back_populates="fixture")
#    result_id = Column(Integer, ForeignKey('result.result_id'))

class PlayerScore(Base):
    __tablename__ = "player_score"
    id = Column(Integer, primary_key=True, autoincrement=True)
    player = relationship("Player", uselist=False)
    player_id = Column(Integer, ForeignKey('player.player_id'))
    result = relationship("Result", uselist=False)
    result_id = Column(Integer, ForeignKey('result.result_id'))
    fixture = relationship("Fixture", uselist=False)
    fixture_id = Column(Integer, ForeignKey('fixture.fixture_id'))
    player_team = Column(String(100), nullable=False)
    opponent = Column(String(100), nullable=False)
    points = Column(Integer, nullable=False)
    goals = Column(Integer, nullable=False)
    assists = Column(Integer, nullable=False)
    bonus = Column(Integer, nullable=False)
    conceded = Column(Integer, nullable=False)
    minutes = Column(Integer, nullable=False)


class PlayerPrediction(Base):
    __tablename__ = "player_prediction"
    id = Column(Integer, primary_key=True, autoincrement=True)
    player = relationship("Player", uselist=False)
    player_id = Column(Integer, ForeignKey('player.player_id'))
    fixture = relationship("Fixture", uselist=False)
    fixture_id = Column(Integer, ForeignKey('fixture.fixture_id'))
    predicted_points = Column(Float, nullable=False)
    tag = Column(String(100), nullable=False)


class Transaction(Base):
    __tablename__ = "transaction"
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, nullable=False)
    gameweek = Column(Integer, nullable=False)
    bought_or_sold = Column(Integer, nullable=False)  # +1 for bought, -1 for sold
    season = Column(String(100), nullable=False)


class TransferSuggestion(Base):
    __tablename__ = "transfer_suggestion"
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, nullable=False)
    in_or_out = Column(Integer, nullable=False)  # +1 for buy, -1 for sell
    gameweek = Column(Integer, nullable=False)
    points_gain = Column(Float, nullable=False)
    timestamp = Column(String(100), nullable=False)  # use this to group suggestions
    season = Column(String(100), nullable=False)


class FifaTeamRating(Base):
    __tablename__ = "fifa_rating"
    team = Column(String(100), nullable=False, primary_key=True)
    att = Column(Integer, nullable=False)
    defn = Column(Integer, nullable=False)
    mid = Column(Integer, nullable=False)
    ovr = Column(Integer, nullable=False)


engine = create_engine("sqlite:///{}".format(db_location))

Base.metadata.create_all(engine)
# Bind the engine to the metadata of the Base class so that the
# declaratives can be accessed through a DBSession instance
Base.metadata.bind = engine

@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    db_session = sessionmaker(bind=engine)
    session = db_session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()
