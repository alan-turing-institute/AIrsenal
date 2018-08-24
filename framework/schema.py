"""
Interface to the SQLite db.
Use SQLAlchemy to convert between DB tables and python objects.
"""

from sqlalchemy import Column, ForeignKey, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from sqlalchemy import create_engine

Base = declarative_base()



class Player(Base):
    __tablename__ = "player"
    player_id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    team = Column(String(100), nullable=False)
    position = Column(String(100), nullable=False)
    current_price = Column(Integer, nullable=True)
    purchased_price = Column(Integer, nullable=True)
#    scores = relationship("PlayerScore")
#    fixtures = relationship("Fixture")

class Match(Base):
    __tablename__ = "match"
    match_id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(100), nullable=False)
    season = Column(String(100), nullable=False)
    gameweek = Column(Integer, nullable=True) # not there for 14/15 season
    home_team = Column(String(100), nullable=False)
    away_team = Column(String(100), nullable=False)
    home_score = Column(Integer, nullable=False)
    away_score = Column(Integer, nullable=False)


class Fixture(Base):
    __tablename__ = "fixture"
    fixture_id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(100), nullable=False)
    gameweek = Column(Integer, nullable=False)
    home_team = Column(String(100), nullable=False)
    away_team = Column(String(100), nullable=False)


class PlayerScore(Base):
    __tablename__ = "player_score"
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, nullable=False)
    match_id = Column(Integer, nullable=False)
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
    player_id = Column(Integer, nullable=False)
    fixture_id = Column(Integer, nullable=False)
    predicted_points = Column(Float, nullable=False)
    method = Column(String(100), nullable=False)

class Transaction(Base):
    __tablename__ = "current_team"
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, nullable=False)
    gameweek = Column(Integer, nullable=False)
    bought_or_sold = Column(Integer, nullable=False) # +1 for bought, -1 for sold


engine = create_engine('sqlite:////tmp/data.db')

Base.metadata.create_all(engine)
# Bind the engine to the metadata of the Base class so that the
# declaratives can be accessed through a DBSession instance
Base.metadata.bind = engine
