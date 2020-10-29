import os
import random
from contextlib import contextmanager

import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from airsenal.framework.mappings import alternative_team_names
from airsenal.tests.resources import dummy_players
from airsenal.framework.schema import Base, Player, PlayerAttributes
from airsenal.framework.utils import CURRENT_SEASON
from airsenal import TMPDIR

API_SESSION_ID = "TESTSESSION"

testengine_dummy = create_engine("sqlite:///{}/test.db".format(TMPDIR))
# testengine_past = create_engine("sqlite:////Users/nbarlow/AIrsenal/airsenal/tests/testdata/testdata_1718_1819.db")
#    .format(os.path.dirname(__file__)))

testengine_past = create_engine(
    "sqlite:///{}/tests/testdata/testdata_1718_1819.db".format(
        os.path.dirname(__file__)
    )
)

Base.metadata.create_all(testengine_dummy)

Base.metadata.bind = testengine_dummy


@contextmanager
def test_session_scope():
    """Provide a transactional scope around a series of operations."""
    db_session = sessionmaker(bind=testengine_dummy)
    testsession = db_session()
    try:
        yield testsession
        testsession.commit()
    except:
        testsession.rollback()
        raise
    finally:
        testsession.close()


@contextmanager
def test_past_data_session_scope():
    """Provide a transactional scope around a series of operations."""
    db_session = sessionmaker(bind=testengine_past)
    testsession = db_session()
    try:
        yield testsession
        testsession.commit()
    except:
        testsession.rollback()
        raise
    finally:
        testsession.close()


def value_generator(index, position):
    """
    make up a price for a dummy player, based on index and position
    """
    if position == "GK":
        value = 40 + index * random.randint(0, 5)
    elif position == "DEF":
        value = 40 + index * random.randint(5, 10)
    elif position == "MID":
        value = 50 + index * random.randint(10, 20)
    elif position == "FWD":
        value = 60 + index * random.randint(15, 20)
    return value


@pytest.fixture(scope="session")
def fill_players():
    """
    fill a bunch of dummy players
    """
    team_list = list(alternative_team_names.keys())
    season = CURRENT_SEASON
    gameweek = 1
    with test_session_scope() as ts:
        if len(ts.query(Player).all()) > 0:
            return
        for i, n in enumerate(dummy_players):
            p = Player()
            p.player_id = i
            p.fpl_api_id = i
            p.name = n
            print("Filling {} {}".format(i, n))
            try:
                ts.add(p)
            except:
                print("Error adding {} {}".format(i, n))
            # now fill player_attributes
            if i % 15 < 2:
                pos = "GK"
            elif i % 15 < 7:
                pos = "DEF"
            elif i % 15 < 12:
                pos = "MID"
            else:
                pos = "FWD"
            team = team_list[i % 20]
            # make the first 15 players affordable,
            # the next 15 almost affordable,
            # the next 15 mostly unaffordable,
            # and rest very expensive
            price = value_generator(i // 15, pos)
            pa = PlayerAttributes()
            pa.season = season
            pa.team = team
            pa.gameweek = gameweek
            pa.price = price
            pa.position = pos
            player = ts.query(Player).filter_by(player_id=i).first()
            pa.player = player
            ts.add(pa)
        ts.commit()
