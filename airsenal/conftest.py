import os
import random
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkdtemp

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from airsenal.framework import env

env.AIRSENAL_HOME = Path(mkdtemp())

from airsenal.framework.mappings import alternative_team_names  # noqa: E402
from airsenal.framework.schema import Base, Player, PlayerAttributes  # noqa: E402
from airsenal.framework.utils import CURRENT_SEASON  # noqa: E402
from airsenal.tests.resources import dummy_players  # noqa: E402

API_SESSION_ID = "TESTSESSION"
TEST_PAST_SEASON = "2021"

testengine_dummy = create_engine(f"sqlite:///{env.AIRSENAL_HOME}/test.db")

testengine_past = create_engine(
    f"sqlite:///{os.path.dirname(__file__)}/tests/testdata/testdata_1718_1819.db"
)


Base.metadata.create_all(testengine_dummy)

Base.metadata.bind = testengine_dummy


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    db_session = sessionmaker(bind=testengine_dummy)
    testsession = db_session()
    try:
        yield testsession
        testsession.commit()
    except Exception:
        testsession.rollback()
        raise
    finally:
        testsession.close()


@contextmanager
def past_data_session_scope():
    """Provide a transactional scope around a series of operations."""
    db_session = sessionmaker(bind=testengine_past)
    testsession = db_session()
    try:
        yield testsession
        testsession.commit()
    except Exception:
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
    with session_scope() as ts:
        if len(ts.query(Player).all()) > 0:
            return
        for i, n in enumerate(dummy_players):
            p = Player()
            p.player_id = i
            p.fpl_api_id = i
            p.name = n
            print(f"Filling {i} {n}")
            try:
                ts.add(p)
            except Exception:
                print(f"Error adding {i} {n}")
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
