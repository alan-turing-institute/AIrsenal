import os
import random
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkdtemp

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

# Typer forces rich's terminal/color output on when GITHUB_ACTIONS is set (so CLI
# help looks nice in workflow logs), which injects ANSI escape codes into
# captured stdout and breaks plain substring assertions against CLI output.
# Disable that forcing so tests behave the same locally and on GitHub Actions.
os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"


from airsenal.core import env
from airsenal.core.output import get_logger

logger = get_logger(__name__)

env.AIRSENAL_HOME = Path(mkdtemp())
# AIRSENAL_DB_FILE/URI/USER/PASSWORD are resolved once, at env.py import time, from
# whatever real AIRSENAL_HOME/env vars are set on the machine running the tests -
# overriding AIRSENAL_HOME above does not change them. Reset them here too, so
# schema.py (imported below) can never bind its default session to a real,
# already-persisted database instead of a fresh one under the temp AIRSENAL_HOME.
env.AIRSENAL_DB_FILE = None
env.AIRSENAL_DB_URI = None
env.AIRSENAL_DB_USER = None
env.AIRSENAL_DB_PASSWORD = None

from airsenal.domain.mappings import alternative_team_names  # noqa: E402
from airsenal.framework.schema import Base, Player, PlayerAttributes  # noqa: E402
from airsenal.framework.utils import (  # noqa: E402
    CURRENT_SEASON,
    set_next_gameweek,
)

# The dummy test database has players but no fixtures, so the next gameweek cannot be
# derived from it. It used to come out as 1 anyway, because utils computed
# NEXT_GAMEWEEK at import and fell back to a live FPL API call that returned 1 for an
# empty database. Pin it explicitly instead: same value, no network, no import-time
# side effect.
set_next_gameweek(1)
from tests.test_resources import dummy_players  # noqa: E402

API_SESSION_ID = "TESTSESSION"
TEST_PAST_SEASON = "2021"

testengine_dummy = create_engine(f"sqlite:///{env.AIRSENAL_HOME}/test.db")

testengine_past = create_engine(
    f"sqlite:///{os.path.dirname(__file__)}/data/testdata_1718_1819.db"
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
        if len(ts.scalars(select(Player)).all()) > 0:
            return
        for i, n in enumerate(dummy_players):
            p = Player()
            p.player_id = i
            p.fpl_api_id = i
            p.name = n
            logger.debug("Filling %d %s", i, n)
            try:
                ts.add(p)
            except Exception:
                logger.exception("Error adding %d %s", i, n)
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
            player = ts.scalars(
                select(Player).where(Player.player_id == i).limit(1)
            ).first()
            pa.player = player
            ts.add(pa)
        ts.commit()
