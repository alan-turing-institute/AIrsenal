"""
Fixtures to be used in AIrsenal tests.
In particular fill a test db
"""
import random
import pytest

from ..framework.schema import *
from ..framework.mappings import *
from .resources import *


testengine = create_engine("sqlite:////tmp/test.db")

Base.metadata.create_all(testengine)

Base.metadata.bind = testengine

@contextmanager
def test_session_scope():
    """Provide a transactional scope around a series of operations."""
    db_session = sessionmaker(bind=testengine)
    testsession = db_session()
    try:
        yield testsession
        testsession.commit()
    except:
        testsession.rollback()
        raise
    finally:
        testsession.close()

@pytest.fixture
def fill_players():
    """
    fill a bunch of dummy players
    """
    team_list = list(alternative_team_names.keys())
    season = "1920"
    gw_valid_from = 1
    with test_session_scope() as ts:
        if len(ts.query(Player).all()) >0:
            return
        for i,n in enumerate(players):
            p = Player()
            p.player_id = i
            p.name = n
            print("Filling {} {}".format(i,n))
            try:
                ts.add(p)
            except:
                print("Error adding {} {}".format(i,n))
            ## now fill player_attributes
            if i % 15 < 2:
                pos = "GK"
            elif i % 15 < 7:
                pos = "DEF"
            elif i% 15 < 12:
                pos = "MID"
            else:
                pos = "FWD"
            team = team_list[i % 20]
            current_price = random.randint(40,1200)
            pa = PlayerAttributes()
            pa.season = season
            pa.team = team
            pa.gw_valid_from = gw_valid_from
            pa.current_price = current_price
            pa.position = pos
            player = ts.query(Player).filter_by(player_id=i).first()
            pa.player = player
            ts.add(pa)
