"""
Recording the gameweeks the FPL API says a player will miss.

The API only lists risks it still considers live, so what it says replaces the
table from the next gameweek on, and what it has stopped listing before then stays.
"""

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from airsenal.db.models import Base, Player, PlayerRisk
from airsenal.ingest import player_risks as risks_module
from airsenal.ingest.player_risks import fill_player_risks_from_api

SEASON = "2627"
NEXT_GAMEWEEK = 10


def _risk(gameweek, prop="loan_ineligible"):
    return {"property": prop, "notes": f"Out in {gameweek}", "gameweek": gameweek}


class FakeFetcher:
    def __init__(self, risks_by_api_id):
        self._risks = risks_by_api_id

    def get_player_summary_data(self):
        return {api_id: {"scout_risks": risks} for api_id, risks in self._risks.items()}


@pytest.fixture
def dbsession(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False)()
    for api_id in (101, 102):
        session.add(Player(player_id=api_id - 100, fpl_api_id=api_id, name=str(api_id)))
    session.commit()
    monkeypatch.setattr(risks_module, "next_gameweek", lambda *a, **k: NEXT_GAMEWEEK)
    yield session
    session.close()


def _use_api(monkeypatch, risks_by_api_id):
    monkeypatch.setattr(
        risks_module, "get_fetcher", lambda: FakeFetcher(risks_by_api_id)
    )


def _recorded(dbsession):
    return sorted(
        (row.player_id, row.gameweek, row.property)
        for row in dbsession.scalars(select(PlayerRisk))
    )


def test_each_risk_is_recorded_against_its_player(dbsession, monkeypatch):
    _use_api(monkeypatch, {101: [_risk(12), _risk(30)], 102: [], 999: [_risk(15)]})
    fill_player_risks_from_api(SEASON, dbsession=dbsession)
    assert _recorded(dbsession) == [
        (1, 12, "loan_ineligible"),
        (1, 30, "loan_ineligible"),
    ]
    note = dbsession.scalars(select(PlayerRisk.notes)).first()
    assert note == "Out in 12"


def test_a_second_update_does_not_duplicate(dbsession, monkeypatch):
    _use_api(monkeypatch, {101: [_risk(12)]})
    fill_player_risks_from_api(SEASON, dbsession=dbsession)
    fill_player_risks_from_api(SEASON, dbsession=dbsession)
    assert _recorded(dbsession) == [(1, 12, "loan_ineligible")]


def test_a_withdrawn_future_risk_goes_and_a_passed_one_stays(dbsession, monkeypatch):
    dbsession.add_all(
        [
            PlayerRisk(
                player_id=1, season=SEASON, gameweek=4, property="loan_ineligible"
            ),
            PlayerRisk(
                player_id=1, season=SEASON, gameweek=20, property="loan_ineligible"
            ),
        ]
    )
    dbsession.commit()
    _use_api(monkeypatch, {102: [_risk(NEXT_GAMEWEEK)]})
    fill_player_risks_from_api(SEASON, dbsession=dbsession)
    assert _recorded(dbsession) == [
        (1, 4, "loan_ineligible"),
        (2, NEXT_GAMEWEEK, "loan_ineligible"),
    ]
