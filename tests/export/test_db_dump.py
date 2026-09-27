"""Dumping the database to CSV writes every column of every table it covers."""

import csv

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from airsenal.db.models import Base, PlayerScore
from airsenal.export import db_dump


@pytest.fixture
def dbsession():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _score(**kwargs):
    return PlayerScore(
        player_team="ARS",
        opponent="CHE",
        points=2,
        goals=0,
        assists=0,
        bonus=0,
        conceded=0,
        minutes=90,
        player_id=1,
        result_id=1,
        fixture_id=1,
        **kwargs,
    )


def test_every_table_is_written_with_all_of_its_columns(
    monkeypatch, tmp_path, dbsession
):
    """Including the columns set on no row, which come out as empty fields."""
    dbsession.add(_score(news="Knee injury", chance_of_playing=75))
    dbsession.commit()
    monkeypatch.setattr(db_dump, "data_file", lambda filename: tmp_path / filename)

    db_dump.dump_db(dbsession=dbsession)

    for filename, dbclass in db_dump.DUMP_FILES.items():
        with (tmp_path / filename).open() as f:
            header = next(csv.reader(f))
        assert header == [column.key for column in inspect(dbclass).column_attrs]

    with (tmp_path / "player_scores.csv").open() as f:
        (row,) = csv.DictReader(f)
    assert row["news"] == "Knee injury"
    assert row["chance_of_playing"] == "75"
    assert row["saves"] == ""
