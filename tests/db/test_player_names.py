"""
Looking a player up under a name spelled differently from the one on file.

Transfermarkt, the FPL API and the packaged season files disagree about how to
write the same person, so the absence ingest falls back to matching folded name
words when the exact lookup finds nothing.
"""

import itertools
from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from airsenal.core.caching import clear_query_caches
from airsenal.db.models import Base, Player, PlayerMapping
from airsenal.db.queries.players import (
    fold_name,
    get_player,
    get_player_by_similar_name,
    name_tokens,
)


@contextmanager
def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/names.db")
    Base.metadata.create_all(engine)
    dbsession = sessionmaker(bind=engine)()
    try:
        yield dbsession
    finally:
        dbsession.close()


@pytest.fixture
def named(tmp_path):
    """A database whose players are named however the test asks for."""
    # The index of names is cached on nothing but the database it was built
    # from - see core/caching.py - so whichever database ran before this one
    # would otherwise answer for it.
    clear_query_caches()
    with _session(tmp_path) as dbsession:
        next_id = itertools.count(1)

        def add(*names):
            player_id = next(next_id)
            player = Player()
            player.player_id = player_id
            player.name = names[0]
            player.display_name = names[1] if len(names) > 1 else None
            dbsession.add(player)
            for alt_name in names[2:]:
                mapping = PlayerMapping()
                mapping.player_id = player_id
                mapping.alt_name = alt_name
                dbsession.add(mapping)
            dbsession.commit()
            return player

        yield dbsession, add
    clear_query_caches()


@pytest.mark.parametrize(
    ("written", "folded"),
    [
        ("Joško Gvardiol", "josko gvardiol"),
        ("Łukasz Fabiański", "lukasz fabianski"),
        ("Albert Grønbæk", "albert gronbaek"),
        ("Takai Kōta", "takai kota"),
    ],
)
def test_fold_name_removes_accents_and_case(written, folded):
    assert fold_name(written) == folded


def test_name_tokens_splits_on_punctuation():
    """A hyphen or an apostrophe is a word break, not part of the word."""
    assert name_tokens("Mark O\u2019Mahony") == {"mark", "o", "mahony"}
    assert name_tokens("Jamie Bynoe-Gittens") == {"jamie", "bynoe", "gittens"}


@pytest.mark.parametrize(
    ("on_file", "asked_for", "why"),
    [
        ("Joško Gvardiol", "Josko Gvardiol", "the accents are dropped"),
        ("Tanaka Ao", "Ao Tanaka", "the names are the other way round"),
        (
            "Matheus Santos Carneiro da Cunha",
            "Matheus Cunha",
            "the family names in the middle are dropped",
        ),
        ("Daniel Ballard", "Dan Ballard", "the given name is shortened"),
        ("Hannibal Mejbri", "Hannibal", "only the name he plays under is given"),
        (
            "Mark O\u2019Mahony",
            "Mark O'Mahony",
            "the apostrophe is a curly one on one side",
        ),
    ],
)
def test_a_player_is_found_under_another_spelling(named, on_file, asked_for, why):
    dbsession, add = named
    player = add(on_file)

    assert get_player(asked_for, dbsession=dbsession) is None, "not an exact match"
    assert get_player_by_similar_name(asked_for, dbsession=dbsession) is player, why


def test_the_closest_match_wins(named):
    """Both fit "Rodrigo Gomes", but one of them has a word to spare."""
    dbsession, add = named
    add("Rodrigo Martins Gomes")
    add("Rodrigo Gomes Ferreira da Silva")

    assert get_player_by_similar_name("Rodrigo Gomes", dbsession=dbsession).name == (
        "Rodrigo Martins Gomes"
    )


def test_a_name_that_fits_two_players_equally_well_matches_neither(named):
    """Guessing would file one player's absence against another."""
    dbsession, add = named
    add("Danny Ward")
    add("Daniel Ward")

    assert get_player_by_similar_name("Dan Ward", dbsession=dbsession) is None


def test_a_short_given_name_does_not_stand_in_for_a_longer_one(named):
    """Two letters would make one player of everyone called Jo-something."""
    dbsession, add = named
    add("Joseph Willock")

    assert get_player_by_similar_name("Jo Willock", dbsession=dbsession) is None


def test_a_surname_on_its_own_is_not_enough(named):
    """The words asked for all have to be there; a shared surname is not a match."""
    dbsession, add = named
    add("Kieffer Moore")

    assert get_player_by_similar_name("Kobei Moore", dbsession=dbsession) is None


def test_display_names_and_alternative_names_are_searched_too(named):
    dbsession, add = named
    player = add("Carlos Henrique Casimiro", "Casemiro", "Casemiro da Silva")

    assert get_player_by_similar_name("Casemiro", dbsession=dbsession) is player
