"""
Objects that cross a process boundary must be picklable.

The transfer optimiser runs a tree search across multiprocessing.Process workers and
passes Squad objects over a queue; utils.fastcopy also round-trips them through pickle
in the inner loop. A live SQLAlchemy Session cannot be pickled, so no object reachable
from a Squad may hold one - and resolving a session eagerly would open a database
connection per candidate player anyway.
"""

import pickle

from sqlalchemy.orm import Session

from airsenal.conftest import session_scope
from airsenal.framework.player import CandidatePlayer
from airsenal.framework.squad import Squad
from airsenal.framework.utils import fastcopy


def _session_attributes(obj, seen=None, path="squad"):
    """Every attribute reachable from obj that holds a Session."""
    seen = seen if seen is not None else set()
    if id(obj) in seen:
        return []
    seen.add(id(obj))

    found = []
    if isinstance(obj, Session):
        return [path]
    if isinstance(obj, list | tuple):
        for i, item in enumerate(obj):
            found.extend(_session_attributes(item, seen, f"{path}[{i}]"))
    elif hasattr(obj, "__dict__"):
        for name, value in vars(obj).items():
            found.extend(_session_attributes(value, seen, f"{path}.{name}"))
    return found


def test_empty_squad_is_picklable():
    squad = Squad()
    assert pickle.loads(pickle.dumps(squad)).budget == squad.budget


def test_empty_squad_survives_fastcopy():
    squad = Squad()
    assert fastcopy(squad).budget == squad.budget


def test_squad_with_players_is_picklable(fill_players):
    """
    A Squad holding CandidatePlayers is what actually goes onto the optimiser's
    queue, so nothing reachable from it may hold a Session.
    """
    with session_scope() as ts:
        squad = Squad()
        added = sum(bool(squad.add_player(i, dbsession=ts)) for i in range(20))
        assert added > 0, "could not build a squad to pickle"

        restored = pickle.loads(pickle.dumps(squad))
        assert len(restored.players) == len(squad.players)
        assert _session_attributes(restored) == []


def test_candidate_player_built_with_a_session_is_still_picklable(fill_players):
    """
    Regression test for:

        PicklingError: Can't pickle <class 'sqlalchemy.orm.session.Session'>

    Squad.add_player resolved its dbsession default to the process-wide session and
    handed it to CandidatePlayer, which stores it. Every Squad therefore contained a
    live Session, and the transfer optimiser could not put one on its queue.

    A session may legitimately be passed in for the constructor's own lookups, so the
    fix has to survive that: it is dropped on pickle rather than never held.
    """
    with session_scope() as ts:
        player = CandidatePlayer(1, dbsession=ts)
        assert player.dbsession is ts

        restored = pickle.loads(pickle.dumps(player))
        assert restored.player_id == player.player_id
        assert restored.dbsession is None
        assert _session_attributes(restored) == []


def test_add_player_does_not_put_a_session_in_the_squad(fill_players):
    """
    Squad.add_player must pass its dbsession through rather than resolving it, so a
    squad built the way the optimiser builds one carries no session at all.
    """
    with session_scope() as ts:
        squad = Squad()
        assert squad.add_player(1, dbsession=ts)

    assert (
        squad.players[0].dbsession is None
        or _session_attributes(pickle.loads(pickle.dumps(squad))) == []
    )
