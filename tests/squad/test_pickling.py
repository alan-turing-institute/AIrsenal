"""
Objects that cross a process boundary must be picklable.

The transfer optimiser passes Squad objects to multiprocessing workers over a queue,
and `core.copy.fastcopy` round-trips them through pickle in the inner loop. A live
SQLAlchemy Session cannot be pickled, so no object reachable from a Squad may hold one -
and resolving a session eagerly would open a database connection per candidate player
anyway.
"""

import pickle

from sqlalchemy.orm import Session

from airsenal.core.copy import fastcopy
from airsenal.squad.player import CandidatePlayer
from airsenal.squad.squad import Squad
from tests.conftest import session_scope


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
    Nothing reachable from a Squad may hold a Session.

    A Squad holding CandidatePlayers is what actually goes onto the optimiser's
    queue.
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
    A CandidatePlayer holding a Session must still pickle.

    A session is legitimately passed in for the constructor's own lookups, so the
    player cannot simply refuse to hold one; it drops it on pickle instead.
    Without that, every Squad contains a live Session and the transfer optimiser
    cannot put one on its queue.
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
    `Squad.add_player` passes its dbsession through rather than resolving it.

    So a squad built the way the optimiser builds one carries no session at all.
    """
    with session_scope() as ts:
        squad = Squad()
        assert squad.add_player(1, dbsession=ts)

    assert (
        squad.players[0].dbsession is None
        or _session_attributes(pickle.loads(pickle.dumps(squad))) == []
    )
