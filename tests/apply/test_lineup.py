"""
The payload that decides which eleven, which captain and which bench order get posted.

As `test_transfers.py`: only the pure parts, and nothing here reaches the network.
`build_lineup_payload` turns a squad into the list the API wants, and
`get_lineup_from_payload` reads one back; between them they are the whole of what
`set_lineup` sends, so the numbering is asserted rather than assumed.
"""

import pytest
from sqlalchemy import select

from airsenal.apply import lineup as lineup_module
from airsenal.apply.lineup import build_lineup_payload, get_lineup_from_payload
from airsenal.db.models import Player
from airsenal.game.enums import Position
from airsenal.game.season import CURRENT_SEASON
from airsenal.squad.squad import Squad
from tests.conftest import session_scope

# The dummy database gives player_id 0-1 GK, 2-6 DEF, 7-11 MID and 12-14 FWD, which
# is one legal squad. A 4-4-2 out of it benches one of each outfield position.
SQUAD_IDS = tuple(range(15))
BENCHED_IDS = (1, 6, 11, 14)


class FakePlayer:
    """What `get_player` returns: enough of a row to be given an api id."""

    def __init__(self, player_id, fpl_api_id):
        self.player_id = player_id
        self.fpl_api_id = fpl_api_id


@pytest.fixture
def api_ids(monkeypatch):
    """
    player_id N has FPL api id 100+N, without going near a database.

    The dummy database gives a player the same api id as its player_id, which
    would hide the payload naming players by the wrong one of the two.
    """
    monkeypatch.setattr(
        lineup_module, "require_player", lambda pid: FakePlayer(pid, 100 + pid)
    )


def lineup_squad(dbsession, sub_positions=(0, 1, 2, 3), captain=0, vice_captain=2):
    """
    A complete squad with the lineup already chosen.

    `sub_positions` is the bench order, given for `BENCHED_IDS` in that order -
    `order_substitutes` numbers all four subs, the reserve keeper included.
    """
    squad = Squad(season=CURRENT_SEASON)
    for player_id in SQUAD_IDS:
        squad.add_player(
            player_id, check_budget=False, check_team=False, dbsession=dbsession
        )
    bench_order = dict(zip(BENCHED_IDS, sub_positions, strict=True))
    for player in squad.players:
        player.is_starting = player.player_id not in bench_order
        player.sub_position = bench_order.get(player.player_id)
        player.is_captain = player.player_id == captain
        player.is_vice_captain = player.player_id == vice_captain
    return squad


def by_slot(payload):
    """{position slot: entry}, which is how the API reads the list."""
    return {entry["position"]: entry for entry in payload}


# ----------------------------------------------------------------- numbering ---


def test_the_payload_numbers_all_fifteen_slots_once(fill_players, api_ids):
    with session_scope() as ts:
        payload = build_lineup_payload(lineup_squad(ts))
    assert sorted(entry["position"] for entry in payload) == list(range(1, 16))


def test_the_starting_eleven_take_the_first_eleven_slots(fill_players, api_ids):
    """Slots 1-11 are the eleven who play; 12-15 are the bench."""
    with session_scope() as ts:
        squad = lineup_squad(ts)
        starting = {p.player_id for p in squad.players if p.is_starting}
        payload = build_lineup_payload(squad)
    slots = by_slot(payload)
    assert {slots[slot]["element"] - 100 for slot in range(1, 12)} == starting


def test_the_starting_eleven_are_ordered_back_to_front(fill_players, api_ids):
    """The API reads the eleven positionally, so a keeper cannot follow a forward."""
    with session_scope() as ts:
        squad = lineup_squad(ts)
        position_of = {p.player_id: p.position for p in squad.players}
        payload = build_lineup_payload(squad)
    slots = by_slot(payload)
    positions = [position_of[slots[slot]["element"] - 100] for slot in range(1, 12)]
    assert positions == [
        Position.GK,
        *[Position.DEF] * 4,
        *[Position.MID] * 4,
        *[Position.FWD] * 2,
    ]


def test_the_reserve_keeper_is_always_slot_twelve(fill_players, api_ids):
    """
    Slot 12 is the keeper's, whatever the bench order says.

    FPL substitutes a keeper only for a keeper, so the reserve one does not
    compete with the outfield bench for a slot - it has its own.
    """
    with session_scope() as ts:
        # bench the keeper last on points and it still takes slot 12
        payload = build_lineup_payload(lineup_squad(ts, sub_positions=(3, 0, 1, 2)))
    assert by_slot(payload)[12]["element"] == 101


def test_the_outfield_bench_keeps_its_own_order(fill_players, api_ids):
    """
    Slots 13-15 follow `sub_position` among the outfield subs.

    The keeper is numbered alongside them by `order_substitutes` but does not take
    one of their slots, so the three that are left close up rather than leaving a
    gap where the keeper's number was.
    """
    with session_scope() as ts:
        # keeper second on points, so the outfield subs hold 0, 2 and 3
        payload = build_lineup_payload(lineup_squad(ts, sub_positions=(1, 3, 0, 2)))
    slots = by_slot(payload)
    assert [slots[slot]["element"] for slot in (13, 14, 15)] == [111, 114, 106]


def test_a_bench_ordered_the_other_way_round_reverses_the_slots(fill_players, api_ids):
    with session_scope() as ts:
        payload = build_lineup_payload(lineup_squad(ts, sub_positions=(0, 3, 2, 1)))
    slots = by_slot(payload)
    assert [slots[slot]["element"] for slot in (13, 14, 15)] == [114, 111, 106]


# -------------------------------------------------------------------- fields ---


def test_the_payload_names_players_by_their_fpl_api_id(fill_players, api_ids):
    """Not by this database's player_id, which the API knows nothing about."""
    with session_scope() as ts:
        payload = build_lineup_payload(lineup_squad(ts))
    assert {entry["element"] for entry in payload} == {100 + i for i in SQUAD_IDS}


def test_exactly_one_captain_and_one_vice_captain_are_marked(fill_players, api_ids):
    with session_scope() as ts:
        payload = build_lineup_payload(lineup_squad(ts, captain=3, vice_captain=8))
    captains = [e["element"] for e in payload if e["is_captain"]]
    vice_captains = [e["element"] for e in payload if e["is_vice_captain"]]
    assert captains == [103]
    assert vice_captains == [108]


def test_a_player_without_an_api_id_is_an_error(fill_players, monkeypatch):
    """The payload has nothing to say about a player the API cannot be told about."""
    monkeypatch.setattr(
        lineup_module, "require_player", lambda pid: FakePlayer(pid, None)
    )
    with session_scope() as ts, pytest.raises(ValueError, match="no FPL API ID"):
        build_lineup_payload(lineup_squad(ts))


def test_a_bench_that_was_never_ordered_is_an_error(fill_players, api_ids):
    """`sub_position` is set by `optimize_lineup`; without it there is no bench."""
    with session_scope() as ts:
        squad = lineup_squad(ts)
        for player in squad.players:
            player.sub_position = None
        with pytest.raises(RuntimeError, match="no bench position"):
            build_lineup_payload(squad)


# ------------------------------------------------------------- reading it back ---


def picks(player_ids=SQUAD_IDS):
    """A `get_lineup` response, cut down to what `get_lineup_from_payload` reads."""
    return {"picks": [{"element": 100 + player_id} for player_id in player_ids]}


@pytest.fixture
def api_id_lookup(monkeypatch):
    """`get_player_from_api_id` against the test database, keyed as 100+player_id."""
    with session_scope() as ts:
        monkeypatch.setattr(
            lineup_module,
            "require_player_from_api_id",
            lambda api_id: ts.scalars(
                select(Player).where(Player.player_id == api_id - 100).limit(1)
            ).one(),
        )
        yield


def test_a_full_set_of_picks_becomes_a_complete_squad(fill_players, api_id_lookup):
    squad = get_lineup_from_payload(picks())
    assert squad.is_complete()
    assert {p.player_id for p in squad.players} == set(SQUAD_IDS)


def test_a_short_set_of_picks_is_an_error(fill_players, api_id_lookup):
    """Fourteen players is not a squad, and posting one back would not be either."""
    with pytest.raises(RuntimeError, match="Squad incomplete"):
        get_lineup_from_payload(picks(player_ids=SQUAD_IDS[:-1]))
