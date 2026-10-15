"""Replaying a plan's transfers to find the price each was made at."""

import pytest

from airsenal.game.enums import Chip
from airsenal.game.season import CURRENT_SEASON
from airsenal.optimization import run_transfers as rt
from airsenal.optimization.moves import ChipGameweeks, GameweekChips, GameweekMove
from airsenal.optimization.plan import (
    GameweekOutcome,
    Plan,
    TransferSearchResult,
)
from airsenal.optimization.run_transfers import run_optimization, transfer_rows
from airsenal.squad.squad import Squad
from tests.conftest import session_scope

STARTING_IDS = tuple(range(15))
FREE_HIT_IDS = tuple(range(15, 30))


def _squad(dbsession):
    squad = Squad(season=CURRENT_SEASON)
    for player_id in STARTING_IDS:
        squad.add_player(
            player_id, check_budget=False, check_team=False, dbsession=dbsession
        )
    return squad


def _outcome(gameweek, move, players_out, players_in):
    return GameweekOutcome(
        gameweek=gameweek,
        move=move,
        points=0.0,
        discount_factor=1.0,
        points_hit=0,
        free_transfers=1,
        players_in=players_in,
        players_out=players_out,
    )


def test_transfer_rows_reverts_a_free_hit(fill_players):
    """
    A free hit is undone before the next gameweek's transfers are priced.

    The search plans the gameweek after a free hit from the pre-free-hit squad
    (`GameweekMove.carry_forward`), so the walk has to do the same - otherwise it
    is asked to sell a player the free-hit squad does not hold.
    """
    with session_scope() as ts:
        plan = Plan(
            root_gameweek=1,
            outcomes=(
                _outcome(
                    1,
                    GameweekMove(chip=Chip.FREE_HIT),
                    STARTING_IDS,
                    FREE_HIT_IDS,
                ),
                _outcome(2, GameweekMove(1), (0,), (30,)),
            ),
        )
        rows = transfer_rows(
            plan, _squad(ts), season=CURRENT_SEASON, use_api=False, dbsession=ts
        )

    # 15 for the free hit, then the one real transfer made off the original squad
    assert len(rows) == 16
    assert rows[-1].gameweek == 2
    assert rows[-1].sale_price is not None


def test_transfer_rows_carries_a_wildcard_forward(fill_players):
    """A wildcard is kept, so the next gameweek transfers out of the new squad."""
    with session_scope() as ts:
        plan = Plan(
            root_gameweek=1,
            outcomes=(
                _outcome(
                    1,
                    GameweekMove(chip=Chip.WILDCARD),
                    STARTING_IDS,
                    FREE_HIT_IDS,
                ),
                _outcome(2, GameweekMove(1), (15,), (30,)),
            ),
        )
        rows = transfer_rows(
            plan, _squad(ts), season=CURRENT_SEASON, use_api=False, dbsession=ts
        )

    assert len(rows) == 16
    assert rows[-1].gameweek == 2


class RecordingOptimizer:
    """Returns a fixed result, so a test can drive run_optimization without a search."""

    def __init__(self, plan: Plan) -> None:
        self.result = TransferSearchResult(best=plan, baseline=plan)
        self.requests = []

    def search(self, request) -> TransferSearchResult:
        self.requests.append(request)
        return self.result


def _stub_reporting(monkeypatch, posted, dbsession):
    """Everything run_optimization does with a result except decide whether to post."""
    monkeypatch.setattr(rt, "get_fetcher", lambda *a, **k: None)
    monkeypatch.setattr(rt, "get_starting_squad", lambda *a, **k: _squad(dbsession))
    monkeypatch.setattr(rt, "get_free_transfers", lambda *a, **k: 1)
    monkeypatch.setattr(rt, "fill_suggestion_table", lambda *a, **k: None)
    monkeypatch.setattr(rt, "fill_transaction_table", lambda *a, **k: None)
    monkeypatch.setattr(rt, "transfer_rows", lambda *a, **k: [])
    monkeypatch.setattr(
        rt, "squad_for_next_gameweek", lambda *a, **k: _squad(dbsession)
    )
    monkeypatch.setattr(rt, "formation_table", lambda *a, **k: "")
    monkeypatch.setattr(rt, "lineup_strings", lambda *a, **k: [])
    monkeypatch.setattr(rt, "discord_payload", lambda *a, **k: {})
    monkeypatch.setattr(rt, "print_result_panel", lambda *a, **k: None)
    monkeypatch.setattr(rt, "print_plan_table", lambda *a, **k: None)
    monkeypatch.setattr(rt, "print_transfer_table", lambda *a, **k: None)
    monkeypatch.setattr(rt, "post_webhook", lambda *a, **k: posted.append(True) or True)


@pytest.mark.parametrize(("is_replay", "n_posts"), [(False, 1), (True, 0)])
def test_only_a_real_run_posts_to_discord(
    monkeypatch, fill_players, is_replay, n_posts
):
    """
    A replay does not announce its transfers to the Discord channel.

    It optimises every gameweek of a past season, so posting would send a
    season's worth of transfers - times `--loop` - for a season nobody is playing.
    """
    plan = Plan(root_gameweek=1, outcomes=(_outcome(1, GameweekMove(1), (0,), (30,)),))
    posted: list[bool] = []
    with session_scope() as ts:
        _stub_reporting(monkeypatch, posted, ts)
        run_optimization(
            gameweeks=[1],
            tag="test_replay_webhook",
            season=CURRENT_SEASON,
            fpl_team_id=4321,
            optimizer=RecordingOptimizer(plan),
            is_replay=is_replay,
        )

    assert len(posted) == n_posts


def test_the_search_plays_the_chips_the_heuristic_decides(monkeypatch, fill_players):
    """The chips the heuristic picks are pinned, and the ones played are kept."""
    decided = []

    def decide_chips(squad, available, gameweeks, tag, season=None):
        decided.append(available)
        return [(1, None, ""), (2, Chip.FREE_HIT, "a blank")]

    monkeypatch.setattr(rt, "decide_chips", decide_chips)
    plan = Plan(root_gameweek=1, outcomes=(_outcome(1, GameweekMove(1), (0,), (30,)),))
    optimizer = RecordingOptimizer(plan)
    played = ((1, Chip.WILDCARD),)
    with session_scope() as ts:
        _stub_reporting(monkeypatch, [], ts)
        run_optimization(
            gameweeks=[1, 2],
            tag="test_chip_heuristic",
            season="2526",
            fpl_team_id=4321,
            chips=ChipGameweeks(free_hit=0, played=played, heuristic=True),
            optimizer=optimizer,
            is_replay=True,
        )

    assert decided == [frozenset(Chip) - {Chip.WILDCARD}]
    (request,) = optimizer.requests
    assert request.chip_schedule.for_gameweek(1) == GameweekChips()
    assert request.chip_schedule.for_gameweek(2) == GameweekChips(
        chip_to_play=Chip.FREE_HIT
    )
    assert request.chips_played == played


class _RecordingSquad:
    """A Squad stand-in that records the gameweek each move was priced at."""

    def __init__(self):
        self.removed_in = []
        self.added_in = []

    def remove_player(self, player_id, price=None, gameweek=None, **kwargs):  # noqa: ARG002
        self.removed_in.append(gameweek)
        return True

    def add_player(self, player, price=None, gameweek=None, **kwargs):  # noqa: ARG002
        self.added_in.append(gameweek)
        return True


def test_the_resulting_squad_is_priced_at_the_gameweek_of_the_move(monkeypatch):
    """
    Not at whatever gameweek the real season happens to be up to.

    `Squad.add_player` and `Squad.remove_player` default their gameweek to
    `next_gameweek()`, which is a gameweek of the *current* season. Left to
    default, a replay of a past season would read every player's club and price
    from that gameweek number of the season being replayed instead of from the
    one the transfer is made in.
    """
    recorder = _RecordingSquad()
    monkeypatch.setattr(rt, "get_starting_squad", lambda **kwargs: recorder)
    plan = Plan(root_gameweek=7, outcomes=(_outcome(7, GameweekMove(1), (0,), (30,)),))

    rt.squad_for_next_gameweek(plan, season="2223")

    assert recorder.removed_in == [7]
    assert recorder.added_in == [7]


def test_the_resulting_squad_sells_at_the_live_price_when_using_the_api(monkeypatch):
    """A live squad's sale prices come from the API, as the search's did."""
    sold_with_api = []

    class _Squad(_RecordingSquad):
        def remove_player(self, player_id, price=None, gameweek=None, **kwargs):
            sold_with_api.append(kwargs.get("use_api"))
            return super().remove_player(player_id, price, gameweek)

    monkeypatch.setattr(rt, "get_starting_squad", lambda **kwargs: _Squad())
    plan = Plan(root_gameweek=7, outcomes=(_outcome(7, GameweekMove(1), (0,), (30,)),))

    rt.squad_for_next_gameweek(plan, use_api=True)

    assert sold_with_api == [True]


def test_a_player_the_squad_cannot_take_fails_loudly(monkeypatch):
    """
    Not an incomplete squad that fails later, somewhere unrelated.

    A wrong sale price leaves a wildcard's rebuild short of money partway
    through, and the squad used to be returned a player short.
    """

    class _FullSquad(_RecordingSquad):
        budget = 0

        def add_player(self, player, price=None, gameweek=None, **kwargs):  # noqa: ARG002
            return False

    monkeypatch.setattr(rt, "get_starting_squad", lambda **kwargs: _FullSquad())
    plan = Plan(root_gameweek=7, outcomes=(_outcome(7, GameweekMove(1), (0,), (30,)),))

    with pytest.raises(RuntimeError, match="Could not add player 30"):
        rt.squad_for_next_gameweek(plan, season="2223")
