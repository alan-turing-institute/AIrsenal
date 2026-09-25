"""
The genetic algorithm making transfers from a squad, rather than building one.

Real players from the past-season test database, so prices, clubs and the squad
rules are the real ones. The score is each squad's total price, so the search
has something to climb that needs no predictions.
"""

from unittest.mock import patch

import pytest

from airsenal.db.queries.players import list_players
from airsenal.game.enums import Position
from airsenal.optimization.moves import GameweekMove
from airsenal.optimization.protocols import TransferRequest
from airsenal.optimization.squad_optimizers import GeneticAlgorithmConfig
from airsenal.optimization.squad_optimizers.genetic_algorithm import SquadOpt
from airsenal.optimization.squad_score import SquadScoringConfig
from airsenal.optimization.strategies import DEFAULT_STRATEGIES
from airsenal.optimization.strategies.genetic import GeneticTransferStrategy
from airsenal.squad.squad import TOTAL_PER_POSITION, Squad, SubWeights
from tests.conftest import past_data_session_scope

MODULE = "airsenal.optimization.squad_optimizers.genetic_algorithm"
SEASON = "1819"
GAMEWEEK = 1
CONFIG = GeneticAlgorithmConfig(population_size=30, generations=15, random_state=7)


def _squad_value(squad, *args, **kwargs):
    return float(sum(p.purchase_price for p in squad.players))


@pytest.fixture
def dbsession():
    with (
        past_data_session_scope() as ts,
        patch(f"{MODULE}.get_discounted_squad_score", _squad_value),
    ):
        yield ts


@pytest.fixture
def cheap_squad(dbsession):
    """The cheapest legal squad the season starts with, and money to spare."""
    squad = Squad(budget=1000, season=SEASON)
    for position in Position.back_to_front():
        candidates = list_players(
            position=position, season=SEASON, gameweek=GAMEWEEK, dbsession=dbsession
        )
        for player in reversed(candidates):  # cheapest first
            if squad.num_position[position] == TOTAL_PER_POSITION[position]:
                break
            squad.add_player(player.player_id, gameweek=GAMEWEEK, dbsession=dbsession)
    assert squad.is_complete()
    return squad


def _transfer_opt(squad, max_transfers, dbsession):
    return SquadOpt(
        [GAMEWEEK],
        "tag",
        season=SEASON,
        remove_zero=False,
        scoring=SquadScoringConfig(sub_weights=SubWeights()),
        base_squad=squad,
        max_transfers=max_transfers,
        dbsession=dbsession,
    )


def _ids(squad):
    return {p.player_id for p in squad.players}


@pytest.mark.parametrize("max_transfers", [1, 3, 5])
def test_the_result_is_within_the_transfer_limit(cheap_squad, dbsession, max_transfers):
    opt = _transfer_opt(cheap_squad, max_transfers, dbsession)
    best, _ = opt.optimize(CONFIG)
    squad = opt.build_squad(best)

    assert squad is not None
    assert squad.is_complete()
    assert squad.budget >= 0
    assert 0 < len(_ids(squad) - _ids(cheap_squad)) <= max_transfers


def test_no_transfers_allowed_keeps_the_squad(cheap_squad, dbsession):
    opt = _transfer_opt(cheap_squad, 0, dbsession)
    best, _ = opt.optimize(CONFIG)

    assert _ids(opt.build_squad(best)) == _ids(cheap_squad)


def test_a_kept_player_keeps_what_was_paid_for_them(cheap_squad, dbsession):
    """
    Not a price of zero, which would make every later sale look like a profit.

    The money for new players comes from the bank and what the sold players
    fetch; a kept player's purchase price is what their own sale is worked out
    from later in the plan.
    """
    paid = {p.player_id: p.purchase_price for p in cheap_squad.players}
    opt = _transfer_opt(cheap_squad, 3, dbsession)
    best, _ = opt.optimize(CONFIG)
    squad = opt.build_squad(best)

    kept = [p for p in squad.players if p.player_id in paid]
    assert kept
    assert all(p.purchase_price == paid[p.player_id] for p in kept)


def test_sales_fund_the_purchases(cheap_squad, dbsession):
    opt = _transfer_opt(cheap_squad, 3, dbsession)
    best, _ = opt.optimize(CONFIG)
    squad = opt.build_squad(best)

    sold = _ids(cheap_squad) - _ids(squad)
    bought = [p for p in squad.players if p.player_id not in _ids(cheap_squad)]
    proceeds = sum(
        cheap_squad.get_sell_price_for_player(
            player_id, gameweek=GAMEWEEK, dbsession=dbsession
        )
        for player_id in sold
    )
    spent = sum(p.purchase_price for p in bought)
    assert squad.budget == cheap_squad.budget + proceeds - spent


def test_the_base_squad_itself_is_untouched(cheap_squad, dbsession):
    before = (_ids(cheap_squad), cheap_squad.budget)
    opt = _transfer_opt(cheap_squad, 3, dbsession)
    opt.build_squad(opt.optimize(CONFIG)[0])

    assert (_ids(cheap_squad), cheap_squad.budget) == before


def test_a_transfer_limit_needs_a_squad_to_make_transfers_from():
    with pytest.raises(ValueError, match="together or not at all"):
        SquadOpt(
            [GAMEWEEK],
            "tag",
            scoring=SquadScoringConfig(sub_weights=SubWeights()),
            max_transfers=3,
        )


def test_the_strategy_pairs_each_sale_with_a_replacement_in_its_position(
    cheap_squad, dbsession
):
    """`transfer_rows` reads players_out and players_in as pairs, in order."""
    request = TransferRequest(
        move=GameweekMove(4),
        squad=cheap_squad,
        tag="tag",
        gameweeks=[GAMEWEEK],
        root_gameweek=GAMEWEEK,
        season=SEASON,
        num_iterations=15,
    )
    with patch.object(SquadOpt, "__init__", _with_dbsession(dbsession)):
        proposal = GeneticTransferStrategy(CONFIG).propose(request)

    assert len(proposal.players_out) == len(proposal.players_in) > 0
    for out_id, in_id in zip(proposal.players_out, proposal.players_in, strict=True):
        position_out = cheap_squad.get_player_from_id(out_id).position
        position_in = proposal.squad.get_player_from_id(in_id).position
        assert position_out == position_in


def _with_dbsession(dbsession):
    """SquadOpt's constructor, pointed at the test database."""
    original = SquadOpt.__init__

    def init(self, *args, **kwargs):
        # the test database has no predictions for the zero-point filter to use
        original(self, *args, remove_zero=False, dbsession=dbsession, **kwargs)

    return init


def test_the_progress_steps_counted_match_the_number_promised(cheap_squad, dbsession):
    """One step per generation, which is what `num_increments` sizes the bar to."""
    steps = []
    request = TransferRequest(
        move=GameweekMove(3),
        squad=cheap_squad,
        tag="tag",
        gameweeks=[GAMEWEEK],
        root_gameweek=GAMEWEEK,
        season=SEASON,
        num_iterations=6,
        progress=lambda: steps.append(1),
    )
    strategy = GeneticTransferStrategy(CONFIG)
    with patch.object(SquadOpt, "__init__", _with_dbsession(dbsession)):
        strategy.propose(request)

    assert len(steps) == strategy.num_increments(request) == 6


def test_three_or_more_transfers_go_to_the_genetic_algorithm():
    assert DEFAULT_STRATEGIES.name_for(GameweekMove(3)) == "genetic"
    assert DEFAULT_STRATEGIES.name_for(GameweekMove(5)) == "genetic"
