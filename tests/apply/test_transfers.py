"""
The arithmetic and payload building behind applying transfers.

`apply/` is the only code that changes the real FPL entry, and it had no tests at
all. These cover the parts that are pure - the money, the de-duplication, the
ordering and the payload - because those are what fail quietly and cost points.
Nothing here reaches the network: the payload is built and asserted on, never
posted.
"""

import pytest

from airsenal.apply import transfers as transfers_module
from airsenal.apply.transfers import (
    build_transfer_payload,
    deduct_transfer_price,
    price_transfers,
    remove_duplicates,
    separate_transfers_in_or_out,
)


class FakeFetcher:
    """Enough of FPLDataFetcher to build a payload. Posts nothing."""

    def __init__(self, team_id=123):
        self.FPL_TEAM_ID = team_id


def priced(element_out, selling_price, element_in, purchase_price):
    return {
        "element_out": element_out,
        "selling_price": selling_price,
        "element_in": element_in,
        "purchase_price": purchase_price,
    }


# ------------------------------------------------------------------- money ---


def test_selling_above_the_purchase_price_adds_to_the_bank():
    transfers = [priced(1, 75, 2, 70)]
    assert deduct_transfer_price(10, transfers) == 15


def test_selling_below_the_purchase_price_takes_from_the_bank():
    transfers = [priced(1, 60, 2, 70)]
    assert deduct_transfer_price(10, transfers) == 0


def test_the_bank_is_the_net_of_every_transfer():
    transfers = [priced(1, 75, 2, 70), priced(3, 50, 4, 65), priced(5, 100, 6, 90)]
    # +5, -15, +10
    assert deduct_transfer_price(20, transfers) == 20


def test_no_transfers_leaves_the_bank_alone():
    assert deduct_transfer_price(37, []) == 37


# -------------------------------------------------------------- duplicates ---


def test_a_player_on_both_sides_is_dropped_from_both():
    """Otherwise the request asks the API to buy a player it is also selling."""
    transfers_in = [{"element_in": 1}, {"element_in": 2}]
    transfers_out = [{"element_out": 2}, {"element_out": 3}]
    result_in, result_out = remove_duplicates(transfers_in, transfers_out)
    assert result_in == [{"element_in": 1}]
    assert result_out == [{"element_out": 3}]


def test_nothing_is_dropped_when_the_sides_are_disjoint():
    transfers_in = [{"element_in": 1}]
    transfers_out = [{"element_out": 2}]
    assert remove_duplicates(transfers_in, transfers_out) == (
        transfers_in,
        transfers_out,
    )


def test_every_player_on_both_sides_leaves_nothing():
    same = [{"element_in": 1}, {"element_in": 2}]
    out = [{"element_out": 1}, {"element_out": 2}]
    assert remove_duplicates(same, out) == ([], [])


# ------------------------------------------------------------------ halves ---


def test_the_two_halves_keep_their_own_prices():
    transfers = [priced(1, 75, 2, 70), priced(3, 50, 4, 65)]
    outs, ins = separate_transfers_in_or_out(transfers)
    assert outs == [
        {"element_out": 1, "selling_price": 75},
        {"element_out": 3, "selling_price": 50},
    ]
    assert ins == [
        {"element_in": 2, "purchase_price": 70},
        {"element_in": 4, "purchase_price": 65},
    ]


def test_the_halves_stay_in_step():
    """The API pairs them positionally, so the two lists must stay aligned."""
    transfers = [priced(i, 50 + i, 100 + i, 40 + i) for i in range(5)]
    outs, ins = separate_transfers_in_or_out(transfers)
    assert len(outs) == len(ins) == len(transfers)
    for out, in_, original in zip(outs, ins, transfers, strict=True):
        assert out["element_out"] == original["element_out"]
        assert in_["element_in"] == original["element_in"]


# ----------------------------------------------------------------- payload ---


def test_the_payload_is_not_confirmed():
    """`confirmed: False` is what makes the API treat this as a proposal."""
    payload = build_transfer_payload([], 7, FakeFetcher(), None)
    assert payload["confirmed"] is False


def test_the_payload_names_the_entry_and_the_gameweek():
    payload = build_transfer_payload([], 7, FakeFetcher(team_id=456), None)
    assert payload["entry"] == 456
    assert payload["event"] == 7


def test_no_chip_leaves_both_chip_flags_off():
    payload = build_transfer_payload([], 7, FakeFetcher(), None)
    assert payload["wildcard"] is False
    assert payload["freehit"] is False


@pytest.mark.parametrize(
    ("chip", "field"), [("wildcard", "wildcard"), ("free_hit", "freehit")]
)
def test_a_chip_sets_its_own_flag(chip, field):
    """The API spells free_hit without the underscore; the payload has to match."""
    payload = build_transfer_payload([], 7, FakeFetcher(), chip)
    assert payload[field] is True


@pytest.mark.parametrize("chip", ["bench_boost", "triple_captain"])
def test_a_lineup_chip_adds_nothing_to_the_transfer_payload(chip):
    """
    Only the two squad chips belong in a transfer.

    The payload used to be built by stripping the underscore out of whatever chip
    the suggestion carried, which posted a `benchboost` key the transfers endpoint
    does not define.
    """
    payload = build_transfer_payload([], 7, FakeFetcher(), chip)
    assert payload["wildcard"] is False
    assert payload["freehit"] is False
    assert set(payload) == {
        "confirmed",
        "entry",
        "event",
        "transfers",
        "wildcard",
        "freehit",
    }


def test_the_transfers_are_carried_through_untouched():
    transfers = [priced(1, 75, 2, 70)]
    payload = build_transfer_payload(transfers, 7, FakeFetcher(), None)
    assert payload["transfers"] == transfers


# ----------------------------------------------------------------- pricing ---


class FakePlayer:
    def __init__(self, player_id, fpl_api_id):
        self.player_id = player_id
        self.fpl_api_id = fpl_api_id


class PricingFetcher(FakeFetcher):
    """A fetcher whose summary data prices the players being bought."""

    def __init__(self, now_costs):
        super().__init__()
        self._now_costs = now_costs

    def get_player_summary_data(self):
        return {api_id: {"now_cost": cost} for api_id, cost in self._now_costs.items()}


@pytest.fixture
def priced_world(monkeypatch):
    """player_id N has FPL api id 100+N, and sells for 50+N."""
    monkeypatch.setattr(
        transfers_module, "get_player", lambda pid: FakePlayer(pid, 100 + pid)
    )
    monkeypatch.setattr(transfers_module, "get_sell_price", lambda _team, pid: 50 + pid)
    return PricingFetcher({102: 70, 103: 70, 104: 80})


def test_price_transfers_pairs_each_player_out_with_a_player_in(priced_world):
    result = price_transfers([[1, 3], [2, 4]], priced_world)
    assert result == [
        {
            "element_out": 101,
            "selling_price": 51,
            "element_in": 102,
            "purchase_price": 70,
        },
        {
            "element_out": 103,
            "selling_price": 53,
            "element_in": 104,
            "purchase_price": 80,
        },
    ]


def test_the_sale_price_is_not_the_current_price(priced_world):
    """
    FPL sells at the purchase price plus half the rise, not at the market price.

    So a transfer must never be priced out with `now_cost`.
    """
    (transfer,) = price_transfers([[1], [3]], priced_world)
    assert transfer["selling_price"] == 51
    assert transfer["purchase_price"] == 70


def test_price_transfers_refuses_without_a_team_id(priced_world):
    priced_world.FPL_TEAM_ID = None
    with pytest.raises(RuntimeError, match="FPL team ID not set"):
        price_transfers([[1], [3]], priced_world)
