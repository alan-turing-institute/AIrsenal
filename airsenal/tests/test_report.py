"""
Tests for the report's formatting and its uncertainty caveats.
"""

from airsenal.scripts.make_report import (
    captain_section,
    format_price,
    squad_section,
)


class FakePlayer:
    def __init__(
        self,
        name,
        position,
        points,
        team="ABC",
        price=50,
        is_starting=True,
        sub_position=None,
        is_captain=False,
        is_vice_captain=False,
    ):
        self.name = name
        self.position = position
        self.team = team
        self.purchase_price = price
        self.is_starting = is_starting
        self.sub_position = sub_position
        self.is_captain = is_captain
        self.is_vice_captain = is_vice_captain
        self.predicted_points = {"tag": {1: points}}

    def __repr__(self):
        return self.name


class FakeSquad:
    def __init__(self, players):
        self.players = players

    def get_expected_points(self, *args, **kwargs):
        return 0.0

    def get_formation(self):
        return {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2}


def make_squad(captain_points=8.0, second_points=4.0):
    players = [
        FakePlayer("Keeper", "GK", 3.0),
        FakePlayer("Star", "MID", captain_points, is_captain=True),
        FakePlayer("Runner Up", "FWD", second_points, is_vice_captain=True),
        FakePlayer("Reserve", "DEF", 1.0, is_starting=False, sub_position=0),
        FakePlayer("Sub Keeper", "GK", 0.5, is_starting=False, sub_position=1),
    ]
    return FakeSquad(players)


def test_format_price():
    assert format_price(105) == "£10.5m"
    assert format_price(None) == "?"


def test_squad_section_separates_xi_and_bench():
    lines = squad_section(make_squad(), gameweek=1, tag="tag")
    text = "\n".join(lines)

    assert "Starting XI (4-4-2)" in text
    assert "Star (C)" in text
    assert "Runner Up (VC)" in text
    # bench players appear under the bench heading, in sub order
    bench_text = text.split("Bench, in the order")[1]
    assert bench_text.index("Reserve") < bench_text.index("Sub Keeper")


def test_captain_section_reports_the_margin():
    lines = captain_section(make_squad(captain_points=8.0), gameweek=1, tag="tag")
    text = "\n".join(lines)

    assert "Star" in text
    assert "4.00" in text, "margin over the next best option"
    assert "equivalent" not in text, "a 4 point margin is decisive"


def test_captain_section_flags_a_narrow_margin():
    """Two captains within a fraction of a point are not distinguishable by the
    model, and the report should say so rather than implying a real preference."""
    lines = captain_section(
        make_squad(captain_points=5.2, second_points=5.0), gameweek=1, tag="tag"
    )
    text = "\n".join(lines)

    assert "equivalent" in text


def test_captain_section_picks_highest_scorer_not_the_flag():
    """The captain shown is whoever the predictions rank first."""
    squad = make_squad(captain_points=3.0, second_points=9.0)
    lines = captain_section(squad, gameweek=1, tag="tag")
    assert "**Runner Up**" in "\n".join(lines)
