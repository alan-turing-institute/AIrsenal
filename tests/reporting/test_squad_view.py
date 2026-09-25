"""The formation table a squad is printed as."""

from rich.console import Console

from airsenal.reporting.squad_view import formation_table
from airsenal.squad.squad import Squad


def test_formation_table():
    t = Squad()

    class MockPlayer:
        def __init__(
            self,
            name: str,
            position: str,
            is_starting: bool,
            sub_position: int | None = None,
        ):
            self.name = name
            self.team = "TEST"
            self.position = position
            self.is_starting = is_starting
            self.is_captain = name == "Captain"
            self.is_vice_captain = name == "Vice"
            self.sub_position = sub_position
            self.predicted_points = {"tag": {1: 5.0}}

        def __str__(self) -> str:
            return self.name

    t.players = [
        MockPlayer("Keeper", "GK", True),
        MockPlayer("Defender One", "DEF", True),
        MockPlayer("Defender Two", "DEF", True),
        MockPlayer("Defender Three", "DEF", True),
        MockPlayer("Midfielder One", "MID", True),
        MockPlayer("Midfielder Two", "MID", True),
        MockPlayer("Midfielder Three", "MID", True),
        MockPlayer("Midfielder Four", "MID", True),
        MockPlayer("Captain", "FWD", True),
        MockPlayer("Vice", "FWD", True),
        MockPlayer("Forward Three", "FWD", True),
        MockPlayer("Sub Keeper", "GK", False, 0),
        MockPlayer("Sub One", "DEF", False, 0),
        MockPlayer("Sub Two", "MID", False, 1),
        MockPlayer("Sub Three", "FWD", False, 2),
    ]
    scoring_calls = []

    def get_expected_points(tag, gameweek, bench_boost=False, triple_captain=False):
        scoring_calls.append((tag, gameweek, bench_boost, triple_captain))
        return 60.0 + 20.0 * bench_boost + 5.0 * triple_captain

    t.get_expected_points = get_expected_points
    console = Console(record=True, width=100)

    console.print(formation_table(t, "tag", 1))
    console.print(formation_table(t, "tag", 1, bench_boost=True))
    console.print(formation_table(t, "tag", 1, triple_captain=True))

    output = console.export_text()
    assert "Captain" in output
    assert "(C)" in output
    assert "Substitutes" in output
    assert "5.0 pts" in output
    assert "GAMEWEEK 1" in output
    assert "60.0pts" in output
    assert "80.0pts" in output
    assert "with bench boost" in output
    assert "65.0pts" in output
    assert "with triple captain" in output
    assert "(TC)" in output
    assert scoring_calls == [
        ("tag", 1, False, False),
        ("tag", 1, True, False),
        ("tag", 1, False, True),
    ]
