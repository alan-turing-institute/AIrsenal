"""
This test is more for development purposes than regular testing, and may
fail between seasons, but keep here for now in case it's useful to plunder
bits from.
"""
import pytest

from airsenal.framework.optimization_utils import get_squad_from_transactions
from airsenal.framework.utils import CURRENT_SEASON

SQUAD_DATA = [
    {
        "gameweek": 1,
        "points": 45,
        "subs": ["Vicente Guaita", "Marcos Alonso", "Bryan Mbeumo", "Sam Greenwood"],
        "captain": "Kevin De Bruyne",
        "vice_captain": "Bruno Borges Fernandes",
    },
    {
        "gameweek": 2,
        "points": 73,
        "subs": ["Vicente Guaita", "Joel Matip", "Anthony Elanga", "Sam Greenwood"],
        "captain": "Kevin De Bruyne",
        "vice_captain": "Phil Foden",
    },
    {
        "gameweek": 8,
        "points": 49,
        "subs": ["Illan Meslier", "Marcus Rashford", "Luis Díaz", "Jack Harrison"],
        "captain": "Erling Haaland",
        "vice_captain": "Gabriel Martinelli Silva",
    },
]


def get_squad_for_gameweek(
    subs, captain, vice_captain, gameweek, season=CURRENT_SEASON, fpl_team_id=2779516
):
    s = get_squad_from_transactions(gameweek)
    for p in s.players:
        try:
            subs_index = subs.index(p.name)
            p.is_starting = False
            p.sub_position = subs_index
        except ValueError:
            p.is_starting = True
        if p.name == captain:
            p.is_captain = True
        if p.name == vice_captain:
            p.is_vice_captain = True
    return s


def compare_gameweek(test_dict, season=CURRENT_SEASON):
    """
    Compare what we get from Squad.get_actual_points from the actual
    points we can see in the FPL website.
    """
    actual_points = test_dict["points"]
    s = get_squad_for_gameweek(
        test_dict["subs"],
        test_dict["captain"],
        test_dict["vice_captain"],
        test_dict["gameweek"],
    )
    calc_points = s.get_actual_points(test_dict["gameweek"], season=season)
    assert calc_points == actual_points


@pytest.mark.skip(reason="test requires real AIrsenal data")
def test_gameweeks():
    for test_dict in SQUAD_DATA:
        compare_gameweek(test_dict)
