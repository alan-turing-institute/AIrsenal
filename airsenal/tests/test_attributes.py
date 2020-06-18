from ..framework.schema import Player, PlayerAttributes


def test_get_price():
    """
    Check Player.price() returns appropriate value both if details are available
    in attributes table for requested gameweek, and if they're not available.
    """
    player_id = 1
    season = "1920"
    team = "TST"
    position = "MID"
    price_dict = {2: 50, 4: 150}  # gw: price

    player = Player()
    player.player_id = player_id
    player.name = "Test Player"
    player.attributes = []

    for gw, price in price_dict.items():
        pa = PlayerAttributes()
        pa.season = season
        pa.team = team
        pa.gameweek = gw
        pa.price = price
        pa.position = position
        pa.player_id = player_id
        player.attributes.append(pa)

    # gameweek available in attributes table
    assert player.price(season, 2) == price_dict[2]
    # gameweek before earliest available: return first available
    assert player.price(season, 1) == price_dict[2]
    # gameweek after last available: return last available
    assert player.price(season, 5) == price_dict[4]
    # gameweek between two available values: interpolate
    assert player.price(season, 3) == (price_dict[2] + price_dict[4]) / 2
    # no gameweek available for seaaon: return None
    assert player.price("1011", 1) is None
