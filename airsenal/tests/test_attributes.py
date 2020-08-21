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


def test_get_team():
    """
    Check Player.team() returns appropriate value both if details are available
    in attributes table for requested gameweek, and if they're not available.
    """
    player_id = 1
    season = "1920"
    price = 50
    position = "MID"
    team_dict = {2: "ABC", 5: "XYZ"}  # gw: team

    player = Player()
    player.player_id = player_id
    player.name = "Test Player"
    player.attributes = []

    for gw, team in team_dict.items():
        pa = PlayerAttributes()
        pa.season = season
        pa.team = team
        pa.gameweek = gw
        pa.price = price
        pa.position = position
        pa.player_id = player_id
        player.attributes.append(pa)

    # gameweek available in attributes table
    assert player.team(season, 2) == team_dict[2]
    # gameweek before earliest available: return first available
    assert player.team(season, 1) == team_dict[2]
    # gameweek after last available: return last available
    assert player.team(season, 6) == team_dict[5]
    # gameweek between two available values: return nearest
    assert player.team(season, 3) == team_dict[2]
    assert player.team(season, 4) == team_dict[5]
    # no gameweek available for seaaon: return None
    assert player.team("1011", 1) is None


def test_get_position():
    """
    Check Player.position() returns appropriate value both if details are available
    in attributes table for requested gameweek, and if they're not available.
    """
    player_id = 1
    gameweek = 1
    price = 50
    pos_dict = {"1819": "MID", "1920": "FWD"}  # season: position
    team = "TST"

    player = Player()
    player.player_id = player_id
    player.name = "Test Player"
    player.attributes = []

    for season, position in pos_dict.items():
        pa = PlayerAttributes()
        pa.season = season
        pa.team = team
        pa.gameweek = gameweek
        pa.price = price
        pa.position = position
        pa.player_id = player_id
        player.attributes.append(pa)

    # season available in attributes table
    assert player.position("1819") == pos_dict["1819"]
    assert player.position("1920") == pos_dict["1920"]
    # season not available
    assert player.position("1011") is None
