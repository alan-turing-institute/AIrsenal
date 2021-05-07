from airsenal.framework.schema import Player, PlayerAttributes


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


def test_is_injured_or_suspended():
    """
    Check Player.is_injured_or_suspended() returns appropriate value both if details are
    available in attributes table for requested gameweek, and if they're not available.
    """
    player_id = 1
    season = "1920"
    price = 50
    position = "MID"
    team = "ABC"
    # gw: (chance_of_playing_next_round, return_gameweek)
    team_dict = {
        2: (100, None),
        3: (75, None),
        4: (50, 5),
        5: (0, None),
    }

    player = Player()
    player.player_id = player_id
    player.name = "Test Player"
    player.attributes = []

    for gw, attr in team_dict.items():
        pa = PlayerAttributes()
        pa.season = season
        pa.team = team
        pa.gameweek = gw
        pa.price = price
        pa.position = position
        pa.player_id = player_id
        pa.chance_of_playing_next_round = attr[0]
        pa.return_gameweek = attr[1]
        player.attributes.append(pa)

    # gameweek available in attributes table
    # not injured, 100% available
    assert player.is_injured_or_suspended(season, 2, 2) is False
    assert player.is_injured_or_suspended(season, 2, 4) is False
    # not injured, 75% available
    assert player.is_injured_or_suspended(season, 3, 3) is False
    assert player.is_injured_or_suspended(season, 3, 5) is False
    # 50% available, expected back gw 5
    assert player.is_injured_or_suspended(season, 4, 4) is True
    assert player.is_injured_or_suspended(season, 4, 5) is False
    # 100% unavailable, mo return gameweek
    assert player.is_injured_or_suspended(season, 5, 6) is True
    assert player.is_injured_or_suspended(season, 5, 7) is True
    # gameweek before earliest available: return status as of first available
    assert player.is_injured_or_suspended(season, 1, 1) is False
    # gameweek after last available: return status as of last available
    assert player.is_injured_or_suspended(season, 6, 1) is True
