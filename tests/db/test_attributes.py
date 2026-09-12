from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from airsenal.db.models import Base, Player, PlayerAttributes


def test_get_price():
    """`Player.price` for a gameweek we have attributes for, and one we do not."""
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
    assert player.price(2, season) == price_dict[2]
    # gameweek before earliest available: return first available
    assert player.price(1, season) == price_dict[2]
    # gameweek after last available: return last available
    assert player.price(5, season) == price_dict[4]
    # gameweek between two available values: interpolate
    assert player.price(3, season) == (price_dict[2] + price_dict[4]) / 2
    # no gameweek available for seaaon: return None
    assert player.price(1, "1011") is None


def test_get_team():
    """`Player.team` for a gameweek we have attributes for, and one we do not."""
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
    assert player.team(2, season) == team_dict[2]
    # gameweek before earliest available: return first available
    assert player.team(1, season) == team_dict[2]
    # gameweek after last available: return last available
    assert player.team(6, season) == team_dict[5]
    # gameweek between two available values: return nearest
    assert player.team(3, season) == team_dict[2]
    assert player.team(4, season) == team_dict[5]
    # no gameweek available for seaaon: return None
    assert player.team(1, "1011") is None


def test_get_position():
    """`Player.position` for a season we have attributes for, and one we do not."""
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
    """`Player.is_injured_or_suspended` with attributes available, and without."""
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


def test_availability_is_not_queried_once_per_fixture():
    """
    Asking about more gameweeks must not mean more queries.

    This is read from the innermost loop of the points prediction, so over a
    replay a per-fixture query is tens of thousands of them for an answer that
    cannot change within a season. `Player.attributes` is a lazy relationship, so
    the first access legitimately loads it and every later one must not.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    dbsession = sessionmaker(bind=engine)()
    season = "1920"

    player = Player()
    player.player_id = 1
    player.name = "Absent"
    dbsession.add(player)
    for gameweek in range(1, 6):
        pa = PlayerAttributes()
        pa.player = player
        pa.player_id = 1
        pa.season = season
        pa.gameweek = gameweek
        pa.price = 50
        pa.team = "ARS"
        pa.position = "MID"
        pa.chance_of_playing_next_round = 0
        pa.return_gameweek = 6
        dbsession.add(pa)
    dbsession.commit()
    dbsession.expire_all()

    statements = []
    event.listen(
        engine, "before_cursor_execute", lambda *args: statements.append(args[2])
    )
    player = dbsession.get(Player, 1)
    for gameweek in range(1, 6):
        assert player.is_injured_or_suspended(season, gameweek, gameweek)

    assert sum("player_attributes" in s for s in statements) == 1
