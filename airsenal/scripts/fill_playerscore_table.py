"""
Fill the "player_score" table with historic results (player_details_xxyy.json).
"""

import contextlib
import json
import os
import urllib.error
import warnings

import pandas as pd
from sqlalchemy import inspect as sqla_inspect
from sqlalchemy.orm.session import Session

from airsenal.framework.data_fetcher import FPLDataFetcher
from airsenal.framework.schema import (
    Fixture,
    Player,
    PlayerScore,
    session,
    session_scope,
)
from airsenal.framework.season import CURRENT_SEASON, sort_seasons
from airsenal.framework.utils import (
    NEXT_GAMEWEEK,
    find_fixture,
    get_past_seasons,
    get_player,
    get_player_from_api_id,
    get_player_scores,
    get_player_team_from_fixture,
    get_team_name,
    is_future_gameweek,
    parse_date,
)


def load_attributes_history(season: str) -> pd.DataFrame | None:
    """
    Load the PlayerAttributes history for a given season from the relevant CSV file.
    """
    try:
        df_attributes = pd.read_csv(
            "https://raw.githubusercontent.com/alan-turing-institute/AIrsenal/refs/"
            f"heads/main/airsenal/data/player_attributes_history_{season}.csv"
        )
        df_attributes["day"] = pd.to_datetime(df_attributes["timestamp"]).dt.date
        df_attributes["season"] = df_attributes["season"].astype(str)
        return df_attributes
    except urllib.error.HTTPError as e:
        if is_future_gameweek(season, 1, "2526", 1):  # got history from 2526 season
            msg = f"Could not load player attributes history for season {season}"
            warnings.warn(f"{e}\n{msg}", stacklevel=2)
    return None


def get_status_from_attributes_history(
    player: Player, fixture: Fixture, df_attributes: pd.DataFrame
) -> tuple[None | str, None | int]:
    """
    Get the player's news and chance_of_playing from their attributes history
    as of the morning of the fixture kickoff time.
    """
    if fixture.season != df_attributes["season"].iloc[0]:
        msg = "Attributes dataframe season does not match fixture season"
        raise ValueError(msg)

    matchday = parse_date(fixture.date)
    mask = df_attributes["day"] == matchday

    if (opta_code := player.opta_code) is not None:
        mask = mask & (df_attributes["opta_code"] == opta_code)
    else:
        msg = f"Player {player} has no opta_code"
        warnings.warn(msg, stacklevel=2)
        mask = mask & (df_attributes["player"] == player.name)

    if mask.sum() != 1 and is_future_gameweek(
        fixture.season,
        fixture.gameweek,
        "2526",
        4,  # gw started saving history
    ):
        warnings.warn(
            (
                f"Found {mask.sum()} attributes for {player} on {matchday}, expected "
                "1 so skipping"
            ),
            stacklevel=2,
        )
        return None, None
    idx = mask.argmax()
    news = df_attributes.iloc[idx]["news"]
    chance_of_playing = df_attributes.iloc[idx]["chance_of_playing_next_round"]
    return news, chance_of_playing


def fill_playerscores_from_json(
    detail_data: list, season: str, dbsession: Session = session
) -> None:
    # Get column metadata once for efficiency
    mapper = sqla_inspect(PlayerScore)
    extended_feats = [
        col.key
        for col in mapper.columns
        if col.key
        not in [
            "id",
            "player_team",
            "opponent",
            "goals",
            "assists",
            "bonus",
            "points",
            "conceded",
            "minutes",
            "player_id",
            "result_id",
            "fixture_id",
            "news",
            "chance_of_playing",
        ]
    ]
    df_attributes = load_attributes_history(season)

    for player_name in detail_data:
        # find the player id in the player table.  If they're not
        # there, then we don't care (probably not a current player).
        player = get_player(player_name, dbsession=dbsession)
        if not player:
            print(f"Couldn't find player {player_name}")
            continue

        print(f"SCORES {season} {player}")
        # now loop through all the fixtures that player played in
        for fixture_data in detail_data[player_name]:
            # try to find the result in the result table
            gameweek = int(fixture_data["gameweek"])
            if "played_for" in fixture_data:
                played_for = fixture_data["played_for"]
            else:
                played_for = player.team(season, gameweek)
            if not played_for:
                continue

            if fixture_data["was_home"] == "True":
                was_home = True
            elif fixture_data["was_home"] == "False":
                was_home = False
            else:
                was_home = None

            fixture = find_fixture(
                played_for,
                was_home=was_home,
                other_team=fixture_data["opponent"],
                gameweek=gameweek,
                season=season,
                kickoff_time=fixture_data["kickoff_time"],
                dbsession=dbsession,
            )

            if not fixture or not fixture.result:
                print(f"  Couldn't find result for {player} in gw {gameweek}")
                continue
            ps = PlayerScore()
            ps.player_team = played_for
            ps.opponent = fixture_data["opponent"]
            ps.goals = fixture_data["goals"]
            ps.assists = fixture_data["assists"]
            ps.bonus = fixture_data["bonus"]
            ps.points = fixture_data["points"]
            ps.conceded = fixture_data["conceded"]
            ps.minutes = fixture_data["minutes"]
            ps.player = player
            ps.result = fixture.result
            ps.fixture = fixture

            # extended features
            # get features excluding the core ones already populated above
            for feat in extended_feats:
                with contextlib.suppress(KeyError):
                    ps.__setattr__(feat, fixture_data[feat])

            # get injury/suspension status from attributes history
            if df_attributes is not None:
                news, chance_of_playing = get_status_from_attributes_history(
                    player, fixture, df_attributes
                )
                ps.news = news
                ps.chance_of_playing = chance_of_playing

            dbsession.add(ps)
    dbsession.commit()


def fill_playerscores_from_api(
    season: str,
    gw_start: int = 1,
    gw_end: int = NEXT_GAMEWEEK,
    dbsession: Session = session,
) -> None:
    # Get column metadata once for efficiency
    mapper = sqla_inspect(PlayerScore)
    extended_feats = [
        col.key
        for col in mapper.columns
        if col.key
        not in [
            "id",
            "player_team",
            "opponent",
            "goals",
            "assists",
            "bonus",
            "points",
            "conceded",
            "minutes",
            "player_id",
            "result_id",
            "fixture_id",
            "news",
            "chance_of_playing",
        ]
    ]
    df_attributes = load_attributes_history(season)
    fetcher = FPLDataFetcher()
    input_data = fetcher.get_player_summary_data()
    for player_api_id in input_data:
        player = get_player_from_api_id(player_api_id, dbsession=dbsession)
        if not player:
            # If no player found with this API ID something has gone wrong with the
            # Player table, e.g. clashes between players with the same name
            print(f"ERROR! No player with API id {player_api_id}. Skipped.")
            continue

        print(f"SCORES {season} {player}")
        player_data = fetcher.get_gameweek_data_for_player(player_api_id)
        # now loop through all the matches that player played in
        for gameweek, results in player_data.items():
            if gameweek not in range(gw_start, gw_end):
                continue
            for result in results:
                # try to find the match in the match table
                opponent = get_team_name(result["opponent_team"])
                if opponent is None:
                    print(f"Couldn't find team {result['opponent_team']}")
                    continue

                fixture = find_fixture(
                    opponent,
                    was_home=not result["was_home"],
                    gameweek=gameweek,
                    season=season,
                    kickoff_time=result["kickoff_time"],
                    dbsession=dbsession,
                )
                if fixture is None or fixture.result is None:
                    print(
                        f"Couldn't find fixture for {player} vs {opponent} in "
                        f"gameweek {gameweek}"
                    )
                    continue
                played_for = get_player_team_from_fixture(
                    fixture,
                    opponent,
                    player_at_home=result["was_home"],
                    season=season,
                    dbsession=dbsession,
                )

                ps = get_player_scores(
                    fixture=fixture, player=player, dbsession=dbsession
                )
                if ps is None:
                    ps = PlayerScore()
                    add = True
                elif isinstance(ps, list):
                    msg = f"Multiple player scores found for {player} in {fixture}"
                    raise ValueError(msg)
                else:
                    add = False
                ps.player_team = played_for
                ps.opponent = opponent
                ps.goals = result["goals_scored"]
                ps.assists = result["assists"]
                ps.bonus = result["bonus"]
                ps.points = result["total_points"]
                ps.conceded = result["goals_conceded"]
                ps.minutes = result["minutes"]
                ps.player = player
                ps.fixture = fixture
                ps.result = fixture.result

                # extended features
                # get features excluding the core ones already populated above
                for feat in extended_feats:
                    with contextlib.suppress(KeyError):
                        ps.__setattr__(feat, result[feat])

                # get injury/suspension status from attributes history
                if df_attributes is not None:
                    news, chance_of_playing = get_status_from_attributes_history(
                        player, fixture, df_attributes
                    )
                    ps.news = news
                    ps.chance_of_playing = chance_of_playing

                if add:
                    dbsession.add(ps)
                print(
                    f"  got {result['total_points']} points vs {opponent} in gameweek "
                    f"{gameweek}"
                )
    dbsession.commit()


def make_playerscore_table(
    seasons: list[str] | None = None, dbsession: Session = session
) -> None:
    # previous seasons data from json files
    if seasons is None:
        seasons = []
    if not seasons:
        seasons = [CURRENT_SEASON]
        seasons += get_past_seasons(3)
    for season in sort_seasons(seasons):
        if season == CURRENT_SEASON:
            # current season - use API
            fill_playerscores_from_api(CURRENT_SEASON, dbsession=dbsession)
        else:
            input_path = os.path.join(
                os.path.dirname(__file__), f"../data/player_details_{season}.json"
            )
            with open(input_path) as f:
                input_data = json.load(f)
            fill_playerscores_from_json(input_data, season, dbsession=dbsession)


if __name__ == "__main__":
    with session_scope() as session:
        make_playerscore_table(dbsession=session)
