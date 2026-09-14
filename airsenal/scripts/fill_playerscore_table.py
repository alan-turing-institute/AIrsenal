"""
Fill the "player_score" table with historic results (player_details_xxyy.json).
"""

import contextlib
import datetime
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
from curl_cffi import requests
from sqlalchemy import inspect as sqla_inspect
from sqlalchemy.orm.session import Session

from airsenal.framework.data_fetcher import FPLDataFetcher
from airsenal.framework.output import get_logger, track
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
    get_fixtures_for_gameweek,
    get_past_seasons,
    get_player,
    get_player_from_api_id,
    get_player_scores,
    get_player_team_from_fixture,
    get_team_name,
    is_future_gameweek,
    parse_date,
)

logger = get_logger(__name__)


def download_with_resume(
    url: str,
    dest: Path,
    attempts: int = 5,
    timeout: float = 30.0,
    chunk_size: int = 1024 * 1024,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    for attempt in range(1, attempts + 1):
        existing = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}
        resp = session.get(
            url,
            headers=headers,
            stream=True,
            timeout=timeout,
        )
        try:
            resp.raise_for_status()

            # If server ignored Range (status 200), restart file from scratch.
            if existing > 0 and resp.status_code == 200:
                mode = "wb"
            else:
                mode = "ab" if existing > 0 else "wb"

            with open(dest, mode) as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

            return dest

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException,
        ):
            if attempt == attempts:
                raise
        finally:
            resp.close()

    return dest


def load_attributes_history(season: str) -> pd.DataFrame | None:
    if not is_future_gameweek(season, 1, "2526", 0):
        logger.info(
            "Player attributes history not available before 2526 season, skipping"
        )
        return None

    logger.info("Downloading player attributes history for season %s", season)
    url = (
        "https://raw.githubusercontent.com/alan-turing-institute/AIrsenal/refs/"
        f"heads/main/airsenal/data/player_attributes_history_{season}.csv"
    )

    try:
        with tempfile.TemporaryDirectory(prefix="airsenal_attrs_") as tmpdir:
            tmp_csv = Path(tmpdir) / f"player_attributes_history_{season}.csv"
            download_with_resume(url=url, dest=tmp_csv)
            df_attributes = pd.read_csv(tmp_csv)
        df_attributes["day"] = pd.to_datetime(df_attributes["timestamp"]).dt.date
        df_attributes["season"] = df_attributes["season"].astype(str)
        return df_attributes

    except requests.exceptions.RequestException:
        logger.warning(
            "Could not load player attributes history for season %s",
            season,
            exc_info=True,
        )
    return None


def _filter_attributes_for_player(
    df_attributes: pd.DataFrame, player: Player
) -> pd.DataFrame:

    if (opta_code := player.opta_code) is not None:
        mask = df_attributes["opta_code"] == opta_code
    else:
        logger.warning("Player %s has no opta_code", player)
        mask = df_attributes["player"] == player.name
    return df_attributes.loc[mask]


def _get_availability_on_date(
    date: datetime.date, player: Player, player_attributes: pd.DataFrame
) -> tuple[str | None, int | None]:
    if date < datetime.date(2025, 9, 12):
        # no player attributes history available before this date
        return None, None
    mask = player_attributes["day"] == date
    if mask.sum() != 1:
        logger.warning(
            "Found %s attributes for %s on %s, expected 1 so skipping",
            mask.sum(),
            player,
            date,
        )
        return None, None

    idx = mask.argmax()
    news = player_attributes.iloc[idx]["news"]
    chance_of_playing = player_attributes.iloc[idx]["chance_of_playing_next_round"]
    return news, chance_of_playing


def get_status_from_attributes_history(
    player: Player,
    fixture: Fixture,
    player_attributes: pd.DataFrame,
    dbsession: Session = session,
) -> tuple[str | None, int | None]:
    """
    Get the player's news and chance_of_playing from their attributes history
    as of the morning of the fixture kickoff time.
    """
    matchday = parse_date(fixture.date)
    news, chance_of_playing = _get_availability_on_date(
        matchday, player, player_attributes
    )

    # Deal with known future unavailability, e.g. international duty, in which case a
    # a player might be flagged as unavailable on match day, but that unavailability
    # doesn't apply until the next gameweek. In this case, look back to their status on
    # the gameweek deadline date.
    if (
        news is not None
        and chance_of_playing is not None
        and chance_of_playing < 100
        and fixture.gameweek is not None
    ):
        for known_unavailability in ["international duty", "parent club"]:
            if known_unavailability in news.lower():
                gw_fixtures = get_fixtures_for_gameweek(
                    fixture.gameweek, fixture.season, dbsession
                )
                gw_deadline = min(parse_date(f.date) for f in gw_fixtures)
                return _get_availability_on_date(gw_deadline, player, player_attributes)
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

    for player_name_or_id in track(detail_data, description=f"PLAYER SCORES {season}"):
        # find the player id in the player table.  If they're not
        # there, then we don't care (probably not a current player).
        player = get_player(player_name_or_id, dbsession=dbsession)
        if not player:
            logger.warning("Couldn't find player %s", player_name_or_id)
            continue

        player_attributes = (
            _filter_attributes_for_player(df_attributes, player)
            if df_attributes is not None
            else None
        )

        # now loop through all the fixtures that player played in
        for fixture_data in detail_data[player_name_or_id]:
            # try to find the result in the result table
            gameweek = int(fixture_data["gameweek"])
            if "played_for" in fixture_data:
                played_for = fixture_data["played_for"]
            else:
                played_for = player.team(season, gameweek)
            if not played_for:
                continue

            if "was_home" in fixture_data:
                if fixture_data["was_home"] == "True":
                    was_home = True
                elif fixture_data["was_home"] == "False":
                    was_home = False
                else:
                    was_home = None
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
                logger.warning("Couldn't find result for %s in gw %s", player, gameweek)
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
            if player_attributes is not None and len(player_attributes) > 0:
                news, chance_of_playing = get_status_from_attributes_history(
                    player, fixture, player_attributes, dbsession
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
    for player_api_id in track(input_data, description=f"PLAYER SCORES {season}"):
        player = get_player_from_api_id(player_api_id, dbsession=dbsession)
        if not player:
            # If no player found with this API ID something has gone wrong with the
            # Player table, e.g. clashes between players with the same name
            logger.error("No player with API id %s. Skipped.", player_api_id)
            continue

        player_attributes = (
            _filter_attributes_for_player(df_attributes, player)
            if df_attributes is not None
            else None
        )

        player_data = fetcher.get_gameweek_data_for_player(player_api_id)
        # now loop through all the matches that player played in
        for gameweek, results in player_data.items():
            if gameweek not in range(gw_start, gw_end):
                continue
            for result in results:
                # try to find the match in the match table
                opponent = get_team_name(result["opponent_team"])
                if opponent is None:
                    logger.warning("Couldn't find team %s", result["opponent_team"])
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
                    logger.warning(
                        "Couldn't find fixture for %s vs %s in gameweek %s",
                        player,
                        opponent,
                        gameweek,
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
                if player_attributes is not None and len(player_attributes) > 0:
                    news, chance_of_playing = get_status_from_attributes_history(
                        player, fixture, player_attributes, dbsession
                    )
                    ps.news = news
                    ps.chance_of_playing = chance_of_playing

                if add:
                    dbsession.add(ps)
                logger.debug(ps)
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
