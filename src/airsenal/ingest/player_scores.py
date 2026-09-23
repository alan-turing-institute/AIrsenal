"""Fill the "player_score" table from this season's FPL API and past seasons' files."""

import contextlib
import json
from typing import Any

import pandas as pd
from sqlalchemy import inspect as sqla_inspect
from sqlalchemy.orm.session import Session

from airsenal.core.console import track
from airsenal.core.data_files import data_file
from airsenal.core.dates import parse_date
from airsenal.core.logging import get_logger
from airsenal.db.models import Fixture, Player, PlayerScore, Result
from airsenal.db.queries.fixtures import (
    find_fixture,
    get_gameweek_start_date,
    get_player_team_from_fixture,
)
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.db.queries.players import (
    get_player,
    get_player_attributes,
    get_player_from_api_id,
)
from airsenal.db.queries.scores import get_player_scores
from airsenal.db.queries.teams import get_team_name
from airsenal.db.session import get_session
from airsenal.game.season import CURRENT_SEASON, default_seasons, sort_seasons
from airsenal.ingest.attributes_history import (
    covers_date,
    filter_attributes_for_player,
    get_availability_on_date,
    load_attributes_history,
)
from airsenal.remote.fpl_api import get_fetcher

logger = get_logger(__name__)


# Set explicitly by the fill functions; every other column is copied by name.
_CORE_COLUMNS = frozenset(
    {
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
    }
)


def _extended_features() -> list[str]:
    """The PlayerScore columns copied straight from the source data by name."""
    return [
        col.key
        for col in sqla_inspect(PlayerScore).columns
        if col.key not in _CORE_COLUMNS
    ]


# Each PlayerScore column, and the key that holds it in that source's match data.
_JSON_STATS = {
    "goals": "goals",
    "assists": "assists",
    "bonus": "bonus",
    "points": "points",
    "conceded": "conceded",
    "minutes": "minutes",
}
_API_STATS = {
    "goals": "goals_scored",
    "assists": "assists",
    "bonus": "bonus",
    "points": "total_points",
    "conceded": "goals_conceded",
    "minutes": "minutes",
}

_WAS_HOME = {"True": True, "False": False}


def _attributes_for_player(
    player: Player, df_attributes: pd.DataFrame | None
) -> pd.DataFrame | None:
    """The player's rows of the attributes history, or None if there is none."""
    if df_attributes is None:
        return None
    return filter_attributes_for_player(player, df_attributes)


def _fill_score(
    score: PlayerScore,
    data: dict[str, Any],
    stats: dict[str, str],
    extended_feats: list[str],
    player_attributes: pd.DataFrame | None,
    fixture: Fixture,
    result: Result,
    opponent: str,
    player: Player,
    team: str,
    dbsession: Session,
) -> None:
    """Set a score's columns from one match's source data, and add it to the session."""
    score.player_team = team
    score.opponent = opponent
    for column, key in stats.items():
        setattr(score, column, data[key])
    score.player = player
    score.fixture = fixture
    score.result = result

    for feat in extended_feats:
        with contextlib.suppress(KeyError):
            setattr(score, feat, data[feat])

    # what was known about their availability for this match
    score.news, score.chance_of_playing = get_availability_for_fixture(
        player, fixture, player_attributes, dbsession
    )
    dbsession.add(score)


def get_status_from_attributes_history(
    player: Player,
    fixture: Fixture,
    player_attributes: pd.DataFrame,
    dbsession: Session | None = None,
) -> tuple[str | None, int | None]:
    """A player's news and chance_of_playing as of the morning of kickoff."""
    dbsession = get_session(dbsession)
    matchday = parse_date(fixture.date)
    news, chance_of_playing = get_availability_on_date(
        matchday, player, player_attributes
    )

    # Known future unavailability, e.g. international duty: a player can be flagged
    # unavailable on match day for something that does not apply until the next
    # gameweek. Look back to their status on the gameweek deadline date instead.
    if (
        news is not None
        and chance_of_playing is not None
        and chance_of_playing < 100
        and fixture.gameweek is not None
    ):
        for known_unavailability in ["international duty", "parent club"]:
            if known_unavailability in news.lower():
                gameweek_deadline = get_gameweek_start_date(
                    fixture.gameweek, fixture.season, dbsession
                )
                if gameweek_deadline is None:
                    break
                return get_availability_on_date(
                    gameweek_deadline, player, player_attributes
                )
    return news, chance_of_playing


def get_availability_for_fixture(
    player: Player,
    fixture: Fixture,
    player_attributes: pd.DataFrame | None,
    dbsession: Session | None = None,
) -> tuple[str | None, int | None]:
    """
    A player's news and chance of playing for one fixture.

    What the per-day history says on the morning of the match, where it covers
    that day. Otherwise the gameweek's attributes row, which before the history
    starts is what the absences scrape says about the gameweek.
    """
    dbsession = get_session(dbsession)
    if (
        player_attributes is not None
        and len(player_attributes) > 0
        and covers_date(parse_date(fixture.date), player_attributes)
    ):
        return get_status_from_attributes_history(
            player, fixture, player_attributes, dbsession
        )

    if fixture.gameweek is None:
        return None, None
    attributes = get_player_attributes(
        player.player_id,
        gameweek=fixture.gameweek,
        season=fixture.season,
        dbsession=dbsession,
    )
    if attributes is None:
        return None, None
    return attributes.news, attributes.chance_of_playing_next_round


def fill_playerscores_from_json(
    detail_data: dict[str, list[dict[str, Any]]],
    season: str,
    dbsession: Session | None = None,
) -> None:
    """
    Fill the player_score table from a packaged `player_details_xxyy.json`.

    Keyed by player name, each holding one entry per fixture that player
    appeared in. Rows are added rather than merged, so this is for filling an
    empty table - `fill_playerscores_from_api` is the one that can be re-run.
    """
    dbsession = get_session(dbsession)
    extended_feats = _extended_features()
    df_attributes = load_attributes_history(season)

    for player_name_or_id in track(detail_data, description=f"PLAYER SCORES {season}"):
        # find the player id in the player table.  If they're not
        # there, then we don't care (probably not a current player).
        player = get_player(player_name_or_id, dbsession=dbsession)
        if not player:
            logger.warning("Couldn't find player %s", player_name_or_id)
            continue

        player_attributes = _attributes_for_player(player, df_attributes)

        # now loop through all the fixtures that player played in
        for fixture_data in detail_data[player_name_or_id]:
            # try to find the result in the result table
            gameweek = int(fixture_data["gameweek"])
            if "played_for" in fixture_data:
                played_for = fixture_data["played_for"]
            else:
                played_for = player.team(gameweek, season)
            if not played_for:
                continue

            fixture = find_fixture(
                played_for,
                was_home=_WAS_HOME.get(fixture_data.get("was_home", "")),
                other_team=fixture_data["opponent"],
                gameweek=gameweek,
                season=season,
                kickoff_time=fixture_data["kickoff_time"],
                dbsession=dbsession,
            )

            if not fixture or not fixture.result:
                logger.warning("Couldn't find result for %s in gw %s", player, gameweek)
                continue
            _fill_score(
                PlayerScore(),
                fixture_data,
                _JSON_STATS,
                extended_feats,
                player_attributes,
                fixture,
                fixture.result,
                fixture_data["opponent"],
                player,
                played_for,
                dbsession,
            )
    dbsession.commit()


def fill_playerscores_from_api(
    season: str,
    gameweek_start: int = 1,
    gameweek_end: int | None = None,
    dbsession: Session | None = None,
) -> None:
    fetcher = get_fetcher()
    gameweek_end = (
        next_gameweek(fetcher=fetcher) if gameweek_end is None else gameweek_end
    )
    dbsession = get_session(dbsession)
    extended_feats = _extended_features()
    df_attributes = load_attributes_history(season)
    input_data = fetcher.get_player_summary_data()
    for player_api_id in track(input_data, description=f"PLAYER SCORES {season}"):
        player = get_player_from_api_id(player_api_id, dbsession=dbsession)
        if not player:
            # If no player found with this API ID something has gone wrong with the
            # Player table, e.g. clashes between players with the same name
            logger.error("No player with API id %s. Skipped.", player_api_id)
            continue

        player_attributes = _attributes_for_player(player, df_attributes)

        player_data = fetcher.get_gameweek_data_for_player(player_api_id)
        # now loop through all the matches that player played in
        for gameweek, results in player_data.items():
            if gameweek not in range(gameweek_start, gameweek_end):
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

                ps = (
                    get_player_scores(
                        fixture=fixture, player=player, dbsession=dbsession
                    )
                    or PlayerScore()
                )
                _fill_score(
                    ps,
                    result,
                    _API_STATS,
                    extended_feats,
                    player_attributes,
                    fixture,
                    fixture.result,
                    opponent,
                    player,
                    played_for,
                    dbsession,
                )
                logger.debug(ps)
    dbsession.commit()


def make_playerscore_table(
    seasons: list[str] | None = None, dbsession: Session | None = None
) -> None:
    dbsession = get_session(dbsession)
    if not seasons:
        seasons = default_seasons()
    for season in sort_seasons(seasons):
        if season == CURRENT_SEASON:
            # current season - use API
            fill_playerscores_from_api(CURRENT_SEASON, dbsession=dbsession)
        else:
            with data_file(f"player_details_{season}.json").open() as f:
                input_data = json.load(f)
            fill_playerscores_from_json(input_data, season, dbsession=dbsession)
