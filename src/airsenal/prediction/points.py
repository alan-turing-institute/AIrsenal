"""Turning fitted models into predicted points for a player in a fixture."""

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import multinomial
from sqlalchemy.orm import Session

from airsenal.core.logging import get_logger
from airsenal.core.scoring import (
    MIN_MINUTES_FULL,
    MIN_MINUTES_SHORT,
    get_appearance_points,
    points_for_assist,
    points_for_cs,
    points_for_goal,
)
from airsenal.db.models import Fixture, Player, PlayerPrediction
from airsenal.db.queries.absences import was_historic_absence
from airsenal.db.queries.fixtures import get_fixtures_for_player
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.db.queries.players import get_player, list_players
from airsenal.db.session import get_session
from airsenal.prediction.features import fit_player_data
from airsenal.prediction.minutes import get_recent_minutes_for_player
from airsenal.prediction.protocols import PlayerModel

logger = get_logger(__name__)


def get_attacking_points(
    position: str,
    minutes: int | float,
    team_score_prob: dict[int, float],
    player_prob: pd.Series,
) -> float:
    """
    Calculate expected attacking points (goals and assists) for a player.
    """
    if minutes == 0.0:
        return 0.0

    pr_score = (minutes / 90.0) * player_prob["prob_score"]
    pr_assist = (minutes / 90.0) * player_prob["prob_assist"]
    pr_neither = 1.0 - pr_score - pr_assist
    multinom_probs = (pr_score, pr_assist, pr_neither)

    def _get_partitions(n: int) -> list[list[int]]:
        partitions = []
        for i in range(n + 1):
            for j in range(n - i + 1):
                partitions.append([i, j, n - i - j])
        return partitions

    def _get_partition_score(partition: list[int]) -> int:
        return (
            points_for_goal[position] * partition[0] + points_for_assist * partition[1]
        )

    exp_points = 0.0
    for ngoals, score_n_prob in team_score_prob.items():
        if ngoals > 0:
            partitions = _get_partitions(ngoals)
            probabilities = multinomial.pmf(
                partitions, n=[ngoals] * len(partitions), p=multinom_probs
            )
            scores = map(_get_partition_score, partitions)
            exp_score_inner = sum(
                pi * si for pi, si in zip(probabilities, scores, strict=False)
            )
            exp_points += exp_score_inner * score_n_prob
    return exp_points


def get_defending_points(
    position: str, minutes: int | float, team_concede_prob: dict[int, float]
) -> float:
    """
    Calculate expected defending points (clean sheets and conceded goals) for a player.
    """
    if position == "FWD" or minutes == 0.0:
        return 0.0

    defending_points = 0.0
    if minutes >= MIN_MINUTES_FULL:
        defending_points = points_for_cs[position] * team_concede_prob[0]

    if position in ["DEF", "GK"]:
        defending_points -= sum(
            (ngoals // 2) * (minutes / 90) * concede_n_prob
            for ngoals, concede_n_prob in team_concede_prob.items()
        )
    return defending_points


def get_bonus_points(
    player_id: int, minutes: int | float, df_bonus: tuple[pd.Series, pd.Series]
) -> float:
    """
    Calculate expected bonus points based on played minutes.
    """
    if minutes >= MIN_MINUTES_FULL:
        return float(df_bonus[0].get(player_id, 0.0))
    if minutes >= MIN_MINUTES_SHORT:
        return float(df_bonus[1].get(player_id, 0.0))
    return 0.0


def get_def_con_points(
    player_id: int, minutes: int | float, df_def_con: tuple[pd.Series, pd.Series]
) -> float:
    """
    Calculate expected defensive contribution points based on played minutes.
    """
    if minutes >= MIN_MINUTES_FULL:
        return float(df_def_con[0].get(player_id, 0.0))
    if minutes >= MIN_MINUTES_SHORT:
        return float(df_def_con[1].get(player_id, 0.0))
    return 0.0


def get_save_points(
    position: str, player_id: int, minutes: int | float, df_saves: pd.Series
) -> float:
    """
    Calculate expected save points for goalkeepers.
    """
    if position != "GK":
        return 0.0
    if minutes >= MIN_MINUTES_FULL:
        return float(df_saves.get(player_id, 0.0))
    return 0.0


def get_card_points(player_id: int, minutes: int | float, df_cards: pd.Series) -> float:
    """
    Calculate expected penalty points for yellow and red cards.
    """
    if minutes >= MIN_MINUTES_SHORT:
        return float(df_cards.get(player_id, 0.0))
    return 0.0


def calc_predicted_points_for_player(
    player: Player | str | int,
    fixture_goal_probs: dict[int, dict[str, dict[int, float]]],
    df_player: dict[str, pd.DataFrame],
    df_bonus: tuple[pd.Series, pd.Series] | None,
    df_saves: pd.Series | None,
    df_cards: pd.Series | None,
    df_def_con: tuple[pd.Series, pd.Series] | None,
    season: str,
    gameweeks: list[int] | None = None,
    fixtures_behind: int | None = None,
    min_fixtures_behind: int = 3,
    tag: str = "",
    dbsession: Session | None = None,
) -> list[PlayerPrediction]:
    """
    Calculate predicted total points for a single player across target gameweeks.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    if isinstance(player, str | int):
        p = get_player(player, dbsession=dbsession)
        if p is None:
            msg = f"Player {player} not found in database"
            raise ValueError(msg)
        player = p

    if not gameweeks:
        gameweeks = list(range(next_gameweek(), min(next_gameweek() + 3, 38)))

    if fixtures_behind is None:
        fixtures_behind = len(gameweeks)

    fixtures_behind = max(fixtures_behind, min_fixtures_behind)

    team = player.team(season, gameweeks[0])
    position = player.position(season)
    if position is None or team is None:
        msg = f"Player {player} has missing team or position for season {season}"
        raise ValueError(msg)

    fixtures = get_fixtures_for_player(
        player, season, gameweeks=gameweeks, dbsession=dbsession
    )

    player_prob = df_player[position].loc[player.player_id]
    if not isinstance(player_prob, pd.Series):
        msg = f"player_prob for {player} is not a Series, but {type(player_prob)}"
        raise RuntimeError(msg)

    recent_minutes = get_recent_minutes_for_player(
        player,
        n_matches_to_use=fixtures_behind,
        season=season,
        last_gw=min(gameweeks) - 1,
        dbsession=dbsession,
    )
    if len(recent_minutes) == 0:
        msg = "Recent minutes is empty."
        raise ValueError(msg)

    expected_points = defaultdict(float)
    predictions = []

    for fixture in fixtures:
        gameweek = fixture.gameweek
        if gameweek is None:
            logger.warning("Skipping fixture %s with no gameweek", fixture)
            continue

        is_home = fixture.home_team == team
        opponent = fixture.away_team if is_home else fixture.home_team
        team_score_prob = fixture_goal_probs[fixture.fixture_id][team]
        team_concede_prob = fixture_goal_probs[fixture.fixture_id][opponent]

        points = 0.0
        expected_points[gameweek] = points

        if (
            sum(recent_minutes) == 0
            or player.is_injured_or_suspended(season, gameweeks[0], gameweek)
            or was_historic_absence(
                player,
                gameweek=gameweek,
                season=season,
                dbsession=dbsession,
            )
        ):
            points = 0.0
        else:
            points = 0
            for mins in recent_minutes:
                points += (
                    get_appearance_points(mins)
                    + get_attacking_points(
                        position,
                        mins,
                        team_score_prob,
                        player_prob,
                    )
                    + get_defending_points(position, mins, team_concede_prob)
                )
                if df_bonus is not None:
                    points += get_bonus_points(player.player_id, mins, df_bonus)
                if df_cards is not None:
                    points += get_card_points(player.player_id, mins, df_cards)
                if df_saves is not None:
                    points += get_save_points(
                        position, player.player_id, mins, df_saves
                    )
                if df_def_con is not None:
                    points += get_def_con_points(player.player_id, mins, df_def_con)

            points /= len(recent_minutes)

        if np.isnan(points):
            msg = f"nan points for {player} {fixture} {points} {tag}"
            raise ValueError(msg)

        predictions.append(make_prediction(player, fixture, points, tag))
        expected_points[gameweek] += points
    return predictions


def calc_predicted_points_for_pos(
    pos: str,
    fixture_goal_probs: dict[int, dict[str, dict[int, float]]],
    df_bonus: tuple[pd.Series, pd.Series] | None,
    df_saves: pd.Series | None,
    df_cards: pd.Series | None,
    df_def_con: tuple[pd.Series, pd.Series] | None,
    season: str,
    gameweeks: list[int],
    tag: str,
    model: PlayerModel | None = None,
    dbsession: Session | None = None,
) -> dict[int, list[PlayerPrediction]]:
    """
    Calculate predicted points for all players in a specific position.
    """
    dbsession = dbsession if dbsession is not None else get_session()
    df_player = {pos: fit_player_data(pos, season, min(gameweeks), model, dbsession)}
    return {
        player.player_id: calc_predicted_points_for_player(
            player=player,
            fixture_goal_probs=fixture_goal_probs,
            df_player=df_player,
            df_bonus=df_bonus,
            df_saves=df_saves,
            df_cards=df_cards,
            df_def_con=df_def_con,
            season=season,
            gameweeks=gameweeks,
            tag=tag,
            dbsession=dbsession,
        )
        for player in list_players(
            position=pos, season=season, gameweek=min(gameweeks), dbsession=dbsession
        )
    }


def make_prediction(
    player: Player, fixture: Fixture, points: float, tag: str
) -> PlayerPrediction:
    """
    Instantiate and populate a PlayerPrediction schema object.
    """
    pp = PlayerPrediction()
    pp.predicted_points = points
    pp.tag = tag
    pp.player = player
    pp.fixture = fixture
    return pp
