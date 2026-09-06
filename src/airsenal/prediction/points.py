"""Turning fitted models into predicted points for a player in a fixture."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from airsenal.core.logging import get_logger
from airsenal.db.models import Fixture, Player, PlayerPrediction
from airsenal.db.queries.absences import was_historic_absence
from airsenal.db.queries.fixtures import get_fixtures_for_player
from airsenal.db.queries.players import require_player
from airsenal.db.session import get_session
from airsenal.prediction.protocols import (
    ComponentRequest,
    MinutesModel,
    MinutesRequest,
    PointComponent,
)

logger = get_logger(__name__)


def calc_predicted_points_for_player(
    player: Player | str | int,
    fixture_goal_probs: dict[int, dict[str, dict[int, float]]],
    df_player: dict[str, pd.DataFrame],
    components: Sequence[PointComponent],
    *,
    minutes_model: MinutesModel,
    gameweeks: list[int],
    tag: str = "",
    season: str,
    dbsession: Session | None = None,
) -> list[PlayerPrediction]:
    """Calculate predicted total points for a single player across target gameweeks."""
    dbsession = dbsession if dbsession is not None else get_session()
    if isinstance(player, str | int):
        player = require_player(player, dbsession=dbsession)

    # The gameweek we are predicting *from*. Everything about the player is read
    # as at this gameweek, whichever gameweek of the window the fixture is in.
    root_gameweek = min(gameweeks)

    team = player.team(root_gameweek, season)
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

    minutes = minutes_model.predict(
        MinutesRequest(
            player=player,
            gameweek=root_gameweek,
            season=season,
            n_gameweeks=len(gameweeks),
            dbsession=dbsession,
        )
    )

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

        # The three fixture-varying values are bound as defaults rather than
        # closed over: ruff's B023 is right that a closure over a loop variable
        # is a trap, even though this one is called before the next iteration.
        def points_for_minutes(
            mins: float,
            position: str = position,
            team_score_prob: dict[int, float] = team_score_prob,
            team_concede_prob: dict[int, float] = team_concede_prob,
        ) -> float:
            """Every component of a score, for one number of minutes played."""
            request = ComponentRequest(
                player_id=player.player_id,
                position=position,
                minutes=mins,
                team_score_probability=team_score_prob,
                team_concede_probability=team_concede_prob,
                prob_score=float(player_prob["prob_score"]),
                prob_assist=float(player_prob["prob_assist"]),
            )
            return sum(component.expected_points(request) for component in components)

        if (
            minutes.expected_minutes == 0.0
            or player.is_injured_or_suspended(season, root_gameweek, gameweek)
            or was_historic_absence(
                player,
                current_gameweek=root_gameweek,
                fixture_gameweek=gameweek,
                season=season,
                dbsession=dbsession,
            )
        ):
            points = 0.0
        else:
            points = minutes.expectation(points_for_minutes)

        if np.isnan(points):
            msg = f"nan points for {player} {fixture} {points} {tag}"
            raise ValueError(msg)

        predictions.append(make_prediction(player, fixture, points, tag))
    return predictions


def make_prediction(
    player: Player, fixture: Fixture, points: float, tag: str
) -> PlayerPrediction:
    """Instantiate and populate a PlayerPrediction schema object."""
    pp = PlayerPrediction()
    pp.predicted_points = points
    pp.tag = tag
    pp.player = player
    pp.fixture = fixture
    return pp
