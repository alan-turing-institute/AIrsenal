"""
Scoring a fitted model against what actually happened.

The other half of making models pluggable: `prediction/protocols.py` says what a
model has to provide, and this says whether one is any good. Both scorers are
typed against the protocols, so anything in `TEAM_MODELS` or `PLAYER_MODELS` -
or a class written in a notebook and never registered - can be scored the same
way and compared on one number.

The number is a held-out log probability: how much probability the model put on
the thing that actually happened. Higher is better, it is always negative, and
it is only comparable between models scored over the same observations - so
`ModelScore` carries the count as well as the total.

Nothing here fits a model to data it has already seen. `backtest_team_model`
walks the season forward, fitting only on matches before the gameweek it scores.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from math import lgamma

import numpy as np
import pandas as pd
from sqlalchemy.orm.session import Session

from airsenal.core.logging import get_logger
from airsenal.db.models import Fixture, PlayerScore
from airsenal.db.queries.fixtures import get_fixtures_for_gameweeks
from airsenal.db.queries.scores import get_player_scores_for_gameweeks
from airsenal.game.enums import Position
from airsenal.game.scoring import MAX_GOALS
from airsenal.prediction.protocols import PlayerModel, TeamModel

logger = get_logger(__name__)

# A model that gives an observed outcome no probability at all would otherwise
# score minus infinity, which makes every such model equally bad. Flooring keeps
# the comparison ordered while still punishing it heavily.
MIN_PROBABILITY = 1e-12


def _log(probability: float) -> float:
    return float(np.log(max(probability, MIN_PROBABILITY)))


@dataclass(frozen=True)
class ModelScore:
    """
    How much probability a model put on what actually happened.

    Higher (less negative) is better. Only comparable against another score over
    the same observations, which is why `n_observations` travels with the total
    and why adding two scores adds both.
    """

    total_log_probability: float = 0.0
    n_observations: int = 0
    # Observations the model could not be asked about - a fixture with no result,
    # a player the fitted frame does not cover. Reported rather than hidden: a
    # score over three observations is not a verdict on a model.
    n_skipped: int = 0

    @property
    def mean_log_probability(self) -> float:
        """The per-observation score, which *is* comparable across sample sizes."""
        if not self.n_observations:
            return 0.0
        return self.total_log_probability / self.n_observations

    def __add__(self, other: "ModelScore") -> "ModelScore":
        return ModelScore(
            total_log_probability=self.total_log_probability
            + other.total_log_probability,
            n_observations=self.n_observations + other.n_observations,
            n_skipped=self.n_skipped + other.n_skipped,
        )


def score_team_model(
    model: TeamModel, fixtures: Iterable[Fixture], max_goals: int = MAX_GOALS
) -> ModelScore:
    """
    How well `model` predicted the scorelines of `fixtures` that have been played.

    Each side's goal count is scored against that side's marginal distribution,
    from `predict_score_n_proba` - the one scoreline method the protocol names,
    so every team model is scored the same way. A model whose joint distribution
    is more than the product of its marginals, as Dixon-Coles' low-score
    correction makes it, is not credited for that here; the point is to rank
    models against each other on equal terms, not to report a model's own
    likelihood.

    Args:
        max_goals: Goal counts above this are scored against the last bin, so a
            freak result cannot be given zero probability by a truncated support.
    """
    goals = np.arange(max_goals + 1)
    total = 0.0
    scored = 0
    skipped = 0
    for fixture in fixtures:
        if fixture.result is None:
            skipped += 1
            continue
        home_goals = min(int(fixture.result.home_score), max_goals)
        away_goals = min(int(fixture.result.away_score), max_goals)
        home_probs = np.asarray(
            model.predict_score_n_proba(
                goals, fixture.home_team, fixture.away_team, home=True
            )
        ).ravel()
        away_probs = np.asarray(
            model.predict_score_n_proba(
                goals, fixture.away_team, fixture.home_team, home=False
            )
        ).ravel()
        total += _log(float(home_probs[home_goals])) + _log(
            float(away_probs[away_goals])
        )
        scored += 1
    return ModelScore(
        total_log_probability=total, n_observations=scored, n_skipped=skipped
    )


def player_outcome_probability(
    goals: int,
    assists: int,
    team_goals: int,
    minutes: int,
    probabilities: Sequence[float],
) -> float:
    """
    The model's probability of one player's involvement in one match's goals.

    The three fitted probabilities are per goal and for a full match, so they are
    scaled by the fraction played before the multinomial is evaluated. Returns
    1.0 - a certainty, contributing nothing to a log score - when the team did
    not score or the player did not appear, because neither says anything about
    how a team's goals are shared out.
    """
    if team_goals <= 0 or minutes <= 0:
        return 1.0
    neither = team_goals - goals - assists
    if neither < 0:
        # own goals and data errors both land here; the observation is not one
        # the model claims to describe
        return 1.0
    played = min(minutes, 90) / 90.0
    prob_score = played * probabilities[0]
    prob_assist = played * probabilities[1]
    prob_neither = 1.0 - prob_score - prob_assist
    if prob_neither < 0:
        return MIN_PROBABILITY
    # multinomial pmf, written out rather than imported: scipy is not a
    # dependency of the prediction layer and this is three terms
    log_coefficient = (
        lgamma(team_goals + 1)
        - lgamma(goals + 1)
        - lgamma(assists + 1)
        - lgamma(neither + 1)
    )
    log_p = (
        log_coefficient
        + goals * _log(prob_score)
        + assists * _log(prob_assist)
        + neither * _log(prob_neither)
    )
    return float(np.exp(log_p))


def score_player_model(
    probabilities: pd.DataFrame, player_scores: Iterable[PlayerScore]
) -> ModelScore:
    """
    How well a fitted player model predicted who scored and assisted.

    Args:
        probabilities: What `fit_player_data` returns - one row per player id,
            with prob_score, prob_assist and prob_neither.
        player_scores: The performances to score against. Rows for a player the
            frame does not cover are skipped rather than guessed at, as are
            matches in which the player's team did not score or the player did
            not appear - neither says anything about how goals are shared out.
    """
    # Lifted out of the frame once rather than looked up per row: a season of
    # performances is tens of thousands of rows, and it types cleanly.
    columns = ["prob_score", "prob_assist", "prob_neither"]
    by_player: dict[int, list[float]] = {
        int(player_id): [float(value) for value in row]
        for player_id, row in zip(
            probabilities.index,
            np.asarray(probabilities[columns], dtype=float),
            strict=True,
        )
    }

    total = 0.0
    scored = 0
    skipped = 0
    for ps in player_scores:
        if ps.player_id not in by_player:
            skipped += 1
            continue
        if ps.fixture.home_team == ps.opponent:
            team_goals = int(ps.result.away_score)
        elif ps.fixture.away_team == ps.opponent:
            team_goals = int(ps.result.home_score)
        else:
            msg = f"opponent {ps.opponent} is not in fixture {ps.fixture}"
            raise ValueError(msg)
        if team_goals <= 0 or not ps.minutes:
            skipped += 1
            continue
        total += _log(
            player_outcome_probability(
                goals=int(ps.goals or 0),
                assists=int(ps.assists or 0),
                team_goals=team_goals,
                minutes=int(ps.minutes),
                probabilities=by_player[ps.player_id],
            )
        )
        scored += 1
    return ModelScore(
        total_log_probability=total, n_observations=scored, n_skipped=skipped
    )


def backtest_team_model(
    build: Callable[[], TeamModel],
    season: str,
    dbsession: Session,
    gameweeks: Sequence[int],
    horizon: int = 1,
) -> ModelScore:
    """
    Score a team model on gameweeks it was not fitted to, walking the season forward.

    For each gameweek in `gameweeks`, fit a fresh model on everything before it
    and score the next `horizon` gameweeks. `build` is called once per gameweek
    because a model is fitted in place - the entries of `TEAM_MODELS` are exactly
    this shape, so `backtest_team_model(TEAM_MODELS["extended"], ...)` works, and
    so does a lambda over a class that no table knows about.
    """
    # imported here rather than at module scope: team_models.fitting imports the
    # table, and the table imports bpl, which imports jax - seconds of import
    # time that scoring a player model should not pay for.
    from airsenal.prediction.team_models.fitting import (  # noqa: PLC0415
        get_fitted_team_model,
    )

    score = ModelScore()
    for gameweek in gameweeks:
        evaluation_gameweeks = list(range(gameweek, gameweek + horizon))
        fixtures = get_fixtures_for_gameweeks(
            evaluation_gameweeks, season=season, dbsession=dbsession
        )
        if not fixtures:
            logger.info("No fixtures for %s GW%s, skipping", season, gameweek)
            continue
        model = get_fitted_team_model(season, gameweek, dbsession, model=build())
        score += score_team_model(model, fixtures)
        logger.info(
            "GW%s: mean log probability %.4f over %s fixtures",
            gameweek,
            score.mean_log_probability,
            score.n_observations,
        )
    return score


def backtest_player_model(
    build: Callable[[], PlayerModel],
    season: str,
    dbsession: Session,
    gameweeks: Sequence[int],
    positions: Sequence[Position] | None = None,
    horizon: int = 1,
) -> ModelScore:
    """
    Score a player model on gameweeks it was not fitted to, walking forward.

    The player-side twin of `backtest_team_model`. A model is fitted once per
    position, because that is how `fit_player_data` works - a forward's share of
    his team's goals says nothing about a goalkeeper's - and the fitted frames
    are scored together.

    Args:
        positions: Which positions to fit and score. Every position by default.
    """
    # as in backtest_team_model: fitting pulls in features.py, and through it the
    # rest of the prediction stack, which a caller only scoring a team model
    # should not pay for
    from airsenal.prediction.player_models.fitting import (  # noqa: PLC0415
        fit_player_data,
    )

    positions = list(positions) if positions is not None else list(Position)
    score = ModelScore()
    for gameweek in gameweeks:
        evaluation_gameweeks = list(range(gameweek, gameweek + horizon))
        player_scores = get_player_scores_for_gameweeks(
            evaluation_gameweeks, season=season, dbsession=dbsession
        )
        if not player_scores:
            logger.info("No performances for %s GW%s, skipping", season, gameweek)
            continue
        probabilities = pd.concat(
            [
                fit_player_data(
                    position, season, gameweek, model=build(), dbsession=dbsession
                )
                for position in positions
            ]
        )
        score += score_player_model(probabilities, player_scores)
        logger.info(
            "GW%s: mean log probability %.4f over %s performances",
            gameweek,
            score.mean_log_probability,
            score.n_observations,
        )
    return score
