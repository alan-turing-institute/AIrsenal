"""
Scoring a fitted model against what actually happened.

Two metrics, because not every model is probabilistic. `ModelScore` is a log
probability - how much probability the model put on the thing that actually
happened; higher is better, it is always negative, and only models scored the
same way over the same matches can be compared. `PointsScore` is the error in
the predicted points themselves, which asks nothing of a model except a number
per player per fixture.

`ModelScore` judges one model, and `PointsScore` judges the whole points
calculation the models feed. A replay's `mean_absolute_error` is a third thing
again: the error for the squad the optimizer chose, rather than over every
player predicted.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from math import lgamma

import numpy as np
import pandas as pd
from sqlalchemy.orm.session import Session

from airsenal.core.logging import get_logger
from airsenal.db.models import Fixture, PlayerPrediction, PlayerScore
from airsenal.db.queries.fixtures import get_fixtures_for_gameweeks
from airsenal.db.queries.predictions import get_predictions_for_gameweeks
from airsenal.db.queries.scores import get_player_scores_for_gameweeks
from airsenal.game.enums import Position
from airsenal.game.scoring import MAX_GOALS, MIN_MINUTES_FULL
from airsenal.prediction.point_components import actual_component_points
from airsenal.prediction.protocols import (
    MinutesModel,
    MinutesRequest,
    PlayerModel,
    PointsFitRequest,
    PointsModel,
    PointsPrediction,
    PointsRequest,
    ScorelineTeamModel,
)

logger = get_logger(__name__)

# A model that gives an observed outcome no probability at all would otherwise
# score minus infinity, which makes every such model equally bad. Flooring keeps
# the comparison ordered while still punishing it heavily.
MIN_PROBABILITY = 1e-12

# Two players can only be ranked right or wrong by luck, so a correlation over
# fewer than this many observations says nothing about the model.
MIN_RANKED = 3

# What FPL pays for is not the minute count but which side of these a player
# lands on: nothing for not appearing, one point for appearing, two and a
# clean-sheet chance from `MIN_MINUTES_FULL`.
MINUTES_BANDS = (
    (0.0, 1.0),
    (1.0, float(MIN_MINUTES_FULL)),
    (float(MIN_MINUTES_FULL), float("inf")),
)


def _log(probability: float) -> float:
    return float(np.log(max(probability, MIN_PROBABILITY)))


@dataclass(frozen=True)
class ModelScore:
    """
    How much probability a model put on what actually happened.

    Higher (less negative) is better. Only comparable against another score over
    the same observations.
    """

    total_log_probability: float = 0.0
    n_observations: int = 0
    # Observations the model could not be asked about - a fixture with no result,
    # a player the fitted frame does not cover.
    n_skipped: int = 0

    @property
    def mean_log_probability(self) -> float:
        """The per-observation score, which is more comparable across sample sizes."""
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


@dataclass(frozen=True)
class PointsScore:
    """
    How close predicted points came to what players actually scored.

    The companion to `ModelScore` for a model that predicts points rather than
    probabilities: nothing here asks a model to be probabilistic, or to predict
    any particular quantity on the way to a score.

    Lower is better for the two errors and higher is better for the rank
    correlation, so unlike `ModelScore` there is no single number to order
    models by. Only comparable against another score over the same observations.
    """

    total_absolute_error: float = 0.0
    total_squared_error: float = 0.0
    n_observations: int = 0
    # The same error over the performances where the player actually appeared.
    # Most non-appearances are predicted at exactly zero and so contribute no
    # error at all, which pulls `mean_absolute_error` well below the error on
    # the players a squad is actually picked from - see `mean_absolute_error_played`.
    total_absolute_error_played: float = 0.0
    n_played: int = 0
    # Observations that could not be scored, in either direction: a prediction
    # whose fixture nobody has a performance for, and a performance the run
    # predicted nothing for. A large count means the two sets barely overlap,
    # which makes the errors below a statement about very few players.
    n_skipped: int = 0
    # Summed over the calls that produced a correlation rather than over
    # observations: a ranking is a property of a set of players, so the mean is
    # per scored group - one gameweek, when `backtest_points` drives it.
    total_rank_correlation: float = 0.0
    n_ranked: int = 0

    @property
    def mean_absolute_error(self) -> float:
        """The per-observation error, in points."""
        if not self.n_observations:
            return 0.0
        return self.total_absolute_error / self.n_observations

    @property
    def mean_absolute_error_played(self) -> float:
        """
        The per-observation error over players who appeared, in points.

        The more sensitive of the two: a change that only affects players who
        start moves this by around two and a half times what it moves
        `mean_absolute_error`, because six observations in ten are a
        non-appearance predicted at zero.
        """
        if not self.n_played:
            return 0.0
        return self.total_absolute_error_played / self.n_played

    @property
    def root_mean_squared_error(self) -> float:
        """Like `mean_absolute_error`, but a big miss counts for more."""
        if not self.n_observations:
            return 0.0
        return float(np.sqrt(self.total_squared_error / self.n_observations))

    @property
    def mean_rank_correlation(self) -> float:
        """
        Average Spearman correlation between predicted and actual points.

        Zero when nothing could be ranked. This is the number a transfer search
        depends on most: it picks players in order, so the order matters more
        than the level.
        """
        if not self.n_ranked:
            return 0.0
        return self.total_rank_correlation / self.n_ranked

    def __add__(self, other: "PointsScore") -> "PointsScore":
        return PointsScore(
            total_absolute_error=self.total_absolute_error + other.total_absolute_error,
            total_squared_error=self.total_squared_error + other.total_squared_error,
            n_observations=self.n_observations + other.n_observations,
            total_absolute_error_played=self.total_absolute_error_played
            + other.total_absolute_error_played,
            n_played=self.n_played + other.n_played,
            n_skipped=self.n_skipped + other.n_skipped,
            total_rank_correlation=self.total_rank_correlation
            + other.total_rank_correlation,
            n_ranked=self.n_ranked + other.n_ranked,
        )


@dataclass(frozen=True)
class MinutesScore:
    """
    How well a minutes model predicted how long players were on the pitch.

    Three views of the same predictions, because a minutes error matters in a
    particular way: `mean_absolute_error` is in minutes, `band_accuracy` and
    `mean_log_probability` are over `MINUTES_BANDS`, which is what the scoring
    rules actually turn minutes into.
    """

    total_absolute_error: float = 0.0
    n_observations: int = 0
    n_correct_band: int = 0
    total_log_probability: float = 0.0
    # Performances the model gave no probability at all, which `MIN_PROBABILITY`
    # floors rather than scoring as minus infinity. A model built from a sample
    # of past appearances does this whenever the band that happened is one it
    # did not sample, and each one costs the log score about 27 - so read
    # `mean_log_probability` next to this, not on its own.
    n_impossible: int = 0
    # Performances the model could not be asked about - a player it knows
    # nothing about, or a fixture with no gameweek.
    n_skipped: int = 0

    @property
    def mean_absolute_error(self) -> float:
        """The per-performance error, in minutes."""
        if not self.n_observations:
            return 0.0
        return self.total_absolute_error / self.n_observations

    @property
    def impossible_fraction(self) -> float:
        """The share of performances the model said could not happen."""
        if not self.n_observations:
            return 0.0
        return self.n_impossible / self.n_observations

    @property
    def band_accuracy(self) -> float:
        """The fraction of performances put in the right band. Higher is better."""
        if not self.n_observations:
            return 0.0
        return self.n_correct_band / self.n_observations

    @property
    def mean_log_probability(self) -> float:
        """
        The per-performance log probability of the band that happened.

        Comparable with `ModelScore`: higher is better, and it rewards a model
        for being uncertain about a rotation risk rather than confidently wrong.
        """
        if not self.n_observations:
            return 0.0
        return self.total_log_probability / self.n_observations

    def __add__(self, other: "MinutesScore") -> "MinutesScore":
        return MinutesScore(
            total_absolute_error=self.total_absolute_error + other.total_absolute_error,
            n_observations=self.n_observations + other.n_observations,
            n_correct_band=self.n_correct_band + other.n_correct_band,
            total_log_probability=self.total_log_probability
            + other.total_log_probability,
            n_impossible=self.n_impossible + other.n_impossible,
            n_skipped=self.n_skipped + other.n_skipped,
        )


def minutes_band(minutes: float) -> int:
    """Which of `MINUTES_BANDS` these minutes fall in."""
    for index, (low, high) in enumerate(MINUTES_BANDS):
        if low <= minutes < high:
            return index
    msg = f"{minutes} minutes is in no band"
    raise ValueError(msg)


def score_team_model(
    model: ScorelineTeamModel, fixtures: Iterable[Fixture], max_goals: int = MAX_GOALS
) -> ModelScore:
    """
    How well `model` predicted the scorelines of `fixtures` that have been played.

    Each side's goal count is scored against that side's marginal distribution,
    from `predict_score_n_proba`.

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
    not score or the player did not appear.
    """
    if team_goals <= 0 or minutes <= 0:
        return 1.0
    neither = team_goals - goals - assists
    if neither < 0:
        # data errors and own goals can cause this
        return 1.0
    played = min(minutes, 90) / 90.0
    prob_score = played * probabilities[0]
    prob_assist = played * probabilities[1]
    prob_neither = 1.0 - prob_score - prob_assist
    if prob_neither < 0:
        return MIN_PROBABILITY
    # multinomial pmf
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


def team_goals_in(score: PlayerScore) -> int:
    """How many goals this player's own team scored in the match."""
    if score.fixture.home_team == score.opponent:
        return int(score.result.away_score)
    if score.fixture.away_team == score.opponent:
        return int(score.result.home_score)
    msg = f"opponent {score.opponent} is not in fixture {score.fixture}"
    raise ValueError(msg)


@dataclass(frozen=True)
class InvolvementScore:
    """
    How close a player model's goal shares came to what the players actually did.

    The error form of `score_player_model`, over the same predictions: that one
    is a log probability and needs a model to be probabilistic, this one asks
    only for a share, so a model that arrives at one without a posterior can be
    judged too. Lower is better.

    Conditioned on what actually happened around the player - the minutes played
    and the goals the team scored - so it measures the share alone, with no
    minutes prediction mixed in. That is what makes it comparable across models
    that predict minutes differently.
    """

    total_absolute_error_goals: float = 0.0
    total_absolute_error_assists: float = 0.0
    n_observations: int = 0
    # Performances the model says nothing about: a player it was not fitted for,
    # a match the team did not score in, or one the player did not appear in.
    n_skipped: int = 0

    @property
    def mean_absolute_error_goals(self) -> float:
        """The per-performance error in goals scored."""
        if not self.n_observations:
            return 0.0
        return self.total_absolute_error_goals / self.n_observations

    @property
    def mean_absolute_error_assists(self) -> float:
        """The per-performance error in assists."""
        if not self.n_observations:
            return 0.0
        return self.total_absolute_error_assists / self.n_observations

    def __add__(self, other: "InvolvementScore") -> "InvolvementScore":
        return InvolvementScore(
            total_absolute_error_goals=self.total_absolute_error_goals
            + other.total_absolute_error_goals,
            total_absolute_error_assists=self.total_absolute_error_assists
            + other.total_absolute_error_assists,
            n_observations=self.n_observations + other.n_observations,
            n_skipped=self.n_skipped + other.n_skipped,
        )


def score_involvement_error(
    probabilities: pd.DataFrame, player_scores: Iterable[PlayerScore]
) -> InvolvementScore:
    """
    The error in a fitted player model's goal shares, given what actually happened.

    Args:
        probabilities: What `fit_player_data` returns - one row per player id,
            with prob_score, prob_assist and prob_neither.
        player_scores: The performances to score against.
    """
    shares = {
        int(player_id): (float(row[0]), float(row[1]))
        for player_id, row in zip(
            probabilities.index,
            np.asarray(probabilities[["prob_score", "prob_assist"]], dtype=float),
            strict=True,
        )
    }
    total = InvolvementScore()
    for ps in player_scores:
        if ps.player_id not in shares:
            total += InvolvementScore(n_skipped=1)
            continue
        prob_score, prob_assist = shares[ps.player_id]
        total += involvement_error(prob_score, prob_assist, ps)
    return total


def involvement_error(
    prob_score: float, prob_assist: float, score: PlayerScore
) -> InvolvementScore:
    """
    The error in one player's shares, given the minutes and goals that happened.

    Skipped - and so weightless - when the team did not score or the player did
    not appear, because a share of nothing says nothing about the model.
    """
    team_goals = team_goals_in(score)
    if team_goals <= 0 or not score.minutes:
        return InvolvementScore(n_skipped=1)
    played = min(int(score.minutes), 90) / 90.0
    return InvolvementScore(
        total_absolute_error_goals=abs(
            played * prob_score * team_goals - int(score.goals or 0)
        ),
        total_absolute_error_assists=abs(
            played * prob_assist * team_goals - int(score.assists or 0)
        ),
        n_observations=1,
    )


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
            not appear.
    """
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
        team_goals = team_goals_in(ps)
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


def _rank_correlation(predicted: np.ndarray, actual: np.ndarray) -> float | None:
    """
    Spearman correlation between predicted and actual points, or None if undefined.

    Undefined below `MIN_RANKED` observations, and where either side is constant
    - a run that predicted the same points for everyone has no ranking to be
    right or wrong about.
    """
    if len(predicted) < MIN_RANKED:
        return None
    predicted_ranks = pd.Series(predicted).rank().to_numpy()
    actual_ranks = pd.Series(actual).rank().to_numpy()
    if predicted_ranks.std() == 0.0 or actual_ranks.std() == 0.0:
        return None
    return float(np.corrcoef(predicted_ranks, actual_ranks)[0, 1])


def score_points_predictions(
    predictions: Iterable[PlayerPrediction],
    player_scores: Iterable[PlayerScore],
) -> PointsScore:
    """
    How close a run's predictions came to the points the players actually scored.

    Matched on player and fixture, so a double gameweek is two observations
    rather than one, and no prediction is compared against a performance in a
    different match.

    Args:
        predictions: Rows a prediction run wrote, from
            `get_predictions_for_gameweeks`.
        player_scores: The performances to score against. An unmatched row on
            either side is skipped rather than treated as zero: the fixture may
            not have been played, or the player may not have been in any squad,
            and neither is the model being wrong.

    Returns:
        The errors summed over every matched observation, again over just the
        appearances among them, and one rank correlation over all of them - so
        the caller decides what a ranking is over by choosing what it passes in.
    """
    actual = {
        (score.player_id, score.fixture_id): (float(score.points), int(score.minutes))
        for score in player_scores
    }
    predicted_points: list[float] = []
    actual_points: list[float] = []
    played: list[bool] = []
    matched: set[tuple[int, int]] = set()
    skipped = 0
    for prediction in predictions:
        key = (prediction.player_id, prediction.fixture_id)
        if key not in actual:
            skipped += 1
            continue
        matched.add(key)
        points, minutes = actual[key]
        predicted_points.append(float(prediction.predicted_points))
        actual_points.append(points)
        played.append(minutes > 0)
    skipped += len(set(actual) - matched)

    predicted_array = np.asarray(predicted_points, dtype=float)
    actual_array = np.asarray(actual_points, dtype=float)
    errors = predicted_array - actual_array
    appeared = np.asarray(played, dtype=bool)
    correlation = _rank_correlation(predicted_array, actual_array)
    return PointsScore(
        total_absolute_error=float(np.abs(errors).sum()),
        total_squared_error=float(np.square(errors).sum()),
        n_observations=len(predicted_points),
        total_absolute_error_played=float(np.abs(errors[appeared]).sum()),
        n_played=int(appeared.sum()),
        n_skipped=skipped,
        total_rank_correlation=0.0 if correlation is None else correlation,
        n_ranked=0 if correlation is None else 1,
    )


def backtest_team_model(
    build: Callable[[], ScorelineTeamModel],
    season: str,
    dbsession: Session,
    gameweeks: Sequence[int],
    horizon: int = 1,
) -> ModelScore:
    """
    Score a team model on gameweeks it was not fitted to, walking the season forward.

    For each gameweek in `gameweeks`, fit a fresh model on everything before it
    and score the next `horizon` gameweeks. `build` is called once per gameweek
    because a model is fitted in place.
    """
    # deferred slow import
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
        model = get_fitted_team_model(gameweek, season, dbsession, model=build())
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

    The player-side twin of `backtest_team_model`.

    Args:
        positions: Which positions to fit and score. Every position by default.
    """
    # deferred slow import
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
                    position, gameweek, season, model=build(), dbsession=dbsession
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


def backtest_points(
    build: Callable[[], PointsModel] | None = None,
    *,
    season: str,
    dbsession: Session,
    gameweeks: Sequence[int],
    horizon: int = 1,
) -> PointsScore:
    """
    Score predicted points on gameweeks the models were not fitted to.

    For each gameweek in `gameweeks`, fit fresh models on everything before it,
    predict the next `horizon` gameweeks, and score those predictions against
    what the players actually scored. The whole points calculation is exercised
    rather than one model in isolation, so this is the number that says whether
    a change to any part of it helped.

    Unlike `backtest_team_model` and `backtest_player_model` this writes to the
    database: each gameweek's predictions are stored under their own tag,
    prefixed `Backtest_<season>_GW<gameweek>_`, as a replay's are.

    Args:
        build: Called once per gameweek, because a model is fitted in place. The
            package default points model when None.
    """
    # deferred slow import
    from airsenal.prediction.run import make_predictedscore_table  # noqa: PLC0415

    score = PointsScore()
    for gameweek in gameweeks:
        evaluation_gameweeks = list(range(gameweek, gameweek + horizon))
        player_scores = get_player_scores_for_gameweeks(
            evaluation_gameweeks, season=season, dbsession=dbsession
        )
        if not player_scores:
            logger.info("No performances for %s GW%s, skipping", season, gameweek)
            continue
        tag = make_predictedscore_table(
            gameweeks=evaluation_gameweeks,
            season=season,
            tag_prefix=f"Backtest_{season}_GW{gameweek}_",
            points_model=build() if build is not None else None,
            dbsession=dbsession,
        )
        predictions = get_predictions_for_gameweeks(
            evaluation_gameweeks, tag, season=season, dbsession=dbsession
        )
        score += score_points_predictions(predictions, player_scores)
        logger.info(
            "GW%s: mean absolute error %.4f over %s performances",
            gameweek,
            score.mean_absolute_error,
            score.n_observations,
        )
    return score


def score_minutes_model(
    model: MinutesModel,
    player_scores: Iterable[PlayerScore],
    n_gameweeks: int = 1,
    *,
    season: str,
    dbsession: Session,
) -> MinutesScore:
    """
    How well `model` predicted the minutes in `player_scores`.

    Each performance is predicted from its own gameweek, so the model never
    sees the match it is being scored on.

    Args:
        n_gameweeks: The window a run would have covered, which is what a model
            reading recent appearances uses to decide how far back to look.
    """
    total = MinutesScore()
    for score in player_scores:
        gameweek = score.fixture.gameweek
        if gameweek is None:
            total += MinutesScore(n_skipped=1)
            continue
        distribution = model.predict(
            MinutesRequest(
                player=score.player,
                root_gameweek=gameweek,
                fixture_gameweek=gameweek,
                season=season,
                n_gameweeks=n_gameweeks,
                dbsession=dbsession,
            )
        )
        actual = float(score.minutes)
        band = minutes_band(actual)
        low, high = MINUTES_BANDS[band]
        probability = distribution.probability_between(low, high)
        total += MinutesScore(
            total_absolute_error=abs(distribution.expected_minutes - actual),
            n_observations=1,
            n_correct_band=int(minutes_band(distribution.expected_minutes) == band),
            total_log_probability=_log(probability),
            n_impossible=int(probability == 0.0),
        )
    return total


def backtest_minutes_model(
    build: Callable[[], MinutesModel],
    season: str,
    dbsession: Session,
    gameweeks: Sequence[int],
    horizon: int = 1,
) -> MinutesScore:
    """
    Score a minutes model over `gameweeks`, walking the season forward.

    The minutes twin of `backtest_team_model`. A minutes model is not fitted, so
    `build` is called once per gameweek only to keep the three backtests alike.
    """
    score = MinutesScore()
    for gameweek in gameweeks:
        evaluation_gameweeks = list(range(gameweek, gameweek + horizon))
        player_scores = get_player_scores_for_gameweeks(
            evaluation_gameweeks, season=season, dbsession=dbsession
        )
        if not player_scores:
            logger.info("No performances for %s GW%s, skipping", season, gameweek)
            continue
        score += score_minutes_model(
            build(),
            player_scores,
            horizon,
            season=season,
            dbsession=dbsession,
        )
        logger.info(
            "GW%s: mean absolute error %.2f minutes, band accuracy %.3f over %s "
            "performances (%.1f%% given no chance at all)",
            gameweek,
            score.mean_absolute_error,
            score.band_accuracy,
            score.n_observations,
            100 * score.impossible_fraction,
        )
    return score


@dataclass(frozen=True)
class ErrorScore:
    """Mean absolute error over some number of observations. Lower is better."""

    total_absolute_error: float = 0.0
    n_observations: int = 0

    @property
    def mean_absolute_error(self) -> float:
        if not self.n_observations:
            return 0.0
        return self.total_absolute_error / self.n_observations

    def __add__(self, other: "ErrorScore") -> "ErrorScore":
        return ErrorScore(
            total_absolute_error=self.total_absolute_error + other.total_absolute_error,
            n_observations=self.n_observations + other.n_observations,
        )


@dataclass(frozen=True)
class BreakdownScore:
    """
    What could be scored about a set of predictions, part by part.

    `points` is always there; the rest is whatever the model reported. A model
    that only predicts a total is scored on the total alone rather than being
    penalised for the parts it does not claim to have.
    """

    points: PointsScore = field(default_factory=PointsScore)
    minutes: ErrorScore | None = None
    involvement: InvolvementScore | None = None
    components: dict[str, ErrorScore] | None = None

    def __add__(self, other: "BreakdownScore") -> "BreakdownScore":
        components: dict[str, ErrorScore] | None = None
        if self.components is not None or other.components is not None:
            components = dict(self.components or {})
            for name, score in (other.components or {}).items():
                components[name] = components.get(name, ErrorScore()) + score
        return BreakdownScore(
            points=self.points + other.points,
            minutes=_add_optional(self.minutes, other.minutes),
            involvement=_add_optional(self.involvement, other.involvement),
            components=components,
        )


def _add_optional[ScoreT: (ErrorScore, InvolvementScore)](
    left: ScoreT | None, right: ScoreT | None
) -> ScoreT | None:
    """Add two scores where either may be absent, and absent plus absent is absent."""
    if left is None:
        return right
    if right is None:
        return left
    return left + right


def score_prediction_breakdown(
    model: PointsModel,
    player_scores: Iterable[PlayerScore],
    *,
    root_gameweek: int,
    season: str,
    dbsession: Session,
) -> BreakdownScore:
    """
    Score a fitted points model on every part of its prediction it will report.

    The whole answer to "was it the minutes or the goals I got wrong?", for a
    model that decomposes - and still an answer for one that does not, because
    every optional part of a `PointsPrediction` is scored only if it is there.

    The minutes part is the model's expected minutes against what was played;
    `score_minutes_model` scores a minutes model on its own, in bands and with a
    log probability, which needs the whole distribution rather than its mean.

    Args:
        model: Already fitted, for a window with `root_gameweek` as its first
            gameweek.
        player_scores: The performances to score against, all within that
            window.
    """
    total = BreakdownScore()
    predicted_points: list[float] = []
    actual_points: list[float] = []
    for score in player_scores:
        if score.fixture.gameweek is None or not Position.is_modelled(
            score.player.position(season)
        ):
            # A fixture with no gameweek, or a manager - who has performances
            # and points in the database like anyone else, and no model here to
            # be scored against.
            total += BreakdownScore(points=PointsScore(n_skipped=1))
            continue
        prediction = model.predict(
            PointsRequest(
                player=score.player,
                fixture=score.fixture,
                root_gameweek=root_gameweek,
                season=season,
                dbsession=dbsession,
            )
        )
        predicted_points.append(prediction.expected_points)
        actual_points.append(float(score.points))
        total += _score_one(prediction, score, season=season)

    # One ranking over everything passed in, as `score_points_predictions` does:
    # a ranking is a property of a set of players, not of one performance, so it
    # cannot be accumulated a performance at a time like the errors above.
    correlation = _rank_correlation(
        np.asarray(predicted_points, dtype=float),
        np.asarray(actual_points, dtype=float),
    )
    if correlation is None:
        return total
    return total + BreakdownScore(
        points=PointsScore(total_rank_correlation=correlation, n_ranked=1)
    )


def _score_one(
    prediction: PointsPrediction, score: PlayerScore, *, season: str
) -> BreakdownScore:
    """One prediction against one performance, part by part."""
    error = abs(prediction.expected_points - float(score.points))
    played = int(score.minutes) > 0
    points = PointsScore(
        total_absolute_error=error,
        total_squared_error=error**2,
        n_observations=1,
        total_absolute_error_played=error if played else 0.0,
        n_played=int(played),
    )

    minutes = None
    if prediction.expected_minutes is not None:
        minutes = ErrorScore(
            total_absolute_error=abs(
                prediction.expected_minutes - float(score.minutes)
            ),
            n_observations=1,
        )

    involvement = None
    if prediction.involvement is not None:
        involvement = involvement_error(
            prediction.involvement.prob_score,
            prediction.involvement.prob_assist,
            score,
        )

    components = None
    if prediction.components is not None:
        position = score.player.position(season)
        try:
            actual = (
                actual_component_points(score, position) if position is not None else {}
            )
        except ValueError:
            # a position with no scoring rules, so nothing to compare against
            actual = {}
        components = {
            name: ErrorScore(
                total_absolute_error=abs(expected - actual[name]), n_observations=1
            )
            for name, expected in prediction.components.items()
            if name in actual
        }
    return BreakdownScore(
        points=points,
        minutes=minutes,
        involvement=involvement,
        components=components,
    )


def backtest_breakdown(
    build: Callable[[], PointsModel] | None = None,
    *,
    season: str,
    dbsession: Session,
    gameweeks: Sequence[int],
    horizon: int = 1,
) -> BreakdownScore:
    """
    Score every part of a points model on gameweeks it was not fitted to.

    The same walk forward as `backtest_points`, reporting the parts as well as
    the total - and without writing anything, because it scores the predictions
    as they are made rather than reading them back from the database.

    Args:
        build: Called once per gameweek, because a model is fitted in place. The
            package default points model when None.
    """
    # deferred slow import
    from airsenal.prediction.points_models import build_points_model  # noqa: PLC0415

    total = BreakdownScore()
    for gameweek in gameweeks:
        evaluation_gameweeks = list(range(gameweek, gameweek + horizon))
        player_scores = get_player_scores_for_gameweeks(
            evaluation_gameweeks, season=season, dbsession=dbsession
        )
        if not player_scores:
            logger.info("No performances for %s GW%s, skipping", season, gameweek)
            continue
        model = build() if build is not None else build_points_model()
        model = model.fit(
            PointsFitRequest(
                gameweeks=evaluation_gameweeks, season=season, dbsession=dbsession
            )
        )
        total += score_prediction_breakdown(
            model,
            player_scores,
            root_gameweek=gameweek,
            season=season,
            dbsession=dbsession,
        )
        logger.info(
            "GW%s: mean absolute error %.4f points over %s performances",
            gameweek,
            total.points.mean_absolute_error,
            total.points.n_observations,
        )
    return total
