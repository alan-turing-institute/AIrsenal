"""What a prediction model has to provide."""

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, NotRequired, Protocol, TypedDict

import numpy as np
from sqlalchemy.orm import Session

from airsenal.db.models import Fixture, Player


class PlayerFitData(TypedDict):
    """
    Everything a player model is fitted to, from `features.process_player_data`.

    One position's players, over the matches in the fitting window. Every player
    has a row for every match, zero-padded where they did not appear, so `y` and
    `minutes` are rectangular and `nplayer`/`nmatch` are their dimensions.
    """

    # (n_players,) the players these rows are about, sorted
    player_ids: np.ndarray
    nplayer: int
    nmatch: int
    # (n_players, n_matches) minutes played
    minutes: np.ndarray
    # (n_players, n_matches, 3) goal involvements, the last axis being
    # (goals, assists, neither). The three sum to the goals the player's team
    # scored in that match.
    y: np.ndarray
    # (3,) Dirichlet prior concentrations over that same last axis. Strictly
    # positive, or a model that builds a real Dirichlet from it cannot be fitted.
    alpha: np.ndarray
    # (n_players, n_matches) years between the match and the gameweek being
    # predicted, for a model that weights recent matches more heavily
    time_diff: np.ndarray


class TeamFitData(TypedDict):
    """
    Everything a team model is fitted to, from `team_models.get_training_data`.

    One entry per past match, so every array here has the same length.
    """

    home_team: np.ndarray
    away_team: np.ndarray
    home_goals: np.ndarray
    away_goals: np.ndarray
    # years between the match and the gameweek being predicted
    time_diff: np.ndarray
    neutral_venue: np.ndarray
    game_weights: np.ndarray
    # FIFA ratings per team name, absent when fitting without them. A promoted
    # team has no results, so its ratings are what `add_new_team` stands in with.
    team_covariates: NotRequired[dict[str, np.ndarray]]


@dataclass(frozen=True, eq=False)
class PlayerInvolvement:
    """
    How each player shares in one of their team's goals.

    Per goal and for a full match, so the points calculation scales by the
    fraction of the match played. The three shares sum to one per player, which
    is checked here rather than described in prose: it used to be a
    `dict[str, np.ndarray]` whose keys and invariants only a docstring knew.

    Not comparable with `==` - the fields are arrays - hence `eq=False`.
    """

    # (n_players,) the players these shares are about
    player_ids: np.ndarray
    prob_score: np.ndarray
    prob_assist: np.ndarray
    prob_neither: np.ndarray

    def __post_init__(self) -> None:
        lengths = {
            len(self.player_ids),
            len(self.prob_score),
            len(self.prob_assist),
            len(self.prob_neither),
        }
        if len(lengths) != 1:
            msg = f"Mismatched lengths in a player involvement: {lengths}"
            raise ValueError(msg)
        total = self.prob_score + self.prob_assist + self.prob_neither
        if len(self.player_ids) and not np.allclose(total, 1.0, atol=1e-6):
            worst = int(np.argmax(np.abs(total - 1.0)))
            msg = (
                f"Involvement shares for player {self.player_ids[worst]} sum to "
                f"{total[worst]}, not one"
            )
            raise ValueError(msg)

    def as_dict(self) -> dict[str, np.ndarray]:
        """The columns, for building the frame `fit_player_data` returns."""
        return {
            "player_id": self.player_ids,
            "prob_score": self.prob_score,
            "prob_assist": self.prob_assist,
            "prob_neither": self.prob_neither,
        }


class PlayerModel(Protocol):
    """Predicts how a team's goals are shared out between its players."""

    def fit(self, data: PlayerFitData) -> "PlayerModel":
        """
        Fit to the data, using the hyperparameters given at construction.

        Deliberately takes no `**kwargs`, so a hyperparameter a model does not
        implement is an error rather than something it silently swallows.
        """
        ...

    def predict_involvement(self) -> PlayerInvolvement:
        """
        Each fitted player's share of scoring, assisting, or neither, for a goal.

        A share, not necessarily a probability: a model that arrives at one
        without a posterior satisfies this too.
        """
        ...


class TeamModel(Protocol):
    """
    A model of how teams perform, that can be fitted to past results.

    What it predicts is not here, because there is more than one useful answer:
    `ScorelineTeamModel` gives a distribution over goal counts, and
    `ExpectedGoalsTeamModel` only a mean. This is what they have in common, and
    what `get_fitted_team_model` needs - which is why fitting a model does not
    require knowing which kind it is.
    """

    @property
    def teams(self) -> list[str] | None:
        """The teams this model knows about, or None before it is fitted."""
        ...

    def fit(self, training_data: TeamFitData) -> "TeamModel":
        """
        Fit to the data, using the settings given at construction.

        Like `PlayerModel.fit`, this takes no `**kwargs`. bpl wants its
        time-weighting arguments at fit time, so `DixonColesTeamModel` holds
        them and passes them on itself.
        """
        ...

    def add_new_team(self, team_name: str, **kwargs: Any) -> None:
        """
        Teach the model a team that has no results to fit to, such as a promoted one.

        Called after `fit`, once per unknown team. `team_covariates` is passed as
        a keyword when the model is being fitted with FIFA ratings; a model that
        does not use covariates ignores it.
        """
        ...


class ScorelineTeamModel(TeamModel, Protocol):
    """
    A team model that gives a whole distribution over goal counts.

    What the points calculation needs: expected attacking points come from a
    multinomial over however many goals the team scores, and a clean sheet is
    the probability of the opponent scoring none, so a mean is not enough. Every
    model in `TEAM_MODELS` is one of these.
    """

    def predict_score_n_proba(
        self, n: np.ndarray, team: str, opponent: str, home: bool = True, **kwargs: Any
    ) -> np.ndarray:
        """
        The probability of `team` scoring each goal count in `n` against `opponent`.

        Returns an array the same length as `n`. `home` says which side of the
        fixture `team` is on.
        """
        ...

    def predict_outcome_proba(
        self, home_team: Sequence[str], away_team: Sequence[str]
    ) -> dict[str, np.ndarray]:
        """
        Win, draw and loss probabilities for each fixture.

        Keyed "home_win", "draw" and "away_win". Here rather than fetched with
        `getattr` at the one call site: a model that cannot answer should fail to
        type-check, not fail at run time. `outcome_proba_from_scores` implements
        it for a model whose two goal counts are independent.
        """
        ...


class ExpectedGoalsTeamModel(TeamModel, Protocol):
    """
    A team model that predicts only how many goals a team will score on average.

    Which is all some models have to say - an xG model fitted to a continuous
    quantity has no natural distribution over goal *counts*. `PoissonScorelines`
    turns one of these into a `ScorelineTeamModel` so it can be predicted with,
    rather than every such model having to invent a distribution of its own.
    """

    def predict_expected_goals(
        self, team: str, opponent: str, home: bool = True, **kwargs: Any
    ) -> float:
        """
        How many goals `team` is expected to score against `opponent`.

        `home` says which side of the fixture `team` is on.
        """
        ...


@dataclass(frozen=True)
class MinutesDistribution:
    """
    The minutes a player might play in one fixture, and how likely each is.

    `weights` line up with `minutes` and sum to one. A model with nothing but a
    sample of recent appearances weights them equally; one that can say a start
    is likelier than a substitute appearance says so here instead of leaving the
    points calculation to average over the sample as though it were a
    distribution.
    """

    minutes: tuple[float, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.minutes:
            msg = "A minutes distribution needs at least one possible value"
            raise ValueError(msg)
        if len(self.minutes) != len(self.weights):
            msg = (
                f"{len(self.minutes)} minutes values but {len(self.weights)} "
                "weights; they have to line up"
            )
            raise ValueError(msg)
        if any(value < 0 for value in self.minutes):
            msg = f"Negative minutes in {self.minutes}"
            raise ValueError(msg)
        if any(weight < 0 for weight in self.weights):
            msg = f"Negative weight in {self.weights}"
            raise ValueError(msg)
        if not math.isclose(sum(self.weights), 1.0, abs_tol=1e-9):
            msg = f"Weights {self.weights} sum to {sum(self.weights)}, not one"
            raise ValueError(msg)

    @classmethod
    def uniform(cls, minutes: Iterable[float]) -> "MinutesDistribution":
        """Every value equally likely - a sample of appearances, not a model of one."""
        values = tuple(float(value) for value in minutes)
        if not values:
            msg = "A minutes distribution needs at least one possible value"
            raise ValueError(msg)
        return cls(minutes=values, weights=(1.0 / len(values),) * len(values))

    @property
    def expected_minutes(self) -> float:
        """The mean, which is what a minutes prediction is scored against."""
        return self.expectation(lambda minutes: minutes)

    def expectation(self, quantity: Callable[[float], float]) -> float:
        """
        The weighted average of `quantity` over the possible minutes.

        The one operation the points calculation performs on a distribution, so
        the weighting lives here rather than in the caller.
        """
        return sum(
            weight * quantity(minutes)
            for minutes, weight in zip(self.minutes, self.weights, strict=True)
        )

    def probability_between(self, low: float, high: float) -> float:
        """How much weight falls in `[low, high)`, for scoring against a band."""
        return sum(
            weight
            for minutes, weight in zip(self.minutes, self.weights, strict=True)
            if low <= minutes < high
        )


@dataclass(frozen=True, kw_only=True)
class MinutesRequest:
    """
    Everything a minutes model needs to predict one player's minutes.

    Read as at `gameweek` - the gameweek being predicted *from* - so a model
    must not look at a match played in it or later.
    """

    player: Player
    gameweek: int
    season: str
    # How many gameweeks the run covers. A model that reads recent appearances
    # has to decide how far back to look, and looking back as far as the run
    # looks forward is one reasonable answer.
    n_gameweeks: int
    dbsession: Session


class MinutesModel(Protocol):
    """Predicts how long a player will be on the pitch."""

    def predict(self, request: MinutesRequest) -> MinutesDistribution:
        """
        The minutes this player might play, and how likely each is.

        Called once per player per prediction run, not once per fixture: what
        varies by fixture is handled by the points calculation.
        """
        ...


@dataclass(frozen=True, kw_only=True)
class ComponentRequest:
    """
    Everything a component of a score needs to say how many points it expects.

    One player, one fixture, one number of minutes. `minutes` is a value the
    minutes model thinks possible rather than a prediction: the points
    calculation asks each component once per possible value and takes the
    expectation over them, so a component is a function of minutes and does not
    need to know how likely they were.
    """

    player_id: int
    position: str
    minutes: float
    # From the team model: how likely the player's own team is to score each
    # number of goals, and how likely their opponent is to.
    team_score_probability: dict[int, float]
    team_concede_probability: dict[int, float]
    # From the player model: this player's share of one of their team's goals.
    prob_score: float
    prob_assist: float


class PointComponent(Protocol):
    """
    One part of an FPL score, fitted to past seasons and then asked what it expects.

    A component is only ever asked about what it can answer, so it takes the
    whole `ComponentRequest` and reads the parts it needs: bonus points depend
    on the player and the minutes, a clean sheet on the opponent's goals, and
    appearance points on nothing but the minutes.
    """

    @property
    def name(self) -> str:
        """What this component is called, in a config and in a score breakdown."""
        ...

    def fit(self, gameweek: int, season: str, dbsession: Session) -> "PointComponent":
        """
        Fit to everything before `gameweek`, and return self.

        A component with nothing to fit - appearance points are a rule, not an
        average - returns itself unchanged.
        """
        ...

    def expected_points(self, request: ComponentRequest) -> float:
        """How many points this component expects, for one player and fixture."""
        ...


@dataclass(frozen=True, kw_only=True)
class PointsFitRequest:
    """
    The window a points model is fitted for.

    Everything is fitted as at `min(gameweeks)` - the gameweek being predicted
    *from* - so that nothing sees a match played later than that, whichever
    gameweek of the window a fixture is in.
    """

    gameweeks: list[int]
    season: str
    dbsession: Session


@dataclass(frozen=True, kw_only=True)
class PointsRequest:
    """One player in one fixture, for a fitted points model to predict."""

    player: Player
    fixture: Fixture
    # The gameweek being predicted from, which is `min(gameweeks)` of the window
    # the model was fitted for and not the fixture's own gameweek.
    root_gameweek: int
    season: str
    dbsession: Session


@dataclass(frozen=True)
class PointsPrediction:
    """
    What a points model expects a player to score in one fixture.

    A number, and nothing a model has to justify: how it arrived at one is its
    own business, which is what lets a model with no notion of minutes,
    involvement or components satisfy this.
    """

    expected_points: float


class PointsModel(Protocol):
    """
    Predicts how many points a player will score in a fixture.

    The whole of prediction behind one seam. `ComponentPointsModel` is the
    shipped one - a team model, a player model, a minutes model and a list of
    components - but nothing here requires that shape: a model that regresses
    points directly from whatever features it likes is a table entry beside it.
    """

    def fit(self, request: PointsFitRequest) -> "PointsModel":
        """Fit to everything before the window, and return self."""
        ...

    def predict(self, request: PointsRequest) -> PointsPrediction:
        """What this player is expected to score in this fixture."""
        ...
