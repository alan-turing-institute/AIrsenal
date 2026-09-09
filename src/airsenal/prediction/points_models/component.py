"""Predicted points as a sum of components, from a team, player and minutes model."""

from functools import partial

import numpy as np
import pandas as pd

from airsenal.core.logging import get_logger
from airsenal.db.queries.fixtures import get_fixtures_for_gameweeks
from airsenal.game.scoring import MAX_GOALS
from airsenal.prediction.minutes_models import build_minutes_model
from airsenal.prediction.player_models import build_player_model
from airsenal.prediction.player_models.fitting import get_all_fitted_player_data
from airsenal.prediction.point_components import PointsConfig
from airsenal.prediction.protocols import (
    ComponentRequest,
    InvolvementShare,
    MinutesDistribution,
    MinutesModel,
    MinutesRequest,
    PlayerModel,
    PointComponent,
    PointsFitRequest,
    PointsPrediction,
    PointsRequest,
    ScorelineTeamModel,
)
from airsenal.prediction.team_models import build_team_model
from airsenal.prediction.team_models.fitting import (
    get_fitted_team_model,
    get_goal_probabilities_for_fixtures,
)

logger = get_logger(__name__)


def describe_component(component: object) -> str:
    """
    What to call one collaborator in a run's record of the parts that produced it.

    A collaborator that wraps another says so through its own
    `describe_component` - `PoissonScorelines` around an expected-goals model -
    so the record names the model a prediction came from and not only the
    wrapper around it. Read by name, the way `describe_pipeline` reads
    `describe`: a component without one is named by its class and left at that.
    """
    describe = getattr(component, "describe_component", None)
    return describe() if callable(describe) else type(component).__name__


class ComponentPointsModel:
    """
    The way AIrsenal has always predicted points, behind the points-model seam.

    Four collaborators, each a pluggable kind of its own: a team model for how
    many goals a fixture produces, a player model for the share of them this
    player takes, a minutes model for how long they are on the pitch, and the
    components that turn all of that into points.

    The team model is not one of the components. A component answers in points
    and this answers in goals, and two components read it rather than one - so
    it is a collaborator that fills in the `ComponentRequest`, not an entry in
    the list.
    """

    def __init__(
        self,
        team_model: ScorelineTeamModel | None = None,
        player_model: PlayerModel | None = None,
        minutes_model: MinutesModel | None = None,
        components: list[PointComponent] | None = None,
        points: PointsConfig | None = None,
    ):
        if components is not None and points is not None:
            msg = (
                "Pass components or a PointsConfig to choose them, not both: "
                "a config that is ignored is worse than one that is refused"
            )
            raise ValueError(msg)
        self.team_model = team_model if team_model is not None else build_team_model()
        self.player_model = (
            player_model if player_model is not None else build_player_model()
        )
        self.minutes_model = (
            minutes_model if minutes_model is not None else build_minutes_model()
        )
        self.points = points if points is not None else PointsConfig()
        self.components = components if components is not None else None
        self.fitted: _FittedComponents | None = None

    def describe(self) -> dict[str, str]:
        """The parts this was built from, for a replay's record of its run."""
        return {
            "points_model": type(self).__name__,
            "team_model": describe_component(self.team_model),
            "player_model": describe_component(self.player_model),
            "minutes_model": describe_component(self.minutes_model),
        }

    def fit(self, request: PointsFitRequest) -> "ComponentPointsModel":
        """
        Fit every part, as at the first gameweek of the window.

        Nothing here may see a match played in that gameweek or later, which is
        also what decides which players there are to predict for.
        """
        root_gameweek = min(request.gameweeks)
        team_model = get_fitted_team_model(
            season=request.season,
            gameweek=root_gameweek,
            dbsession=request.dbsession,
            model=self.team_model,
        )
        logger.info("Calculating fixture score probabilities...")
        fixtures = get_fixtures_for_gameweeks(
            request.gameweeks, season=request.season, dbsession=request.dbsession
        )
        components = [
            component.fit(root_gameweek, request.season, request.dbsession)
            for component in (
                self.components
                if self.components is not None
                else self.points.components()
            )
        ]
        logger.info("Predicting %s", ", ".join(c.name for c in components))
        self.fitted = _FittedComponents(
            goal_probabilities=get_goal_probabilities_for_fixtures(
                fixtures, team_model, max_goals=MAX_GOALS
            ),
            involvement=get_all_fitted_player_data(
                root_gameweek,
                request.season,
                model=self.player_model,
                dbsession=request.dbsession,
            ),
            components=components,
            n_gameweeks=len(request.gameweeks),
        )
        return self

    def predict(self, request: PointsRequest) -> PointsPrediction:
        """Sum every component over the minutes the player might play."""
        if self.fitted is None:
            msg = "The points model has not been fitted yet."
            raise RuntimeError(msg)
        return self.fitted.predict(request, self.minutes_model)


class _FittedComponents:
    """What `ComponentPointsModel.fit` produced, and how a prediction uses it."""

    def __init__(
        self,
        goal_probabilities: dict[int, dict[str, dict[int, float]]],
        involvement: dict[str, pd.DataFrame],
        components: list[PointComponent],
        n_gameweeks: int,
    ):
        self.goal_probabilities = goal_probabilities
        self.involvement = involvement
        self.components = components
        self.n_gameweeks = n_gameweeks
        # Minutes are read as at the root gameweek but predicted for a
        # particular one - a player can be back from injury later in the window
        # - so they are remembered per player per gameweek rather than per
        # player per fixture.
        self._minutes: dict[tuple[int, int], MinutesDistribution] = {}

    def minutes_for(
        self, request: PointsRequest, minutes_model: MinutesModel, gameweek: int
    ) -> MinutesDistribution:
        key = (request.player.player_id, gameweek)
        if key not in self._minutes:
            self._minutes[key] = minutes_model.predict(
                MinutesRequest(
                    player=request.player,
                    root_gameweek=request.root_gameweek,
                    fixture_gameweek=gameweek,
                    season=request.season,
                    n_gameweeks=self.n_gameweeks,
                    dbsession=request.dbsession,
                )
            )
        return self._minutes[key]

    def predict(
        self, request: PointsRequest, minutes_model: MinutesModel
    ) -> PointsPrediction:
        player = request.player
        gameweek = request.fixture.gameweek
        if gameweek is None:
            msg = f"Fixture {request.fixture} has no gameweek to predict for"
            raise ValueError(msg)

        team = player.team(request.root_gameweek, request.season)
        position = player.position(request.season)
        if position is None or team is None:
            msg = (
                f"Player {player} has missing team or position for season "
                f"{request.season}"
            )
            raise ValueError(msg)

        involvement = self.involvement[position].loc[player.player_id]
        if not isinstance(involvement, pd.Series):
            msg = f"involvement for {player} is not a Series, but {type(involvement)}"
            raise RuntimeError(msg)

        minutes = self.minutes_for(request, minutes_model, gameweek)
        if minutes.expected_minutes == 0.0:
            # Not a refusal to answer: a player who will not be on the pitch is
            # predicted zero from every component, which is what the components
            # themselves would say. Whether an injured player will be on the
            # pitch is the minutes model's business, not this one's.
            return PointsPrediction(
                expected_points=0.0,
                expected_minutes=0.0,
                involvement=InvolvementShare(
                    prob_score=float(involvement["prob_score"]),
                    prob_assist=float(involvement["prob_assist"]),
                ),
                components={component.name: 0.0 for component in self.components},
            )

        is_home = request.fixture.home_team == team
        opponent = request.fixture.away_team if is_home else request.fixture.home_team
        fixture_probabilities = self.goal_probabilities[request.fixture.fixture_id]

        shares = InvolvementShare(
            prob_score=float(involvement["prob_score"]),
            prob_assist=float(involvement["prob_assist"]),
        )

        def points_from(component: PointComponent, mins: float) -> float:
            """One component's points, for one number of minutes played."""
            return component.expected_points(
                ComponentRequest(
                    player_id=player.player_id,
                    position=position,
                    minutes=mins,
                    team_score_probability=fixture_probabilities[team],
                    team_concede_probability=fixture_probabilities[opponent],
                    prob_score=shares.prob_score,
                    prob_assist=shares.prob_assist,
                )
            )

        # Each component's own expectation over the possible minutes. They sum
        # to the total, because an expectation of a sum is a sum of
        # expectations - so the breakdown always adds up to what was predicted.
        components = {
            component.name: minutes.expectation(partial(points_from, component))
            for component in self.components
        }
        points = sum(components.values())
        if np.isnan(points):
            msg = f"nan points for {player} {request.fixture}"
            raise ValueError(msg)
        return PointsPrediction(
            expected_points=points,
            expected_minutes=minutes.expected_minutes,
            involvement=shares,
            components=components,
        )
