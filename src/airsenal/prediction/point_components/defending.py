"""Points for a clean sheet, and deductions for goals conceded."""

from sqlalchemy.orm import Session

from airsenal.game.enums import Position
from airsenal.game.scoring import MIN_MINUTES_FULL, points_for_cs
from airsenal.prediction.protocols import ComponentRequest


def get_defending_points(
    position: str, minutes: int | float, team_concede_prob: dict[int, float]
) -> float:
    """Expected defending points: clean sheets and goals conceded."""
    if position == Position.FWD or minutes == 0.0:
        return 0.0

    defending_points = 0.0
    if minutes >= MIN_MINUTES_FULL:
        defending_points = points_for_cs[position] * team_concede_prob[0]

    if position in (Position.DEF, Position.GK):
        defending_points -= sum(
            (ngoals // 2) * (minutes / 90) * concede_n_prob
            for ngoals, concede_n_prob in team_concede_prob.items()
        )
    return defending_points


class DefendingComponent:
    """A clean sheet needs the hour; conceding costs defenders and keepers."""

    name = "defending"

    def fit(
        self, gameweek: int, season: str, dbsession: Session
    ) -> "DefendingComponent":
        """Nothing to fit: the opponent's goals come from the team model."""
        del gameweek, season, dbsession
        return self

    def expected_points(self, request: ComponentRequest) -> float:
        return get_defending_points(
            request.position, request.minutes, request.team_concede_probability
        )
