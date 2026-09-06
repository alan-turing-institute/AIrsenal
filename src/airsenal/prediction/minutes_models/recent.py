"""Minutes from a player's recent appearances."""

from dataclasses import dataclass

from airsenal.prediction.minutes import get_recent_minutes_for_player
from airsenal.prediction.protocols import (
    MinutesDistribution,
    MinutesRequest,
)


@dataclass(frozen=True)
class RecentMinutesConfig:
    """
    How far back to read appearances.

    A run looks back as far as it looks forward, but never fewer than
    `min_fixtures_behind` fixtures - one or two appearances are too few to say
    anything about how much a player is being used.
    """

    min_fixtures_behind: int = 3


class RecentMinutesModel:
    """
    The player's last few appearances, each taken as equally likely.

    Which is a sample rather than a model: it carries no view about whether the
    player will start, only that a repeat of any of their recent outings is as
    plausible as any other. Its weakness is rotation - at the end of a season
    when a team has nothing to play for, recent minutes overstate the next
    match.
    """

    def __init__(self, config: RecentMinutesConfig | None = None):
        self.config = config or RecentMinutesConfig()

    def predict(self, request: MinutesRequest) -> MinutesDistribution:
        fixtures_behind = max(request.n_gameweeks, self.config.min_fixtures_behind)
        return MinutesDistribution.uniform(
            get_recent_minutes_for_player(
                request.player,
                n_matches_to_use=fixtures_behind,
                season=request.season,
                # everything is read as at the gameweek being predicted from,
                # so the last gameweek this may see is the one before it
                last_gameweek=request.gameweek - 1,
                dbsession=request.dbsession,
            )
        )
