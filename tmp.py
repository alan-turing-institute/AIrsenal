import pandas as pd
from sqlalchemy import and_, or_, select
from sqlalchemy.orm import selectinload

from airsenal.framework.schema import Fixture, Player, PlayerScore, session
from airsenal.framework.season import CURRENT_SEASON
from airsenal.framework.utils import NEXT_GAMEWEEK

season = CURRENT_SEASON
gameweek = NEXT_GAMEWEEK
appearances_only = True

query = (
    select(PlayerScore)
    .join(PlayerScore.fixture)
    .options(
        selectinload(PlayerScore.fixture),
        selectinload(PlayerScore.result),
        selectinload(PlayerScore.player).selectinload(Player.attributes),
    )
    .where(
        or_(
            Fixture.season < season,
            and_(Fixture.season == season, Fixture.gameweek < gameweek),
        )
    )
)
if appearances_only:
    query = query.where(PlayerScore.minutes > 0)

print(query)

scores = session.scalars(query).all()

print(len(scores))

col_names = [
    "player_id",
    "player_name",
    "player_team",
    "position",
    "match_id",
    "date",
    "season",
    "gameweek",
    "goals",
    "assists",
    "minutes",
    "team_goals",
    "expected_goals",
    "expected_assists",
    "chance_of_playing",
    "news",
]
player_data = []

for score in scores:
    if score.fixture.home_team == score.player_team:
        team_goals = score.result.home_score
    else:
        team_goals = score.result.away_score

    player_data.append(
        [
            score.player.player_id,
            score.player.name,
            score.player_team,
            score.player.position(score.fixture.season),
            score.result_id,
            score.fixture.date,
            score.fixture.season,
            score.fixture.gameweek,
            score.goals,
            score.assists,
            score.minutes,
            team_goals,
            score.expected_goals,
            score.expected_assists,
            score.chance_of_playing,
            score.news,
        ]
    )

df = pd.DataFrame(player_data, columns=col_names)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
print(df.head())
