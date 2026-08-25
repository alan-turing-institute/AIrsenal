"""Writing an optimised plan, or a from-scratch squad, into the database."""

from dataclasses import dataclass

from sqlalchemy import select

from airsenal.db.models import TransferSuggestion
from airsenal.optimization.persist import fill_initial_suggestion_table
from tests.conftest import session_scope

# conftest pins next_gameweek() to 1, so a squad built for any other gameweek is
# what tells the two apart.
FUTURE_GAMEWEEK = 7

# unique to this test, so the rows it writes cannot be confused with another's
FPL_TEAM_ID = 987654


@dataclass
class FakePlayer:
    player_id: int


class FakeSquad:
    """Only what `fill_initial_suggestion_table` reads off a Squad."""

    def __init__(self, n_players: int = 15) -> None:
        self.players = [FakePlayer(player_id=i) for i in range(n_players)]

    # signature matches Squad.get_expected_points, which is called positionally
    def get_expected_points(self, gameweek: int, tag: str) -> float:  # noqa: ARG002
        return 50.0


def test_initial_suggestions_are_stamped_with_the_requested_gameweek():
    """
    The rows must carry the gameweek the squad was built for.

    It used to resolve `gameweek`, score with it, and then write
    `next_gameweek()` into the row anyway - so a replay or a back-test
    stamped every suggestion with the wrong gameweek.
    """
    with session_scope() as ts:
        fill_initial_suggestion_table(
            FakeSquad(),
            fpl_team_id=FPL_TEAM_ID,
            tag="test_initial_suggestion_gameweek",
            season="2324",
            gameweek=FUTURE_GAMEWEEK,
            dbsession=ts,
        )
        # read inside the session: fill_* commits, which expires the instances
        gameweeks = ts.scalars(
            select(TransferSuggestion.gameweek).where(
                TransferSuggestion.fpl_team_id == FPL_TEAM_ID
            )
        ).all()

    assert len(gameweeks) == 15
    assert set(gameweeks) == {FUTURE_GAMEWEEK}
