"""Writing an optimised plan, or a from-scratch squad, into the database."""

from dataclasses import dataclass

from sqlalchemy import select

from airsenal.db.models import TransferSuggestion
from airsenal.game.enums import Chip
from airsenal.optimization.moves import GameweekMove
from airsenal.optimization.persist import (
    fill_initial_suggestion_table,
    fill_suggestion_table,
)
from airsenal.optimization.plan import GameweekOutcome, Plan
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
    Rows carry the gameweek the squad was built for, not `next_gameweek()`.

    Otherwise a replay or a back-test stamps every suggestion with the wrong one.
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


def plan(gameweek: int, players_in, players_out, chip=None, points=100.0) -> Plan:
    return Plan(
        root_gameweek=gameweek,
        outcomes=(
            GameweekOutcome(
                gameweek=gameweek,
                move=GameweekMove(n_transfers=len(players_in), chip=chip),
                points=points,
                discount_factor=1.0,
                points_hit=0,
                free_transfers=1,
                players_in=tuple(players_in),
                players_out=tuple(players_out),
            ),
        ),
    )


def suggestions_for(dbsession, fpl_team_id):
    """(player_id, in_or_out, gameweek, chip_played) for one team's rows."""
    return sorted(
        dbsession.execute(
            select(
                TransferSuggestion.player_id,
                TransferSuggestion.in_or_out,
                TransferSuggestion.gameweek,
                TransferSuggestion.chip_played,
            ).where(TransferSuggestion.fpl_team_id == fpl_team_id)
        ).all()
    )


def test_a_plan_is_written_as_one_row_per_player_moved():
    fpl_team_id = 987655
    with session_scope() as ts:
        fill_suggestion_table(
            baseline_score=60.0,
            best_plan=plan(4, players_in=[11, 12], players_out=[21, 22]),
            season="2324",
            fpl_team_id=fpl_team_id,
            dbsession=ts,
        )
        rows = suggestions_for(ts, fpl_team_id)

    assert rows == [
        (11, 1, 4, None),
        (12, 1, 4, None),
        (21, -1, 4, None),
        (22, -1, 4, None),
    ]


def test_every_row_of_a_plan_carries_the_points_gain_over_the_baseline():
    fpl_team_id = 987656
    with session_scope() as ts:
        fill_suggestion_table(
            baseline_score=60.0,
            best_plan=plan(4, players_in=[11], players_out=[21], points=100.0),
            season="2324",
            fpl_team_id=fpl_team_id,
            dbsession=ts,
        )
        gains = ts.scalars(
            select(TransferSuggestion.points_gain).where(
                TransferSuggestion.fpl_team_id == fpl_team_id
            )
        ).all()

    assert set(gains) == {40.0}


def test_a_chip_played_is_recorded_on_the_rows_of_its_gameweek():
    fpl_team_id = 987657
    with session_scope() as ts:
        fill_suggestion_table(
            baseline_score=0.0,
            best_plan=plan(
                6, players_in=[11], players_out=[21], chip=Chip.TRIPLE_CAPTAIN
            ),
            season="2324",
            fpl_team_id=fpl_team_id,
            dbsession=ts,
        )
        rows = suggestions_for(ts, fpl_team_id)

    assert {row[3] for row in rows} == {str(Chip.TRIPLE_CAPTAIN)}
