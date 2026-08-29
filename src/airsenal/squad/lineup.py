"""
Choosing a starting eleven, a bench order and a captain.

Free functions over a squad's players rather than methods, because none of them
needs a Squad: a formation is a fact about eleven players, not about the object
that happens to hold fifteen. `Squad` calls into here and keeps the two entry
points anything outside this package uses.
"""

from collections.abc import Callable
from operator import itemgetter

import numpy as np

from airsenal.core.logging import get_logger
from airsenal.game.enums import Position
from airsenal.squad.player import SquadPlayer

logger = get_logger(__name__)

# the outfield positions a formation names, in the order it names them
FORMATION_POSITIONS = (Position.DEF, Position.MID, Position.FWD)

FORMATIONS = [
    (3, 4, 3),
    (3, 5, 2),
    (4, 3, 3),
    (4, 4, 2),
    (4, 5, 1),
    (5, 4, 1),
    (5, 3, 2),
    (5, 2, 3),
]

FORMATION_SLOTS = {
    0: (),
    1: (2,),
    2: (1, 3),
    3: (1, 2, 3),
    4: (0, 1, 3, 4),
    5: (0, 1, 2, 3, 4),
}

# players of each position, best predicted points first
type PlayersByPosition = dict[str, list[tuple[SquadPlayer, float]]]


def choose_starting_eleven(
    players: list[SquadPlayer],
    tag: str,
    gameweek: int,
    score_starting_eleven: Callable[[], float],
) -> float:
    """
    Pick the best legal starting eleven, and order the bench behind it.

    `score_starting_eleven` scores whatever `is_starting` currently says, which
    is how each candidate formation is compared. It is a callback rather than a
    sum written out here because scoring a squad is the squad's own job, and it
    is the same sum a caller asking for expected points gets.
    """
    by_position: PlayersByPosition = {position: [] for position in Position}
    for p in players:
        try:
            points_prediction = p.predicted_points[tag][gameweek]
        except KeyError:
            # player does not have a game in this gameweek
            points_prediction = 0.0
        by_position[p.position].append((p, points_prediction))
    for v in by_position.values():
        v.sort(key=itemgetter(1), reverse=True)

    # always start the first-placed and sub the second-placed keeper
    by_position[Position.GK][0][0].is_starting = True
    by_position[Position.GK][1][0].is_starting = False
    best_score = 0.0
    best_formation = None
    for f in FORMATIONS:
        apply_formation(by_position, f)
        score = score_starting_eleven()
        if score >= best_score:
            best_score = score
            best_formation = f
    logger.debug("Best formation is %s", best_formation)
    if best_formation is None:
        msg = "No valid formation found for squad"
        raise RuntimeError(msg)
    apply_formation(by_position, best_formation)
    order_substitutes(players, tag, gameweek)

    return best_score


def order_substitutes(players: list[SquadPlayer], tag: str, gameweek: int) -> None:
    """Number the bench by predicted points, best first."""
    subs = [p for p in players if not p.is_starting]

    points = []
    for player in subs:
        try:
            points.append(player.predicted_points[tag][gameweek])
        except ValueError:
            points.append(0)

    # sort the players by points (descending)
    ordered_sub_inds = reversed(np.argsort(points))
    for sub_position, sub_ind in enumerate(ordered_sub_inds):
        subs[sub_ind].sub_position = sub_position


def apply_formation(
    by_position: PlayersByPosition, formation: tuple[int, int, int]
) -> None:
    """Set each player's `is_starting` to match a formation given as e.g. (4, 4, 2)."""
    for i, pos in enumerate(FORMATION_POSITIONS):
        for index, player in enumerate(by_position[pos]):
            player[0].is_starting = index < formation[i]


def formation_of(players: list[SquadPlayer]) -> dict[str, int]:
    """A starting eleven's formation, as {"DEF": n, "MID": n, "FWD": n}."""
    formation: dict[str, int] = dict.fromkeys(Position, 0)
    for player in players:
        if player.is_starting:
            formation[player.position] += 1
    return formation


def is_substitution_allowed(
    players: list[SquadPlayer], player_out: SquadPlayer, player_in: SquadPlayer
) -> bool:
    """Whether swapping one player for another leaves a legal formation."""
    formation = formation_of(players)
    formation[player_out.position] -= 1
    formation[player_in.position] += 1
    return tuple(formation[pos] for pos in FORMATION_POSITIONS) in FORMATIONS


def pick_captains(players: list[SquadPlayer], tag: str, gameweek: int) -> None:
    """
    Make the two highest-scoring players captain and vice-captain.

    Clears any existing captaincy first, and mutates the players in place.
    """
    player_list = []
    for p in players:
        p.is_captain = False
        p.is_vice_captain = False
        player_list.append((p, p.predicted_points[tag][gameweek]))

    player_list.sort(key=itemgetter(1), reverse=True)
    player_list[0][0].is_captain = True
    player_list[1][0].is_vice_captain = True
