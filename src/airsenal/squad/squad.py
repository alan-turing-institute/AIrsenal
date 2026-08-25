"""
The class for an FPL squad.
Contains a set of players.
Is able to check that it obeys all constraints.
"""

from collections import defaultdict
from dataclasses import dataclass
from operator import itemgetter

from sqlalchemy.orm import Session

from airsenal.core.enums import Position
from airsenal.core.logging import get_logger
from airsenal.core.scoring import SQUAD_SIZE
from airsenal.core.season import CURRENT_SEASON
from airsenal.db.models import Player
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.db.queries.players import get_player_from_api_id
from airsenal.db.queries.scores import get_playerscores_for_player_gameweek
from airsenal.db.session import get_session
from airsenal.remote.fpl_api import FPLDataFetcher, get_fetcher
from airsenal.squad.lineup import (
    choose_starting_eleven,
    is_substitution_allowed,
    order_substitutes,
    pick_captains,
)
from airsenal.squad.player import (
    CandidatePlayer,
    SquadPlayer,
    bench_position,
)
from airsenal.squad.pricing import sell_price
from airsenal.squad.state import get_bank

logger = get_logger(__name__)

# how many players do we need to add. Keyed by Position, annotated `str`
# because what indexes these is a position read off a database row - Position
# is a StrEnum, so the two are the same key.
TOTAL_PER_POSITION: dict[str, int] = {
    Position.GK: 2,
    Position.DEF: 5,
    Position.MID: 5,
    Position.FWD: 3,
}


@dataclass(frozen=True)
class SubWeights:
    """
    How much a substitute's predicted points count towards a squad's score.

    Outfield weights are ordered by bench position: first substitute, second,
    third. Here rather than in `optimization/`, which is where it is configured,
    because this is the layer that reads it - it used to be flattened into a
    `dict[str, Any]` by an `as_dict()` on the way down, purely to cross the
    boundary.
    """

    gk: float = 0.03
    outfield: tuple[float, float, float] = (0.65, 0.3, 0.1)

    @classmethod
    def none(cls) -> "SubWeights":
        """Ignore the bench entirely."""
        return cls(gk=0.0, outfield=(0.0, 0.0, 0.0))

    @classmethod
    def full(cls) -> "SubWeights":
        """Count every substitute in full, which is what a bench boost does."""
        return cls(gk=1.0, outfield=(1.0, 1.0, 1.0))


class Squad:
    """
    Squad class.  Contains 15 players
    """

    def __init__(self, budget: int = 1000, season: str = CURRENT_SEASON) -> None:
        """
        constructor - start with an initial empty player list,
        and £100M
        """
        self.players: list[SquadPlayer] = []
        self.budget = budget
        self.season = season
        self.num_position: dict[str, int] = dict.fromkeys(Position, 0)
        self.free_subs = 0
        self.subs_this_week = 0
        self.count_per_team: defaultdict[str, int] = defaultdict(int)

    def __repr__(self) -> str:
        """Return a concise representation without rendering to the console."""
        return f"Squad(players={len(self.players)}, budget={self.budget})"

    def is_complete(self) -> bool:
        """
        See if we have a full squad.
        """
        return sum(self.num_position.values()) == SQUAD_SIZE

    def add_player(
        self,
        p: SquadPlayer | int | str | Player,
        price: int | None = None,
        gameweek: int | None = None,
        check_budget: bool = True,
        check_team: bool = True,
        dbsession: Session | None = None,
    ) -> bool:
        """
        Add a player.  Can do it by name or by player_id.
        If no price is specified, CandidatePlayer constructor will use the
        current price as found in DB, but if one is specified, we override
        with that value.
        """
        # dbsession is passed through unresolved: CandidatePlayer keeps it, and this
        # Squad gets pickled onto the optimiser's multiprocessing queue.
        gameweek = next_gameweek() if gameweek is None else gameweek
        if isinstance(p, int | str | Player):
            player: SquadPlayer = CandidatePlayer(
                p, self.season, gameweek, purchase_price=price, dbsession=dbsession
            )
        else:  # already a CandidatePlayer (or an equivalent test class)
            player = p
            player.season = self.season
            if price is not None:
                player.purchase_price = price

        logger.debug("Adding player %s", p)

        if player.position == "MNG":
            logger.warning(
                "Skipped adding manager %s, assistant manager not implemented. "
                "Reduced squad budget by %s.",
                player,
                player.purchase_price,
            )
            self.budget -= player.purchase_price
            return True

        # check if constraints are met
        if not self.check_no_duplicate_player(player):
            logger.debug("Already have %s in team", player)
            return False
        if not self.check_num_in_position(player):
            logger.debug(
                "Unable to add player %s - too many %s", player, player.position
            )
            return False
        if check_budget and not self.check_cost(player):
            logger.debug("Cannot afford player %s", player)
            return False
        if check_team and not self.check_num_per_team(player):
            logger.debug(
                "Cannot add %s - too many players from %s", player, player.team
            )
            return False
        self.players.append(player)
        self.count_per_team[player.team] += 1
        self.num_position[player.position] += 1
        self.budget -= player.purchase_price
        return True

    def remove_player(
        self,
        player_id: int | str,
        price: int | None = None,
        gameweek: int | None = None,
        use_api: bool = False,
        dbsession: Session | None = None,
    ) -> bool:
        """
        Remove player from our list.
        If a price is specified, we use that, otherwise we
        calculate the player's sale price based on his price in the
        team vs. his current price in the API (or if the API fails
        or use_api is False, the current price for that player in the database.)
        """
        gameweek = next_gameweek() if gameweek is None else gameweek
        dbsession = dbsession if dbsession is not None else get_session()
        for p in self.players:
            if p.player_id == player_id:
                if price:
                    self.budget += price
                else:
                    self.budget += self.get_sell_price_for_player(
                        p,
                        use_api=use_api,
                        gameweek=gameweek,
                        dbsession=dbsession,
                    )
                self.num_position[p.position] -= 1
                self.count_per_team[p.team] -= 1
                self.players.remove(p)
                return True
        return False

    def get_player_from_id(self, player_id: int | str) -> SquadPlayer:
        for p in self.players:
            if p.player_id == player_id:
                return p
        msg = f"Player {player_id} not in squad"
        raise ValueError(msg)

    def get_sell_price_for_player(
        self,
        player: SquadPlayer | int,
        use_api: bool = False,
        gameweek: int | None = None,
        dbsession: Session | None = None,
        fetcher: FPLDataFetcher | None = None,
    ) -> int:
        """Get sale price for player (a player in self.players) in the current
        gameweek of the current season.
        """
        if isinstance(player, int):
            player = self.get_player_from_id(player)  # get CandidatePlayer from squad
        return sell_price(
            player,
            self.season,
            use_api=use_api,
            gameweek=gameweek,
            dbsession=dbsession,
            fetcher=fetcher,
        )

    def check_no_duplicate_player(self, player: SquadPlayer) -> bool:
        """
        Check we don't already have the player.
        """
        return all(p.player_id != player.player_id for p in self.players)

    def check_num_in_position(self, player: SquadPlayer) -> bool:
        """
        check we have fewer than the limit of
        num players in the chosen players position.
        """
        position = player.position
        return self.num_position[position] < TOTAL_PER_POSITION[position]

    def check_num_per_team(self, player: SquadPlayer) -> bool:
        """
        Check that the squad currently has a maximum of 3 players from the same team,
        and that adding the specified player would not exceed this limit.
        """
        return (
            self.count_per_team[player.team] < 3
            and max(self.count_per_team.values()) < 4
        )

    def check_cost(self, player: SquadPlayer) -> bool:
        """
        check we can afford the player.
        """
        return player.purchase_price <= self.budget

    def _calc_expected_points(self, tag: str) -> None:
        """
        estimate the expected points for the specified gameweek.
        If no gameweek is specified, it will be the next fixture
        """
        for p in self.players:
            p.calc_predicted_points(tag)

    def order_substitutes(self, gameweek: int, tag: str) -> None:
        """Number the bench by predicted points, best first."""
        order_substitutes(self.players, gameweek, tag)

    def total_points_for_starting_11(
        self, gameweek: int, tag: str, triple_captain: bool = False
    ) -> float:
        """
        simple sum over starting players
        """
        total = 0.0
        for player in self.players:
            if player.is_starting:
                total += player.predicted_points[tag][gameweek]
                if player.is_captain:
                    total += player.predicted_points[tag][gameweek]
                    if triple_captain:
                        total += player.predicted_points[tag][gameweek]

        return total

    def total_points_for_subs(
        self, gameweek: int, tag: str, sub_weights: "SubWeights | None" = None
    ) -> float:
        # None means a bench boost: every substitute counts in full
        sub_weights = sub_weights if sub_weights is not None else SubWeights.full()
        outfield_subs = [
            p
            for p in self.players
            if (not p.is_starting) and (p.position != Position.GK)
        ]
        outfield_subs = sorted(outfield_subs, key=bench_position)

        gk_sub = next(
            p
            for p in self.players
            if (not p.is_starting) and (p.position == Position.GK)
        )

        total: float = sub_weights.gk * gk_sub.predicted_points[tag][gameweek]

        for i, player in enumerate(outfield_subs):
            total += sub_weights.outfield[i] * player.predicted_points[tag][gameweek]

        return total

    def optimize_lineup(self, gameweek: int, tag: str) -> None:
        if not self.is_complete():
            msg = "Squad is incomplete"
            raise RuntimeError(msg)

        self._calc_expected_points(tag)
        choose_starting_eleven(
            self.players,
            gameweek,
            tag,
            lambda: self.total_points_for_starting_11(gameweek, tag),
        )
        pick_captains(self.players, gameweek, tag)

    def get_expected_points(
        self,
        gameweek: int,
        tag: str,
        bench_boost: bool = False,
        triple_captain: bool = False,
    ) -> float:
        """
        expected points for the starting 11.
        """

        self.optimize_lineup(gameweek, tag)

        total_score = self.total_points_for_starting_11(
            gameweek, tag, triple_captain=triple_captain
        )

        if bench_boost:
            total_score += self.total_points_for_subs(gameweek, tag)

        return total_score

    def get_actual_points(
        self,
        gameweek: int,
        season: str,
        triple_captain: bool = False,
        bench_boost: bool = False,
    ) -> int:
        """
        Calculate the actual points a squad stored in a historical gameweek/season.
        """
        total_points = 0
        # we will first loop through the list of players to identify
        # subs / captain / vice captain changes, and add up scores
        # for the starting 11, and then after that deal with points
        # for subs and vice captain.

        need_vice_captain = False
        vice_captain_points = 0

        # this will be used to make an ordered list of subs
        subs: list[tuple[int, SquadPlayer]] = []
        need_sub = []
        for p in self.players:
            if p.is_starting or bench_boost:
                scores = get_playerscores_for_player_gameweek(
                    p.player_id, gameweek, season
                )
                minutes = sum(s.minutes for s in scores)
                if minutes > 0:
                    for score in scores:
                        total_points += score.points
                        if p.is_captain:
                            # double their score!
                            total_points += score.points
                            if triple_captain:
                                # TREBLE their score!
                                total_points += score.points
                        elif p.is_vice_captain:
                            vice_captain_points = score.points
                else:  # starting player didn't get any minutes
                    need_sub.append(p)
                    if p.is_captain:
                        need_vice_captain = True

            else:  # player not in our initial starting 11
                # put the subs in order
                subs.append((bench_position(p), p))

        ordered_subs = [s[1] for s in sorted(subs, key=itemgetter(0))]

        # now take account of possibility that captain didn't play
        if need_vice_captain:
            total_points += vice_captain_points  # double them
            if triple_captain:
                total_points += vice_captain_points  # TREBLE them!
        # now take account of subs.
        # UNLESS bench_boost (in which case we've already counted subs points)
        if need_sub and not bench_boost:
            for p_out in need_sub:
                for p_in in ordered_subs:
                    if not is_substitution_allowed(self.players, p_out, p_in):
                        continue
                    scores = get_playerscores_for_player_gameweek(
                        p_in.player_id, gameweek, season
                    )
                    minutes = sum(s.minutes for s in scores)
                    if minutes > 0:
                        for score in scores:
                            total_points += score.points
                        ordered_subs.remove(p_in)
                        break
        return total_points

    def sale_value(self, gameweek: int, use_api: bool) -> int:
        total_value = self.budget  # initialise total to amount in the bank
        for p in self.players:
            total_value += self.get_sell_price_for_player(
                p, use_api=use_api, gameweek=gameweek
            )
        return total_value


def get_current_squad_from_api(
    fpl_team_id: int, fetcher: FPLDataFetcher | None = None, next_gw: int | None = None
) -> Squad:
    """
    Return a list [(player_id, purchase_price)] from the current picks.
    Requires the data fetcher to be logged in.
    """
    fetcher = fetcher if fetcher is not None else get_fetcher()
    next_gw = next_gameweek() if next_gw is None else next_gw
    picks = fetcher.get_current_picks(fpl_team_id)

    squad = Squad(season=CURRENT_SEASON)
    for p in picks.values():
        player = get_player_from_api_id(p["element"])
        if not player:
            continue
        squad.add_player(
            player,
            price=p["purchase_price"],
            gameweek=next_gw,
            check_budget=False,
            check_team=False,
        )
    squad.budget = get_bank(fpl_team_id, season=CURRENT_SEASON, fetcher=fetcher)

    return squad
