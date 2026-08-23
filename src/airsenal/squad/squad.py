"""
The class for an FPL squad.
Contains a set of players.
Is able to check that it obeys all constraints.
"""

from collections import defaultdict
from operator import itemgetter

import numpy as np
from sqlalchemy.orm import Session

from airsenal.core.logging import get_logger
from airsenal.core.season import CURRENT_SEASON
from airsenal.db.models import Player
from airsenal.db.queries.gameweeks import next_gameweek
from airsenal.db.queries.players import get_player, get_player_from_api_id
from airsenal.db.queries.scores import get_playerscores_for_player_gameweek
from airsenal.db.session import get_session
from airsenal.fetch.fpl_api import FPLDataFetcher, get_fetcher
from airsenal.squad.player import CandidatePlayer, DummyPlayer
from airsenal.squad.state import get_bank

logger = get_logger(__name__)

# how many players do we need to add
TOTAL_PER_POSITION = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

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


class Squad:
    """
    Squad class.  Contains 15 players
    """

    def __init__(self, budget=1000, season=CURRENT_SEASON):
        """
        constructor - start with an initial empty player list,
        and £100M
        """
        self.players = []
        self.budget = budget
        self.season = season
        self.num_position = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
        self.free_subs = 0
        self.subs_this_week = 0
        self.count_per_team = defaultdict(int)

    def __repr__(self) -> str:
        """Return a concise representation without rendering to the console."""
        return f"Squad(players={len(self.players)}, budget={self.budget})"

    def is_complete(self):
        """
        See if we have 15 players.
        """
        num_players = sum(self.num_position.values())
        return num_players == 15

    def add_player(
        self,
        p: CandidatePlayer | DummyPlayer | int | str | Player,
        price=None,
        gameweek: int | None = None,
        check_budget=True,
        check_team=True,
        dbsession: Session | None = None,
    ):
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
            player: CandidatePlayer | DummyPlayer = CandidatePlayer(
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
        player_id,
        price=None,
        gameweek: int | None = None,
        use_api=False,
        dbsession: Session | None = None,
    ):
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

    def get_player_from_id(self, player_id):
        for p in self.players:
            if p.player_id == player_id:
                return p
        msg = f"Player {player_id} not in squad"
        raise ValueError(msg)

    def get_sell_price_for_player(
        self,
        player,
        use_api=False,
        gameweek: int | None = None,
        dbsession: Session | None = None,
        fetcher: FPLDataFetcher | None = None,
    ):
        """Get sale price for player (a player in self.players) in the current
        gameweek of the current season.
        """
        fetcher = fetcher if fetcher is not None else get_fetcher()
        gameweek = next_gameweek() if gameweek is None else gameweek
        dbsession = dbsession if dbsession is not None else get_session()
        if isinstance(player, int):
            player = self.get_player_from_id(player)  # get CandidatePlayer from squad
        player_id = player.player_id

        price_now = None
        player_db = get_player(player_id, dbsession=dbsession)
        if (
            use_api
            and self.season == CURRENT_SEASON
            and gameweek >= next_gameweek()
            and player_db is not None
            and player_db.fpl_api_id is not None
        ):
            api_id = player_db.fpl_api_id
            # first try getting the actual sale price from a logged in API
            try:
                return fetcher.get_current_picks()[api_id]["selling_price"]
            except Exception:
                logger.warning(
                    "Failed to login to get actual sale price for %s from API. "
                    "Will estimate based on the player's current price instead",
                    player,
                    exc_info=True,
                )
            # if not logged in, just get current price from API
            try:
                price_now = fetcher.get_player_summary_data()[api_id]["now_cost"]
            except Exception:
                logger.warning(
                    "Failed to get current price of %s from API. "
                    "Will attempt to use latest price in DB instead.",
                    player,
                    exc_info=True,
                )

        # retrieve how much we originally bought the player for from db
        price_bought = player.purchase_price

        # get player's current price from db if the API wasn't used
        if not price_now and player_db:
            price_now = player_db.price(self.season, gameweek)

        # if all else fails just use the purchase price as the sale price for the player
        if not price_now:
            logger.warning(
                "Using purchase price as sale price for %s, %s",
                player.player_id,
                player,
            )
            price_now = price_bought

        if price_now > price_bought:
            return (price_now + price_bought) // 2
        return price_now

    def check_no_duplicate_player(self, player):
        """
        Check we don't already have the player.
        """
        return all(p.player_id != player.player_id for p in self.players)

    def check_num_in_position(self, player):
        """
        check we have fewer than the limit of
        num players in the chosen players position.
        """
        position = player.position
        return self.num_position[position] < TOTAL_PER_POSITION[position]

    def check_num_per_team(self, player):
        """
        Check that the squad currently has a maximum of 3 players from the same team,
        and that adding the specified player would not exceed this limit.
        """
        return (
            self.count_per_team[player.team] < 3
            and max(self.count_per_team.values()) < 4
        )

    def check_cost(self, player):
        """
        check we can afford the player.
        """
        return player.purchase_price <= self.budget

    def _calc_expected_points(self, tag):
        """
        estimate the expected points for the specified gameweek.
        If no gameweek is specified, it will be the next fixture
        """
        for p in self.players:
            p.calc_predicted_points(tag)

    def optimize_subs(self, gameweek, tag):
        """
        based on pre-calculated expected points,
        choose the best starting 11, obeying constraints.
        """
        # first order all the players by expected points
        player_dict: dict[str, list[tuple[CandidatePlayer, float]]] = {
            "GK": [],
            "DEF": [],
            "MID": [],
            "FWD": [],
        }
        for p in self.players:
            try:
                points_prediction = p.predicted_points[tag][gameweek]

            except KeyError:
                # player does not have a game in this gameweek
                points_prediction = 0
            player_dict[p.position].append((p, points_prediction))
        for v in player_dict.values():
            v.sort(key=itemgetter(1), reverse=True)

        # always start the first-placed and sub the second-placed keeper
        player_dict["GK"][0][0].is_starting = True
        player_dict["GK"][1][0].is_starting = False
        best_score = 0.0
        best_formation = None
        for f in FORMATIONS:
            self.apply_formation(player_dict, f)
            score = self.total_points_for_starting_11(gameweek, tag)
            if score >= best_score:
                best_score = score
                best_formation = f
        logger.debug("Best formation is %s", best_formation)
        self.apply_formation(player_dict, best_formation)
        self.order_substitutes(gameweek, tag)

        return best_score

    def order_substitutes(self, gameweek, tag):
        # order substitutes by expected points (descending)
        subs = [p for p in self.players if not p.is_starting]

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

    def apply_formation(self, player_dict, formation):
        """
        set players' is_starting to True or False
        depending on specified formation in format e.g.
        (4,4,2)
        """
        for i, pos in enumerate(["DEF", "MID", "FWD"]):
            for index, player in enumerate(player_dict[pos]):
                player[0].is_starting = index < formation[i]

    def get_formation(self):
        """
        Return the formation of a starting 11 in the form
        of a dict {"DEF": nDEF, "MID": nMID, "FWD": nFWD}
        """
        formation = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
        for player in self.players:
            if player.is_starting:
                formation[player.position] += 1
        return formation

    def is_substitution_allowed(self, player_out, player_in):
        """
        for a given player out and player in, would the substitution result in a
        valid formation?
        """
        formation = self.get_formation()
        formation[player_out.position] -= 1
        formation[player_in.position] += 1
        return (formation["DEF"], formation["MID"], formation["FWD"]) in FORMATIONS

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
        self, gameweek: int, tag: str, sub_weights: dict | None = None
    ) -> float:
        if sub_weights is None:
            sub_weights = {"GK": 1, "Outfield": (1, 1, 1)}
        outfield_subs = [
            p for p in self.players if (not p.is_starting) and (p.position != "GK")
        ]
        outfield_subs = sorted(outfield_subs, key=lambda p: p.sub_position)

        gk_sub = next(
            p for p in self.players if (not p.is_starting) and (p.position == "GK")
        )

        total = sub_weights["GK"] * gk_sub.predicted_points[tag][gameweek]

        for i, player in enumerate(outfield_subs):
            total += sub_weights["Outfield"][i] * player.predicted_points[tag][gameweek]

        return total

    def optimize_lineup(self, gameweek: int, tag: str):
        if not self.is_complete():
            msg = "Squad is incomplete"
            raise RuntimeError(msg)

        self._calc_expected_points(tag)
        self.optimize_subs(gameweek, tag)
        self.pick_captains(gameweek, tag)

    def get_expected_points(
        self, gameweek, tag, bench_boost=False, triple_captain=False
    ):
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

    def pick_captains(self, gameweek, tag):
        """
        pick the highest two expected points for captain and vice-captain
        """
        player_list = []
        for p in self.players:
            p.is_captain = False
            p.is_vice_captain = False
            player_list.append((p, p.predicted_points[tag][gameweek]))

        player_list.sort(key=itemgetter(1), reverse=True)
        player_list[0][0].is_captain = True
        player_list[1][0].is_vice_captain = True

    def get_actual_points(
        self, gameweek, season, triple_captain=False, bench_boost=False
    ):
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
        subs: list[tuple[int, Player]] = []
        need_sub = []
        for p in self.players:
            if p.is_starting or bench_boost:
                scores = get_playerscores_for_player_gameweek(p, gameweek, season)
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
                subs.append((p.sub_position, p))

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
                    if not self.is_substitution_allowed(p_out, p_in):
                        continue
                    scores = get_playerscores_for_player_gameweek(
                        p_in, gameweek, season
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
