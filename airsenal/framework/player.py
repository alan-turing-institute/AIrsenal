"""
Class for a player in FPL
"""

from .schema import Player
from .utils import get_player, get_predicted_points_for_player, CURRENT_SEASON, NEXT_GAMEWEEK


class CandidatePlayer(object):
    """
    player class
    """

    def __init__(self, player, season=CURRENT_SEASON,
                 gameweek=NEXT_GAMEWEEK, purchase_price=None, dbsession=None):
        """
        initialize either by name or by ID
        """
        self.dbsession = dbsession
        if isinstance(player, Player):
            pdata = player
        else:
            pdata = get_player(player, self.dbsession)
            
        if not pdata:
            raise ValueError("Could not find player", player.player_id)
        
        self.pdata = pdata
        self.player_id = pdata.player_id
        self.name = pdata.name
        self.team = pdata.team(season, gameweek)
        self.position = pdata.position(season)
        
        if purchase_price:
            self.purchase_price = purchase_price
        else:
            self.purchase_price = pdata.price(season, gameweek)
        
        self.is_starting = True  # by default
        self.is_captain = False  # by default
        self.is_vice_captain = False  # by default
        self.predicted_points = {}

    def sale_price(self, use_api=False, season=CURRENT_SEASON,
                   gameweek=NEXT_GAMEWEEK):
        """Get sale price for player (a player in self.players) in the current
        gameweek of the current season.
        """
        price_now = None
        
        if use_api and season == CURRENT_SEASON and gameweek >= NEXT_GAMEWEEK:
            try:
                # first try getting the price for the player from the API
                data = fetcher.get_player_summary_data()
                price_now = data[self.player_id]["now_cost"]
            except:
                pass
            
        if not price_now:
            price_now = self.pdata.price(season, gameweek)
            
        if not price_now:
            # if all else fails just use the purchase price as the sale
            # price for this player.
            print("Using purchase price as sale price for",
                  self.player_id,
                  self.name)
            price_now = self.purchase_price
        
        if price_now > self.purchase_price:
            price_sell = (price_now + self.purchase_price) // 2
        else:
            price_sell = price_now
        
        return price_sell

    def calc_predicted_points(self, method):
        """
        get expected points from the db.
        Will be a dict of dicts, keyed by method and gameweeek
        """
        if not method in self.predicted_points.keys():
            self.predicted_points[method] = get_predicted_points_for_player(
                self.player_id, method, dbsession=self.dbsession
            )

    def get_predicted_points(self, gameweek, method):
        """
        get points for a specific gameweek
        """
        if not method in self.predicted_points.keys():
            self.calc_predicted_points(method)
        if not gameweek in self.predicted_points[method].keys():
            print(
                "No prediction available for {} week {}".format(
                    self.data.name, gameweek
                )
            )
            return 0.
        return self.predicted_points[method][gameweek]
