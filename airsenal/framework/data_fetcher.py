"""
Classes to query the FPL API to retrieve current FPL data,
and to query football-data.org to retrieve match and fixture data.
"""
import os
import requests
import json
import time

from .mappings import alternative_team_names


class FPLDataFetcher(object):
    """
    hold current and historic FPL data in memory,
    or retrieve it if not already cached.
    """

    def __init__(self):
        self.current_summary_data = None
        self.current_event_data = None
        self.current_player_data = None
        self.current_team_data = None
        self.player_gameweek_data = {}
        self.fpl_team_history_data = None
        self.fpl_transfer_history_data = None
        self.fpl_league_data = None
        self.fpl_team_data = {} # players in squad, by gameweek
        self.fixture_data = None
        for ID in ["FPL_LEAGUE_ID",
                   "FPL_TEAM_ID",
                   "FPL_LOGIN",
                   "FPL_PASSWORD"]:
            if ID in os.environ.keys():
                self.__setattr__(ID, os.environ[ID])
            elif os.path.exists(os.path.join(os.path.dirname(__file__), "../data/{}".format(ID))):
                self.__setattr__(
                    ID, open(
                        os.path.join(os.path.dirname(__file__), "../data/{}".format(ID))
                    ).read().strip())
            else:
                print("Couldn't find {} - some data may be unavailable".format(ID))
                self.__setattr__(ID, "MISSING_ID")
        self.FPL_SUMMARY_API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
        self.FPL_DETAIL_URL = "https://fantasy.premierleague.com/api/element-summary/{}/"
        self.FPL_HISTORY_URL = "https://fantasy.premierleague.com/api/entry/{}/history/"
        self.FPL_TEAM_URL = "https://fantasy.premierleague.com/api/entry/{}/event/{}/picks/"
        self.FPL_TEAM_TRANSFER_URL = "https://fantasy.premierleague.com/api/entry/{}/transfers/"
        self.FPL_LEAGUE_URL = "https://fantasy.premierleague.com/api/leagues-classic/{}/standings/?page_new_entries=1&page_standings=1".format(
            self.FPL_LEAGUE_ID
        )
        self.FPL_FIXTURE_URL = "https://fantasy.premierleague.com/api/fixtures/"



    def get_current_summary_data(self):
        """
        return cached data if present, otherwise retrieve it
        from the API.
        """
        if self.current_summary_data:
            return self.current_summary_data
        else:
            r = requests.get(self.FPL_SUMMARY_API_URL)
            if not r.status_code == 200:
                print("Unable to access FPL API")
                return None
            self.current_summary_data = json.loads(r.content.decode("utf-8"))
        return self.current_summary_data


    def get_fpl_team_data(self, gameweek, team_id=None):
        """
        Use team id to get team data from the FPL API.
        If no team_id is specified, we assume it is 'our' team
        $TEAM_ID, and cache the results in a dictionary.
        """
        if not team_id and gameweek in self.fpl_team_data.keys():
            return self.fpl_team_data[gameweek]
        else:
            if not team_id:
                team_id = self.FPL_TEAM_ID
            url = self.FPL_TEAM_URL.format(team_id,gameweek)
            r = requests.get(url)
            if not r.status_code == 200:
                print("Unable to access FPL team API {}".format(url))
                return None
            team_data = json.loads(r.content.decode("utf-8"))
            if not team_id:
                self.fpl_team_data[gameweek] = team_data['picks']
        return team_data['picks']


    def get_fpl_team_history_data(self, team_id=None):
        """
        Use our team id to get history data from the FPL API.
        """
        if self.fpl_team_history_data and not team_id:
            return self.fpl_team_history_data
        else:
            if not team_id:
                team_id = self.FPL_TEAM_ID
            url = self.FPL_HISTORY_URL.format(team_id)
            r = requests.get(url)
            if not r.status_code == 200:
                print("Unable to access FPL team history API")
                return None
            self.fpl_team_history_data = json.loads(r.content.decode("utf-8"))
        return self.fpl_team_history_data


    def get_fpl_transfer_data(self):
        """
        Get our transfer history from the FPL API.
        """
        ## return cached value if we already retrieved it.
        if self.fpl_transfer_history_data:
            return self.fpl_transfer_history_data
        ## or get it from the API.
        url = self.FPL_TEAM_TRANSFER_URL.format(self.FPL_TEAM_ID)
        r=requests.get(url)
        if not r.status_code == 200:
            print("Unable to access FPL transfer history API")
            return None
        self.fpl_transfer_history_data = json.loads(r.content.decode("utf-8"))
        return self.fpl_transfer_history_data


    def get_fpl_league_data(self):
        """
        Use our league id to get history data from the FPL API.
        """
        if self.fpl_league_data:
            return self.fpl_league_data
        else:
            headers = {"login": self.FPL_LOGIN,
                       "password": self.FPL_PASSWORD,
                       "app": "plfpl-web",
                       "redirect_uri": "https://fantasy.premierleague.com/"}
            r = requests.get(self.FPL_LEAGUE_URL, headers=headers)
            if not r.status_code == 200:
                print("Unable to access FPL league API")
                return None
            self.fpl_league_data = json.loads(r.content.decode("utf-8"))
        return self.fpl_league_data

    def get_event_data(self):
        """
        return a dict of gameweeks - whether they are finished or not, and
        the transfer deadline.
        """
        if self.current_event_data:
            return self.current_event_data
        self.current_event_data = {}
        all_data = self.get_current_summary_data()
        for event in all_data["events"]:
            self.current_event_data[event["id"]] = {
                "deadline": event["deadline_time"],
                "is_finished": event["finished"],
            }
        return self.current_event_data

    def get_player_summary_data(self):
        """
        Use the current_data to build a dictionary, keyed by player_id
        in order to retrieve a player without having to loop through
        a whole list.
        """
        if self.current_player_data:
            return self.current_player_data
        self.current_player_data = {}
        all_data = self.get_current_summary_data()
        for player in all_data["elements"]:
            self.current_player_data[player["id"]] = player
        return self.current_player_data

    def get_current_team_data(self):
        """
        Use the current_data to build a dictionary keyed by team code,
        in order to retrieve a player's team without looping through the
        whole list.
        """
        if self.current_team_data:
            return self.current_team_data
        self.current_team_data = {}
        all_data = self.get_current_summary_data()
        for team in all_data["teams"]:
            self.current_team_data[team["code"]] = team
        return self.current_team_data

    def get_gameweek_data_for_player(self, player_id, gameweek=None):
        """
        return cached data if available, otherwise
        fetch it from API.
        Return a list, as in double-gameweeks, a player can play more than
        one match in a gameweek.
        """
        if not player_id in self.player_gameweek_data.keys():
            self.player_gameweek_data[player_id] = {}
            if (not gameweek) or (
                not gameweek in self.player_gameweek_data[player_id].keys()
            ):

                r = requests.get(self.FPL_DETAIL_URL.format(player_id))
                if not r.status_code == 200:
                    print("Error retrieving data for player {}".format(player_id))
                    return []
                player_detail = json.loads(r.content)

                for game in player_detail["history"]:
                    gw = game["round"]
                    if not gw in self.player_gameweek_data[player_id].keys():
                        self.player_gameweek_data[player_id][gw] = []
                    self.player_gameweek_data[player_id][gw].append(game)
        if gameweek:
            if not gameweek in self.player_gameweek_data[player_id].keys():
                print(
                    "Data not available for player {} week {}".format(
                        player_id, gameweek
                    )
                )
                return []
            return self.player_gameweek_data[player_id][gameweek]
        else:
            return self.player_gameweek_data[player_id]

    def get_fixture_data(self):
        """
        Get the fixture list from the FPL API.
        """
        if not self.fixture_data:
            self.fixture_data = requests.get(self.FPL_FIXTURE_URL).json()
        else:
            pass
        return self.fixture_data


class MatchDataFetcher(object):
    """
    Access the football-data.org API to get information on match results and fixtures.
    """

    def __init__(self):
        self.data = {}
        self.FOOTBALL_DATA_URL = "http://api.football-data.org/v2/competitions/2021"
        if "FD_API_KEY" in os.environ.keys():
            self.__setattr__("FOOTBALL_DATA_API_KEY", os.environ["FD_API_KEY"])
        elif os.path.exists(os.path.join(os.path.dirname(__file__), "../data/FD_API_KEY")):
            self.__setattr__("FOOTBALL_DATA_API_KEY", open(
                os.path.join(os.path.dirname(__file__), "../data/FD_API_KEY")
            ).read().strip())
        else:
            print("Couldn't find FD_API_KEY - can't use football-data.org API")
        pass


    def _make_request(self, url, headers):
        """
        API rate limit means we sometimes have to wait.
        """
        status_code = 0
        content = {}
        while not status_code == 200:
            r = requests.get(url, headers=headers)
            status_code = r.status_code
            content = json.loads(r.content)
            if (status_code != 200):
                if 'request limit' in content['message']:
                    time.sleep(61)
                else:
                    raise RuntimeError("Unable to make request {} {}".format(status_code, content))
        return content


    def _get_gameweek_data(self, gameweek, season):
        """
        query the matches endpoint
        """
        print("Getting gameweek data for {}".format(gameweek))
        uri = "{}/matches?matchday={}&season={}".format(self.FOOTBALL_DATA_URL, gameweek, season)
        headers = {"X-Auth-Token": self.FOOTBALL_DATA_API_KEY}
        request_data = self._make_request(uri, headers)
        self.data[gameweek] = request_data["matches"]



    def get_results(self, gameweek, season="2019"):
        """
        get results for matches that have been played.
        Return list of tuples:
        [(date, home_team, away_team, home_score, away_score)]
        """
        output_list = []
        if not gameweek in self.data.keys():
            self._get_gameweek_data(gameweek, season)
            time.sleep(1)
#        if not self.data[gameweek][0]["status"] == "FINISHED":
#            print("Fixtures not finished - have they been played yet?")
#            return output_list

        for match in self.data[gameweek]:
            home_team = None
            away_team = None
            for k, v in alternative_team_names.items():
                if match["homeTeam"]["name"] in v:
                    home_team = k
                elif match["awayTeam"]["name"] in v:
                    away_team = k
                if home_team and away_team:
                    break
            output_list.append(
                (
                    match["utcDate"],
                    home_team,
                    away_team,
                    match["score"]["fullTime"]["homeTeam"],
                    match["score"]["fullTime"]["awayTeam"],
                )
            )
        return output_list

    def get_fixtures(self, gameweek, season="2019"):
        """
        get upcoming fixtures.
        Return list of tuples:
        [(date, home_team, away_team)]
        """
        output_list = []
        if not gameweek in self.data.keys():
            self._get_gameweek_data(gameweek,season)
        if not self.data[gameweek][0]["status"] == "SCHEDULED":
            print("Fixtures not scheduled - have they already been played?")
            return output_list

        for fixture in self.data[gameweek]:
            home_team = None
            away_team = None
            for k, v in alternative_team_names.items():
                if fixture["homeTeam"] in v:
                    home_team = k
                elif fixture["awayTeam"] in v:
                    away_team = k
                if home_team and away_team:
                    break
            output_list.append((fixture["utcDate"], home_team, away_team))
        return output_list
