"""
Classes to query the FPL API.

The login flow lives in `fpl_auth.py` and the request helpers in `fpl_http.py`;
what is left here is the endpoints and the responses they cache.

Thanks to:
- https://github.com/amosbastian/fpl/blob/master/fpl/utils.py for posting transfers and
  lineups.
"""

from typing import Any, overload

from airsenal.core.env import (
    FPL_LEAGUE_ID,
    FPL_TEAM_ID,
)
from airsenal.core.logging import get_logger
from airsenal.remote.errors import RemoteError
from airsenal.remote.fpl_auth import FPLAuth
from airsenal.remote.fpl_http import API_HOME, Session, get_json, post_json

logger = get_logger(__name__)

# The endpoints, as constants rather than as attributes rebuilt on every
# instance - none of them depends on anything an instance knows.
FPL_SUMMARY_API_URL = f"{API_HOME}/bootstrap-static/"
FPL_DETAIL_URL = API_HOME + "/element-summary/{}/"
FPL_HISTORY_URL = API_HOME + "/entry/{}/history/"
FPL_TEAM_URL = API_HOME + "/entry/{}/event/{}/picks/"
FPL_GET_TRANSFERS_URL = API_HOME + "/entry/{}/transfers/"
FPL_SET_TRANSFERS_URL = API_HOME + "/transfers/"
FPL_FIXTURE_URL = f"{API_HOME}/fixtures/"
FPL_MYTEAM_URL = API_HOME + "/my-team/{}/"


class FPLDataFetcher:
    """
    hold current and historic FPL data in memory,
    or retrieve it if not already cached.
    """

    def __init__(
        self,
        fpl_team_id: int | None = None,
        rsession: Session | None = None,
        auth: FPLAuth | None = None,
    ) -> None:
        # `rsession` stays a parameter of its own: it is how a test hands in a
        # session that cannot connect.
        self.auth = auth if auth is not None else FPLAuth(rsession)
        # The FPL API is not typed and not versioned, so a payload is a
        # dict[str, Any]; what each cache is keyed by is worth being exact about.
        self.current_summary_data: dict[str, Any] = {}
        self.current_event_data: dict[int, dict[str, Any]] = {}  # by gameweek
        self.current_player_data: dict[int, dict[str, Any]] = {}  # by player api id
        self.current_team_data: dict[int, dict[str, Any]] = {}  # by team code
        self.current_squad_data: dict[int, dict[str, Any]] = {}  # by fpl_team_id
        # by player api id, then gameweek - a player can have two in a double GW
        self.player_gameweek_data: dict[int, dict[int, list[dict[str, Any]]]] = {}
        self.fpl_team_history_data: dict[str, Any] = {}
        # transfer history data is a dict, keyed by fpl_team_id
        self.fpl_transfer_history_data: dict[int, list[dict[str, Any]]] = {}
        self.fpl_league_data: dict[str, Any] = {}
        self.fpl_team_data: dict[int, dict[str, Any]] = {}  # squad, by gameweek
        # a list, not a dict: /fixtures/ returns a JSON array. It was declared
        # `dict` and initialised `{}`, which only ever worked because both are
        # falsy and so the emptiness check behaves the same either way.
        self.fixture_data: list[dict[str, Any]] = []

        self.FPL_TEAM_ID = FPL_TEAM_ID if fpl_team_id is None else fpl_team_id
        self.FPL_LEAGUE_ID = FPL_LEAGUE_ID
        self.FPL_LEAGUE_URL = (
            f"{API_HOME}/leagues-classic/{self.FPL_LEAGUE_ID}"
            "/standings/?page_new_entries=1&page_standings=1"
        )

    # The login state, so that callers and tests need not know it is delegated.
    @property
    def logged_in(self) -> bool:
        return self.auth.logged_in

    @property
    def rsession(self) -> Session:
        return self.auth.session

    def login(self) -> None:
        """Log in, if the credentials to do so are available."""
        self.auth.login()

    def _get(
        self, url: str, err_msg: str = "Unable to access FPL API", **params: Any
    ) -> Any:
        """A GET on this fetcher's session, with whatever header login produced."""
        return get_json(
            self.auth.session, url, headers=self.auth.headers, err_msg=err_msg, **params
        )

    def _post(
        self, url: str, data: Any, err_msg: str = "Failed to post data to FPL API"
    ) -> None:
        post_json(
            self.auth.session, url, data, headers=self.auth.headers, err_msg=err_msg
        )

    def get_current_squad_data(self, fpl_team_id: int | None = None) -> dict[str, Any]:
        """
        Requires login.  Return the current squad data, including
        "picks", bank, and free transfers.
        """
        if fpl_team_id is None:
            if self.FPL_TEAM_ID is None:
                msg = "Please specify FPL team ID"
                raise RuntimeError(msg)
            fpl_team_id = self.FPL_TEAM_ID

        if fpl_team_id in self.current_squad_data:
            return self.current_squad_data[fpl_team_id]

        self.login()
        url = FPL_MYTEAM_URL.format(fpl_team_id)
        self.current_squad_data[fpl_team_id] = self._get(url)
        return self.current_squad_data[fpl_team_id]

    def get_current_picks(
        self, fpl_team_id: int | None = None
    ) -> dict[int, dict[str, Any]]:
        """
        Returns the players picked for the upcoming gameweek, including
        purchase and selling prices, and whether they are subs or not.
        Requires login
        """
        squad_data = self.get_current_squad_data(fpl_team_id)
        return {pick["element"]: pick for pick in squad_data["picks"]}

    def get_num_free_transfers(self, fpl_team_id: int | None = None) -> int:
        """
        Returns the number of free transfers for the upcoming gameweek.
        Requires login
        """
        squad_data = self.get_current_squad_data(fpl_team_id)
        return max(
            0,
            int(squad_data["transfers"]["limit"])
            - int(squad_data["transfers"]["made"]),
        )

    def get_current_bank(self, fpl_team_id: int | None = None) -> int:
        """
        Returns the remaining bank (in 0.1M) for the upcoming gameweek.
        Requires login
        """
        squad_data = self.get_current_squad_data(fpl_team_id)
        return int(squad_data["transfers"]["bank"])

    def get_available_chips(self, fpl_team_id: int | None = None) -> list[str]:
        """
        Returns a list of chips that are available to be played in upcoming gameweek.
        Requires login
        """
        squad_data = self.get_current_squad_data(fpl_team_id)
        return [
            chip["name"]
            for chip in squad_data["chips"]
            if chip["status_for_entry"] == "available"
        ]

    def get_current_summary_data(self) -> dict[str, Any]:
        """
        return cached data if present, otherwise retrieve it
        from the API.
        """
        if self.current_summary_data:
            return self.current_summary_data
        self.current_summary_data = self._get(FPL_SUMMARY_API_URL)
        return self.current_summary_data

    def get_fpl_team_data(
        self, gameweek: int, fpl_team_id: int | None = None
    ) -> dict[str, Any]:
        """
        Use FPL team id to get team data from the FPL API.
        If no fpl_team_id is specified, we assume it is 'our' team
        $FPL_TEAM_ID, and cache the results in a dictionary.
        """
        if (not fpl_team_id) and (gameweek in self.fpl_team_data):
            return self.fpl_team_data[gameweek]
        if not fpl_team_id:
            fpl_team_id = self.FPL_TEAM_ID
        url = FPL_TEAM_URL.format(fpl_team_id, gameweek)
        fpl_team_data: dict[str, Any] = self._get(
            url, err_msg=f"Unable to access FPL team API {url}"
        )
        if not fpl_team_id:
            self.fpl_team_data[gameweek] = fpl_team_data
        return fpl_team_data

    def get_fpl_team_history_data(self, team_id: int | None = None) -> dict[str, Any]:
        """
        Use our team id to get history data from the FPL API.
        """
        if self.fpl_team_history_data and not team_id:
            return self.fpl_team_history_data
        if not team_id:
            team_id = self.FPL_TEAM_ID
        url = FPL_HISTORY_URL.format(team_id)
        self.fpl_team_history_data = self._get(
            url, err_msg="Unable to access FPL team history API"
        )
        return self.fpl_team_history_data

    def get_fpl_transfer_data(
        self, fpl_team_id: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Get our transfer history from the FPL API.
        """
        if fpl_team_id is None:
            if self.FPL_TEAM_ID is None:
                msg = "Please specify FPL team ID"
                raise RuntimeError(msg)
            fpl_team_id = self.FPL_TEAM_ID
        # return cached value if we already retrieved it.
        if fpl_team_id in self.fpl_transfer_history_data:
            return self.fpl_transfer_history_data[fpl_team_id]
        # or get it from the API.
        url = FPL_GET_TRANSFERS_URL.format(fpl_team_id)
        # get transfer history from api and reverse order so that
        # oldest transfers at start of list and newest at end.
        self.fpl_transfer_history_data[fpl_team_id] = list(
            reversed(
                self._get(
                    url,
                    (
                        "Unable to access FPL transfer history API for "
                        f"team_id {fpl_team_id}"
                    ),
                )
            )
        )
        return self.fpl_transfer_history_data[fpl_team_id]

    def get_fpl_league_data(self) -> dict[str, Any] | None:
        """
        Use our league id to get history data from the FPL API.
        """
        if self.fpl_league_data:
            return self.fpl_league_data

        self.login()
        # _get_request returns the decoded body and raises on a bad status, so
        # this used to reach for .status_code and .content on a dict and blow up
        # with an AttributeError on every call.
        try:
            self.fpl_league_data = self._get(self.FPL_LEAGUE_URL)
        except RemoteError:
            logger.warning("Unable to access FPL league API")
            return None
        return self.fpl_league_data

    def get_event_data(self) -> dict[int, dict[str, Any]]:
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

    def get_last_finished_gameweek(self) -> int:
        """
        The last gameweek the API has marked as finished, or 0 before the season starts.

        Stops at the first unfinished gameweek rather than taking the maximum, so a
        stray `finished` flag after a gap cannot pull the answer forward.
        """
        event_data = self.get_event_data()
        last_finished = 0
        for gw in sorted(event_data.keys()):
            if not event_data[gw]["is_finished"]:
                return last_finished
            last_finished = gw
        return last_finished

    def get_player_summary_data(self) -> dict[int, dict[str, Any]]:
        """
        Use the current_data to build a dictionary, keyed by player_api_id
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

    def get_current_team_data(self) -> dict[int, dict[str, Any]]:
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

    @overload
    def get_gameweek_data_for_player(
        self, player_api_id: int, gameweek: None = None
    ) -> dict[int, list[dict[str, Any]]]: ...

    @overload
    def get_gameweek_data_for_player(
        self, player_api_id: int, gameweek: int
    ) -> list[dict[str, Any]]: ...

    def get_gameweek_data_for_player(
        self, player_api_id: int, gameweek: int | None = None
    ) -> dict[int, list[dict[str, Any]]] | list[dict[str, Any]]:
        """
        return cached data if available, otherwise
        fetch it from API.
        Return a list, as in double-gameweeks, a player can play more than
        one match in a gameweek.
        """
        if player_api_id not in self.player_gameweek_data:
            self.player_gameweek_data[player_api_id] = {}
            if (not gameweek) or (
                gameweek not in self.player_gameweek_data[player_api_id]
            ):
                player_detail = self._get(
                    FPL_DETAIL_URL.format(player_api_id),
                    f"Error retrieving data for player {player_api_id}",
                )
                for game in player_detail["history"]:
                    gw = game["round"]
                    if gw not in self.player_gameweek_data[player_api_id]:
                        self.player_gameweek_data[player_api_id][gw] = []
                    self.player_gameweek_data[player_api_id][gw].append(game)
        if not gameweek:
            return self.player_gameweek_data[player_api_id]

        if gameweek not in self.player_gameweek_data[player_api_id]:
            logger.warning(
                "Data not available for player %s week %s", player_api_id, gameweek
            )
            return []
        return self.player_gameweek_data[player_api_id][gameweek]

    def get_fixture_data(self) -> list[dict[str, Any]]:
        """
        Get the fixture list from the FPL API.
        """
        if not self.fixture_data:
            self.fixture_data = self._get(FPL_FIXTURE_URL)
        return self.fixture_data

    def get_lineup(self) -> dict[str, Any]:
        """
        Retrieve up to date lineup from api
        """
        self.login()
        team_url = FPL_MYTEAM_URL.format(self.FPL_TEAM_ID)
        lineup: dict[str, Any] = self._get(team_url)
        return lineup

    def post_lineup(self, payload: list[dict[str, Any]]) -> None:
        """Set the lineup for a specific team"""
        self.login()
        body = {"chip": None, "picks": payload}
        team_url = FPL_MYTEAM_URL.format(self.FPL_TEAM_ID)
        self._post(
            team_url,
            body,
            err_msg=(
                "Failed to set lineup. Make the changes manually on the web-site if "
                "needed"
            ),
        )
        logger.info("Lineup set!")

    def post_transfers(self, transfer_payload: dict[str, Any]) -> None:
        """Make transfers via the API.

        WARNING: This can't be undone and may incur points hits. It also doesn't support
        activating chips currently, so this must be done manually especially if you are
        using a wildcard or free hit chip (in which case the transfers will be applied
        as normal transfers with points hits).
        """
        self.login()
        err_msg = (
            "Failed to set transfers. Make the changes manually on the web-site if "
            "needed."
        )
        self._post(
            FPL_SET_TRANSFERS_URL,
            data=transfer_payload,
            err_msg=err_msg,
        )
        logger.info("Transfers made!")


def get_fetcher(fpl_team_id: int | None = None) -> FPLDataFetcher:
    """
    The shared FPL API client, created on first use.

    Cached so that callers keep hitting the same instance and therefore the same
    response cache; a fresh FPLDataFetcher would re-request everything. It lives
    beside the client rather than a layer up, so that anything above `fetch` can
    reach it without importing something that imports everything.
    """
    return FPLDataFetcher(fpl_team_id)


def require_fpl_team_id(fpl_team_id: int | None = None) -> int:
    """
    The FPL team id to act for, or a clear error saying how to set one.

    Three commands resolved this themselves and only one of them checked the
    result, so `airsenal run` with no FPL_TEAM_ID configured passed None all the
    way down into the database setup.
    """
    resolved = fpl_team_id if fpl_team_id is not None else get_fetcher().FPL_TEAM_ID
    if resolved is None:
        msg = (
            "fpl_team_id must be set as an argument, an environment variable, or "
            "in the config file (see `airsenal env set FPL_TEAM_ID`)."
        )
        raise ValueError(msg)
    return resolved
