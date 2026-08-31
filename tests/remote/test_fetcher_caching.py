"""Which responses FPLDataFetcher holds on to, and for whose team."""

from airsenal.remote import fpl_api
from airsenal.remote.fpl_api import FPLDataFetcher, get_fetcher

OWN_TEAM = 111
OTHER_TEAM = 222


class CountingFetcher(FPLDataFetcher):
    """Counts requests instead of making them."""

    def __init__(self, fpl_team_id=None):
        super().__init__(fpl_team_id=fpl_team_id)
        self.urls: list[str] = []

    def _get(self, url, err_msg=None):  # noqa: ARG002
        self.urls.append(url)
        return {"url": url}


def test_our_own_team_data_is_fetched_once_per_gameweek():
    """The read side could never hit, because the write side never ran."""
    fetcher = CountingFetcher(fpl_team_id=OWN_TEAM)

    first = fetcher.get_fpl_team_data(3)
    second = fetcher.get_fpl_team_data(3)

    assert first == second
    assert len(fetcher.urls) == 1


def test_another_teams_data_is_never_cached_as_our_own():
    fetcher = CountingFetcher(fpl_team_id=OWN_TEAM)

    other = fetcher.get_fpl_team_data(3, fpl_team_id=OTHER_TEAM)
    own = fetcher.get_fpl_team_data(3)

    assert other != own
    assert str(OTHER_TEAM) in other["url"]
    assert str(OWN_TEAM) in own["url"]


def test_history_for_another_team_does_not_poison_our_own():
    """
    Asking about someone else used to overwrite our own cached history.

    The next call with no team id then handed back whoever was asked about last.
    """
    fetcher = CountingFetcher(fpl_team_id=OWN_TEAM)

    other = fetcher.get_fpl_team_history_data(OTHER_TEAM)
    own = fetcher.get_fpl_team_history_data()

    assert str(OTHER_TEAM) in other["url"]
    assert str(OWN_TEAM) in own["url"]
    assert fetcher.get_fpl_team_history_data() == own


def test_the_default_team_gets_one_client_however_it_is_asked_for(monkeypatch):
    """
    `functools.cache` keys on what the caller passed, not on what they meant.

    `get_fetcher()`, `get_fetcher(None)`, `get_fetcher(fpl_team_id=None)` and the
    default team named explicitly are one entry, not four clients with four
    empty response caches.
    """
    monkeypatch.setattr(fpl_api, "FPL_TEAM_ID", OWN_TEAM)
    fpl_api._fetcher_for.cache_clear()

    fetchers = {
        id(get_fetcher()),
        id(get_fetcher(None)),
        id(get_fetcher(fpl_team_id=None)),
        id(get_fetcher(OWN_TEAM)),
    }
    assert len(fetchers) == 1
    assert id(get_fetcher(OTHER_TEAM)) not in fetchers

    fpl_api._fetcher_for.cache_clear()
