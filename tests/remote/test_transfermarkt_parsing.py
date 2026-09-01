"""
Parsing the Transfermarkt pages, against recorded copies of them.

`test_transfermarkt.py` covers fetching; this covers what happens to what comes
back. Every function here reads a file from `transfermarkt_pages/` in place of a
request, so a change to Transfermarkt's markup shows up as a failure here rather
than as a season of absence data that quietly holds nothing but injuries.
"""

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from airsenal.remote import transfermarkt
from airsenal.remote.errors import RemoteError
from airsenal.remote.transfermarkt import (
    Team,
    first_season_to_walk,
    get_player_injuries,
    get_player_suspensions,
    get_player_transfers,
    get_teams_for_season,
    played_in_premier_league,
    premier_league_absences,
    tidy_df,
)

PAGES = Path(__file__).parent / "transfermarkt_pages"
PLAYER = "/kyle-walker/profil/spieler/95424"


def page(name: str) -> bytes:
    with gzip.open(PAGES / f"{name}.gz", "rb") as infile:
        return infile.read()


class RecordedResponse:
    """Enough of a `requests.Response` for the parsers to read."""

    def __init__(self, content: bytes) -> None:
        self.content = content

    @property
    def text(self) -> str:
        return self.content.decode("utf-8")

    def json(self):
        return json.loads(self.text)


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    """Recorded pages do not need pacing."""
    monkeypatch.setattr(transfermarkt, "REQUEST_DELAY_SECONDS", 0)


@pytest.fixture
def recorded(monkeypatch):
    """Serve each Transfermarkt URL from the recorded copy of that page."""
    by_url_fragment = {
        "wettbewerb/GB1": "premier_league_2025.html",
        "/verletzungen/": "player_verletzungen.html",
        "/ausfaelle/": "player_ausfaelle.html",
        "/ceapi/transferHistory/": "player_transfers.json",
    }

    def fake_get(url, timeout=None):
        for fragment, name in by_url_fragment.items():
            if fragment in url:
                return RecordedResponse(page(name))
        msg = f"No recorded page for {url}"
        raise AssertionError(msg)

    monkeypatch.setattr(transfermarkt, "_get", fake_get)


def test_the_teams_carry_the_club_id_the_rest_of_the_site_uses(recorded):
    teams = get_teams_for_season(2025)

    assert len(teams) == 20
    by_name = {t.name: t for t in teams}
    assert by_name["Manchester City"].club_id == "281"
    assert by_name["Arsenal FC"].club_id == "11"
    assert all(t.club_id for t in teams)


def test_injuries_are_read_with_their_dates_and_games_missed(recorded):
    injuries = get_player_injuries(PLAYER)

    assert set(injuries["reason"]) == {"injury"}
    broken_arm = injuries[injuries["details"] == "Broken arm"].iloc[0]
    assert broken_arm["season"] == "2425"
    assert broken_arm["from"] == pd.Timestamp(2025, 4, 8)
    assert broken_arm["until"] == pd.Timestamp(2025, 4, 26)
    assert broken_arm["days"] == 19
    assert broken_arm["games"] == 3


def test_suspensions_and_other_absences_are_read(recorded):
    """
    The regression that emptied `absences_2425.csv` and `absences_2526.csv`.

    Transfermarkt writes this table's duration in the same language as the rest
    of the page - "8 days" - where it used to write "8 Tage". Stripping the
    German unit left "8 days" to be read as a number, which raised, which the
    scrape caught and skipped, per player, for every player.
    """
    absences = get_player_suspensions(PLAYER)

    assert len(absences) > 0
    suspension = absences[absences["season"] == "2526"].iloc[0]
    assert suspension["details"] == "Yellow card suspension"
    assert suspension["reason"] == "suspension"
    assert suspension["competition"] == "Premier League"
    assert suspension["from"] == pd.Timestamp(2025, 12, 7)
    assert suspension["days"] == 8

    # Anything that is not a suspension - an international call-up, a leave of
    # absence, being ineligible for a competition - is reason "absence".
    assert set(absences["reason"]) == {"suspension", "absence"}
    assert "No eligibility" in set(absences["details"])


def test_an_absence_keeps_the_competition_it_cost_games_in(recorded):
    """`get_season_absences` filters on this to keep league absences only."""
    absences = get_player_suspensions(PLAYER)

    assert "UEFA Champions League" in set(absences["competition"])
    assert "Premier League" in set(absences["competition"])


def test_a_page_with_no_absence_table_is_empty_rather_than_an_error(monkeypatch):
    """
    A player who has never been injured is the ordinary case, not a failure.

    Telling the two apart is what lets `get_season_absences` complain about a
    page it could not parse without complaining about every clean record.
    """

    def no_table(url, timeout=None):
        return RecordedResponse(b"<html><body>Nothing</body></html>")

    monkeypatch.setattr(transfermarkt, "_get", no_table)

    assert get_player_injuries(PLAYER).empty
    assert get_player_suspensions(PLAYER).empty


def test_transfers_are_read_from_the_endpoint_the_page_calls(recorded):
    """
    The transfer table is no longer in the HTML.

    `/kyle-walker/transfers/spieler/95424` renders a `<tm-player-transfer-history>`
    element that fetches the data, so the grid classes the old scrape looked for
    are not there and every player raised.
    """
    transfers = get_player_transfers(PLAYER)

    assert len(transfers) > 0
    # Oldest first: callers walk forwards through a career from the first row.
    assert transfers.iloc[0]["date"] < transfers.iloc[-1]["date"]

    to_burnley = transfers[transfers["new"] == "Burnley"].iloc[0]
    assert to_burnley["season"] == "25/26"
    assert to_burnley["date"] == pd.Timestamp(2025, 7, 5)
    assert to_burnley["old_TM"] == "281"
    assert to_burnley["new_TM"] == "1132"


def test_the_most_recent_transfer_is_included(recorded):
    """The old scrape skipped index 0, which was a header row it no longer has."""
    transfers = get_player_transfers(PLAYER)

    assert transfers.iloc[-1]["new"] == "Burnley"


@pytest.mark.parametrize(
    ("club_id", "club_url", "expected"),
    [
        # The transfer history abbreviates club names, so only the id matches.
        ("281", "/man-city/transfers/verein/281/saison_id/2025", True),
        ("11", "/arsenal/transfers/verein/11/saison_id/2025", True),
        # A youth side has an id of its own, but its players are selectable.
        ("9249", "/arsenal-u21/transfers/verein/9249/saison_id/2025", True),
        ("50677", "/chelsea-youth/transfers/verein/50677/saison_id/2025", True),
        # Genuinely elsewhere.
        ("5", "/ac-milan/transfers/verein/5/saison_id/2024", False),
        ("24219", "/stoke-u21/transfers/verein/24219/saison_id/2025", False),
        ("", "", False),
    ],
)
def test_which_clubs_count_as_premier_league(recorded, club_id, club_url, expected):
    teams = get_teams_for_season(2025)

    assert played_in_premier_league(club_id, club_url, teams) is expected


def test_a_club_is_matched_by_id_not_by_name():
    """
    A club is matched by id, not by name.

    Manchester City is "manchester-city" on a squad page and "man-city" in a
    transfer, so comparing names marks a player at a club they are at as away.
    """
    teams = [Team("Manchester City", "/manchester-city/x/verein/281/y", "281", set())]

    assert played_in_premier_league("281", "/man-city/x/verein/281/y", teams) is True


@pytest.mark.parametrize("duration", ["8 days", "8 Tage", "8"])
def test_a_duration_is_read_whatever_unit_it_is_written_in(duration):
    """The unit has changed under us once; reading past it means it cannot again."""
    df = tidy_df(
        pd.DataFrame(
            {
                "Season": ["25/26"],
                "Details": ["Knock"],
                "from": ["07/12/2025"],
                "until": ["14/12/2025"],
                "Days": [duration],
                "Games missed": ["1"],
                "Reason": ["injury"],
            }
        )
    )

    assert df.iloc[0]["days"] == 8
    assert df.iloc[0]["games"] == 1


def test_an_unknown_duration_is_missing_rather_than_zero():
    df = tidy_df(
        pd.DataFrame(
            {
                "Season": ["25/26"],
                "Details": ["Knock"],
                "from": ["07/12/2025"],
                "until": [None],
                "Days": ["? days"],
                "Games missed": ["-"],
                "Reason": ["injury"],
            }
        )
    )

    assert pd.isna(df.iloc[0]["days"])
    assert pd.isna(df.iloc[0]["games"])


def test_all_three_kinds_of_absence_end_up_in_one_season(monkeypatch):
    """
    A season's file should hold injuries, suspensions and transfers.

    The three are scraped from three different places, and losing one of them is
    invisible in the result: `absences_2425.csv` looks like a well formed file
    with 380 rows in it, none of which are suspensions.
    """
    monkeypatch.setattr(
        transfermarkt,
        "get_players_for_season",
        lambda season: [("Kyle Walker", PLAYER)],  # noqa: ARG005
    )

    def one_row(reason):
        return pd.DataFrame(
            {
                "season": ["2526"],
                "details": [reason],
                "from": [pd.Timestamp(2025, 12, 7)],
                "until": [pd.Timestamp(2025, 12, 14)],
                "days": [8],
                "games": [1],
                "reason": [reason],
            }
        )

    monkeypatch.setattr(
        transfermarkt,
        "get_player_injuries",
        lambda player_profile_url: one_row("injury"),  # noqa: ARG005
    )
    monkeypatch.setattr(
        transfermarkt,
        "get_player_suspensions",
        lambda player_profile_url: one_row("suspension"),  # noqa: ARG005
    )
    monkeypatch.setattr(
        transfermarkt,
        "get_player_transfer_unavailability",
        lambda player_profile_url, pl_teams_in_season, end_season: one_row(  # noqa: ARG005
            "Transfer"
        ),
    )

    absences = transfermarkt.get_season_absences("2526", {"2526": []})

    assert set(absences["reason"]) == {"injury", "suspension", "Transfer"}
    assert set(absences["player"]) == {"Kyle Walker"}


def test_a_page_that_cannot_be_parsed_is_reported(monkeypatch, caplog):
    """
    The failure mode that hid the two breakages this module was fixed for.

    Each kind is scraped in a try/except so one bad page does not stop a scrape
    of 600 players - which means the count has to be logged, or a page that stops
    parsing for every player is silent.
    """
    monkeypatch.setattr(
        transfermarkt,
        "get_players_for_season",
        lambda season: [("Kyle Walker", PLAYER)],  # noqa: ARG005
    )
    monkeypatch.setattr(
        transfermarkt,
        "get_player_injuries",
        lambda player_profile_url: pd.DataFrame(  # noqa: ARG005
            {
                "season": ["2526"],
                "details": ["Knock"],
                "from": [pd.Timestamp(2025, 12, 7)],
                "until": [pd.Timestamp(2025, 12, 14)],
                "days": [8],
                "games": [1],
                "reason": ["injury"],
            }
        ),
    )

    def cannot_parse(*args, **kwargs):
        msg = "could not convert string to float: '8 days'"
        raise ValueError(msg)

    monkeypatch.setattr(transfermarkt, "get_player_suspensions", cannot_parse)
    monkeypatch.setattr(
        transfermarkt, "get_player_transfer_unavailability", cannot_parse
    )

    with caplog.at_level("ERROR"):
        absences = transfermarkt.get_season_absences("2526", {"2526": []})

    assert set(absences["reason"]) == {"injury"}
    assert "Failed to read suspensions for 1 of 1 players" in caplog.text
    assert "Failed to read transfers for 1 of 1 players" in caplog.text


def test_a_season_with_nothing_at_all_is_an_error(monkeypatch):
    """Writing an empty csv over a good one is worse than failing the scrape."""
    monkeypatch.setattr(
        transfermarkt,
        "get_players_for_season",
        lambda season: [("Kyle Walker", PLAYER)],  # noqa: ARG005
    )
    for name in (
        "get_player_injuries",
        "get_player_suspensions",
        "get_player_transfer_unavailability",
    ):
        monkeypatch.setattr(
            transfermarkt, name, lambda *args, **kwargs: transfermarkt.empty_absences()
        )

    with pytest.raises(RemoteError, match="no absences at all"):
        transfermarkt.get_season_absences("2526", {"2526": []})


@pytest.mark.parametrize(
    ("first_transfer_season", "expected"),
    [
        # An ordinary career, walked from where it started.
        ("0708", "0708"),
        ("2425", "2425"),
        # Last century. "9899" reads as 2098/99, and stepping forwards from it
        # runs 99 into "100": Gündoğan (98/99) and Heaton (97/98) both crashed
        # the walk with "year must be in 1..9999, not 20100".
        ("9899", "0001"),
        ("9700", "0001"),
        # Nothing usable at all.
        ("", "0001"),
        ("25/26", "0001"),
    ],
)
def test_where_a_team_history_starts(first_transfer_season, expected):
    assert first_season_to_walk(first_transfer_season, "2526") == expected


def test_a_career_that_began_last_century_is_walked_rather_than_crashing():
    """
    Two of 781 players lost their transfer data to this on the last scrape.

    The walk is bounded by `end_season` now, so a first transfer decades earlier
    costs a few wasted iterations rather than an unrepresentable season.
    """
    transfers = pd.DataFrame(
        {
            "season": ["98/99", "25/26"],
            "date": [pd.Timestamp(1998, 7, 1), pd.Timestamp(2025, 7, 5)],
            "old": ["Bochum", "Man City"],
            "new": ["Nürnberg", "Burnley"],
            "old_TM": ["80", "281"],
            "new_TM": ["4", "1132"],
            "old_link": ["/bochum/x/verein/80/y", "/man-city/x/verein/281/y"],
            "new_link": ["/nurnberg/x/verein/4/y", "/burnley/x/verein/1132/y"],
        }
    )
    teams = [
        Team("Burnley FC", "/burnley-fc/x/verein/1132/y", "1132", {"burnley", "fc"})
    ]

    history = transfermarkt.get_player_team_history(
        transfers,
        pl_teams_in_season=dict.fromkeys(
            [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2000, 2027)], teams
        ),
        end_season="2526",
    )

    assert len(history) > 0
    assert history["season"].iloc[0] == "0001"
    assert history["season"].iloc[-1] == "2526"
    # The last club of the walk is where the last transfer took them.
    assert history["team"].iloc[-1] == "Burnley"
    assert bool(history["pl"].iloc[-1]) is True


def _absence_rows(competitions):
    return pd.DataFrame(
        {
            "season": ["2526"] * len(competitions),
            "details": ["Called up to national team"] * len(competitions),
            "competition": competitions,
            "from": [pd.Timestamp(2025, 12, 15)] * len(competitions),
            "until": [pd.Timestamp(2026, 1, 12)] * len(competitions),
            "days": [29] * len(competitions),
            "games": [7] * len(competitions),
            "reason": ["absence"] * len(competitions),
        }
    )


def test_a_national_team_call_up_is_kept(recorded):
    """
    Transfermarkt tags a call-up with the player's club, not a competition.

    That cell carries a club crest where a suspension carries a competition
    logo, so filtering on "Premier League" alone threw away every Africa Cup of
    Nations absence - 44 players in 25/26, of which the FPL API caught all 44 and
    Transfermarkt appeared to catch 3.
    """
    teams = get_teams_for_season(2025)
    kept = premier_league_absences(
        _absence_rows(["Manchester City", "Brighton & Hove Albion", "Premier League"]),
        teams,
    )

    assert len(kept) == 3
    assert "competition" not in kept.columns


def test_another_competition_is_still_dropped(recorded):
    """An absence that only cost cup games did not cost league games."""
    teams = get_teams_for_season(2025)
    kept = premier_league_absences(
        _absence_rows(["UEFA Champions League", "EFL Cup", "Community Shield"]),
        teams,
    )

    assert len(kept) == 0


def test_a_call_up_from_a_club_outside_the_league_is_dropped(recorded):
    """It cost that player no Premier League matches."""
    teams = get_teams_for_season(2025)
    kept = premier_league_absences(_absence_rows(["AC Milan"]), teams)

    assert len(kept) == 0
