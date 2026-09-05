"""Scrape player injury, suspension and availability data from TransferMarkt."""

import contextlib
import os
import re
import time
from io import StringIO
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from airsenal.core.console import track
from airsenal.core.data_files import data_dir
from airsenal.core.logging import get_logger
from airsenal.game.season import (
    CURRENT_SEASON,
    get_next_season,
    season_str_to_year,
)
from airsenal.remote.errors import (
    RemoteConnectionError,
    RemoteError,
    RemoteHTTPError,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)

TRANSFERMARKT_URL = "https://www.transfermarkt.co.uk"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)"
        "Chrome/47.0.2526.106 Safari/537.36"
    )
}

TIMEOUT_SECONDS = 30.0
RETRIES = 3
RETRY_BACKOFF_SECONDS = 5.0
RETRY_STATUS_CODES = (429, 500, 502, 503, 504)
REQUEST_DELAY_SECONDS = 1.0

ABSENCE_COLUMNS = ("season", "details", "from", "until", "days", "games", "reason")


def _fetch_once(url: str, timeout: float) -> requests.Response:
    """
    One request for a Transfermarkt page.

    Checks the status, so an error page is a `RemoteError` and a timeout is a
    `RemoteConnectionError`.
    """
    time.sleep(REQUEST_DELAY_SECONDS)
    try:
        page = requests.get(url, headers=HEADERS, timeout=timeout)
    except requests.exceptions.RequestException as e:
        msg = f"Unable to reach Transfermarkt at {url}"
        raise RemoteConnectionError(msg) from e
    try:
        page.raise_for_status()
    except requests.exceptions.HTTPError as e:
        msg = f"Transfermarkt returned {page.status_code} for {url}"
        raise RemoteHTTPError(msg, page.status_code) from e
    return page


def _get(url: str, timeout: float = TIMEOUT_SECONDS) -> requests.Response:
    """
    Fetch a Transfermarkt page, retrying while the failure looks temporary.

    A timeout or a 429 part way through a scrape is the site asking us to slow
    down, not a page that cannot be read, so those are waited out. A 404 is not
    retried: it is the answer.
    """
    for attempt in range(RETRIES):
        try:
            return _fetch_once(url, timeout)
        except RemoteHTTPError as e:
            if e.status_code not in RETRY_STATUS_CODES or attempt == RETRIES - 1:
                raise
        except RemoteConnectionError:
            if attempt == RETRIES - 1:
                raise
        wait = RETRY_BACKOFF_SECONDS * (attempt + 1)
        logger.warning("Retrying %s in %ss", url, wait)
        time.sleep(wait)
    # Unreachable: the last attempt either returns or re-raises.
    msg = f"Gave up fetching {url}"
    raise RemoteConnectionError(msg)


class Team(NamedTuple):
    """A Premier League club in one season, as TransferMarkt identifies it."""

    name: str
    url: str
    club_id: str
    slug_words: set[str]


def _club_id(href: str) -> str:
    """
    TransferMarkt's numeric club id from any of its club URLs, or "".

    The id is the only part of a club URL that is stable across the site: the
    squad pages call Manchester City "manchester-city" and the transfer history
    calls it "man-city", but both spell it `/verein/281/`.
    """
    match = re.search(r"/verein/(\d+)", href)
    return match.group(1) if match else ""


def _club_slug(href: str) -> str:
    """The name part of a TransferMarkt club URL, e.g. "arsenal-u21"."""
    parts = href.split("/")
    return parts[1] if len(parts) > 1 else ""


def get_teams_for_season(season: int) -> list[Team]:
    """
    Get the names and TransferMarkt URLs for all the teams in this season.

    Args:
        season: The year the season started, not the usual four-digit string.
    """
    logger.debug("getting teams for %s/%s season", str(season)[2:], str(season + 1)[2:])

    # get list of teams
    url_season = (
        f"{TRANSFERMARKT_URL}/premier-league/startseite/wettbewerb/GB1/plus/"
        f"?saison_id={season}"
    )
    page = _get(url_season)
    soup = BeautifulSoup(page.content, features="lxml")
    rows = soup.find_all("td", {"class": "zentriert no-border-rechts"})

    return [
        Team(
            name=str(r.a.get("title")),
            url=str(r.a.get("href")),
            club_id=_club_id(str(r.a.get("href"))),
            slug_words=set(_club_slug(str(r.a.get("href"))).split("-")),
        )
        for r in rows
        if r.a is not None
    ][:20]


def get_team_players(team_season_url: str) -> list[tuple[str, str]]:
    """
    Get all the players in a team's squad for a season, as name and relative URL.

    Scrapes pages like
    https://www.transfermarkt.co.uk/manchester-city/startseite/verein/281/saison_id/2021
    """
    page = _get(f"{TRANSFERMARKT_URL}{team_season_url}")
    team_soup = BeautifulSoup(page.content, features="lxml")
    player_rows = team_soup.find_all("td", {"class": "posrela"})

    player_names_urls = []
    for r in player_rows:
        last_a_tag = r.find_all("a")[-1]
        name = str(last_a_tag.contents[0]).strip()
        url = str(last_a_tag.get("href"))
        player_names_urls.append((name, url))

    return player_names_urls


def _leading_int(column: pd.Series) -> pd.Series:
    """
    The number at the start of each value, as a nullable integer.

    TransferMarkt writes a duration as a number and a unit - "8 days" on the
    English pages, "8 Tage" on some of them, "? days" when it does not know. Take the
    leading digits and discard the rest, or NA for a value with no number in it at all.
    """
    return (
        column.astype("string")
        .str.extract(r"(-?\d+)", expand=False)
        .astype("float")
        .astype("Int32")
    )


def tidy_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names, data types, and missing data for injury/suspension data."""
    df.columns = df.columns.str.lower()
    df = df.rename(columns={"games missed": "games"})

    with contextlib.suppress(AttributeError):
        # can fail with AttributeError if all values are missing
        df["season"] = df["season"].str.replace("/", "")
    df = df.replace({"-": np.nan, "?": np.nan})
    df["from"] = pd.to_datetime(df["from"], format="%d/%m/%Y", errors="coerce")
    df["until"] = pd.to_datetime(df["until"], format="%d/%m/%Y", errors="coerce")

    df["days"] = _leading_int(df["days"])
    df["games"] = _leading_int(df["games"])

    return df.convert_dtypes()


def filter_season(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Extract the rows for one season, matching on `df`'s "1819"-format column."""
    return df[df["season"] == season]


def empty_absences() -> pd.DataFrame:
    """No absences of one kind, shaped so it can be concatenated with the rest."""
    return pd.DataFrame(columns=list(ABSENCE_COLUMNS))


def _read_absence_table(html: str, heading: str) -> pd.DataFrame | None:
    """
    The table on an absence page whose header holds `heading`, or None.

    None means the page has no such table, which is the ordinary state of a
    player who has never been injured or suspended.
    """
    try:
        return pd.read_html(StringIO(html), match=heading)[0]
    except ValueError:
        return None


def get_player_injuries(player_profile_url: str) -> pd.DataFrame:
    """
    Get a player's injury history: type, date, length and games missed.

    Scrapes pages like
    https://www.transfermarkt.co.uk/kyle-walker/verletzungen/spieler/95424
    """
    logger.debug("getting player injuries for %s", player_profile_url)

    page = _get(
        f"{TRANSFERMARKT_URL}{player_profile_url.replace('/profil/', '/verletzungen/')}"
    )
    logger.debug("processing player injuries for %s", player_profile_url)

    injuries = _read_absence_table(page.text, "Injury")
    if injuries is None:
        return empty_absences()
    injuries = injuries.rename(columns={"Injury": "Details"})
    injuries["Reason"] = "injury"

    return tidy_df(injuries)


def get_reason(details: str) -> str:
    """The category of a non-injury absence, e.g. suspension."""
    return "suspension" if "suspen" in details.lower() else "absence"


def get_player_suspensions(
    player_profile_url: str,
) -> pd.DataFrame:
    """
    Get a player's non-injury unavailability.

    Reason, competition, date, length and games missed. Scrapes pages like
    https://www.transfermarkt.co.uk/kyle-walker/ausfaelle/spieler/95424
    """
    logger.debug("getting player suspensions for %s", player_profile_url)

    p = _get(
        f"{TRANSFERMARKT_URL}{player_profile_url.replace('/profil/', '/ausfaelle/')}"
    )

    logger.debug("processing player suspensions for %s", player_profile_url)

    suspended = _read_absence_table(p.text, "Absence/Suspension")
    if suspended is None:
        return empty_absences()
    player_soup = BeautifulSoup(p.content, features="lxml")

    table = player_soup.find_all("table")[0]
    rows = table.find_all("tr")[1:]  # skip header row
    # a Tag attribute can be a list when the HTML repeats it; the column wants
    # one string per row, so take the first value
    competitions: list[str] = []
    for row in rows:
        try:
            title = row.find_all("img")[0].get("title")
        except IndexError:
            title = None
        if isinstance(title, list):
            title = title[0] if title else None
        competitions.append(str(title) if title is not None else "")
    suspended["Competition"] = competitions
    suspended = suspended.rename(columns={"Absence/Suspension": "Details"})
    suspended["Reason"] = [get_reason(detail) for detail in suspended["Details"]]

    return tidy_df(suspended)


def get_players_for_season(season: int) -> list[tuple[str, str]]:
    """
    Get every player at a Premier League club in a season, as name and relative URL.

    Args:
        season: The year the season started, not the usual four-digit string.
    """
    teams = get_teams_for_season(season)
    players = set()
    for _, team_url, __, ___ in track(teams):
        players.update(get_team_players(team_url))

    return list(players)


def remove_youth_or_reserve_suffix(team_name: str) -> str:
    """
    Strip a youth or reserve suffix from a TransferMarkt team name.

    So "arsenal-fc-u21" becomes "arsenal-fc". Being in the youth team does not in
    practice stop a player appearing in a Premier League game, so keeping the
    suffix would mark them unavailable when they are not.
    """
    suffix_to_remove = [
        "-youth",
        "-b",
        "u16",
        "-u17",
        "-u18",
        "u19",
        "-u20",
        "-u21",
        "-u23",
    ]
    for suffix in suffix_to_remove:
        if team_name.endswith(suffix):
            team_name = team_name[: -len(suffix)]
    return team_name


TRANSFER_COLUMNS = (
    "season",
    "date",
    "old",
    "new",
    "old_TM",
    "new_TM",
    "old_link",
    "new_link",
)


def _player_id(player_profile_url: str) -> str:
    """TransferMarkt's numeric player id from a profile URL, or ""."""
    match = re.search(r"/spieler/(\d+)", player_profile_url)
    return match.group(1) if match else ""


def get_player_transfers(
    player_profile_url: str,
) -> pd.DataFrame:
    """
    Get a player's transfer history: season, date, old team and new team.

    Read from the endpoint the transfers page itself calls, e.g.
    https://www.transfermarkt.co.uk/ceapi/transferHistory/list/95424.

    Returns:
        One row per completed transfer, oldest first, with the columns
        `TRANSFER_COLUMNS`. `old_TM` and `new_TM` are club ids rather than name
        slugs: this endpoint abbreviates club names ("man-city" for the squad
        pages' "manchester-city"), so the slugs cannot be compared across the two.
        Empty if the player has never transferred.
    """
    logger.debug("getting player transfer history for %s", player_profile_url)

    player_id = _player_id(player_profile_url)
    if not player_id:
        msg = f"No player id in Transfermarkt URL {player_profile_url}"
        raise RemoteError(msg)
    page = _get(f"{TRANSFERMARKT_URL}/ceapi/transferHistory/list/{player_id}")

    logger.debug("processing player transfer history for %s", player_profile_url)

    rows = []
    for transfer in page.json().get("transfers", []):
        if transfer.get("upcoming") or transfer.get("futureTransfer"):
            # Announced but not yet happened, so it says nothing about who the
            # player was available for in a season we are scraping.
            continue
        old, new = transfer.get("from", {}), transfer.get("to", {})
        old_href, new_href = old.get("href", ""), new.get("href", "")
        if not _club_id(old_href) or not _club_id(new_href):
            logger.warning(
                "Skipping a transfer for %s with no club id: %r -> %r",
                player_profile_url,
                old_href,
                new_href,
            )
            continue
        rows.append(
            [
                transfer.get("season", ""),
                transfer.get("dateUnformatted", ""),
                old.get("clubName", ""),
                new.get("clubName", ""),
                _club_id(old_href),
                _club_id(new_href),
                old_href,
                new_href,
            ]
        )

    raw = pd.DataFrame(rows, columns=list(TRANSFER_COLUMNS))
    raw["date"] = pd.to_datetime(raw["date"], format="%Y-%m-%d", errors="coerce")

    # The endpoint lists the most recent transfer first; callers walk forwards
    # through a career.
    return raw.iloc[::-1].reset_index(drop=True)


def get_start_end_dates_of_season(season: str) -> list[pd.Timestamp]:
    """
    Rough start and end dates for a season.

    19/20 and 20/21 were shorter and longer than usual, and are special-cased.
    """
    start_year = int(f"20{season[:2]}")
    end_year = int(f"20{season[2:]}")
    if season == "1920":
        # regular start, late end to season
        return [pd.Timestamp(2019, 7, 1), pd.Timestamp(2020, 7, 31)]
    if season == "2021":
        # late start to season, regular end
        return [pd.Timestamp(2020, 8, 1), pd.Timestamp(2021, 6, 30)]
    # regular season
    return [pd.Timestamp(start_year, 7, 1), pd.Timestamp(end_year, 6, 30)]


def played_in_premier_league(club_id: str, club_url: str, teams: list[Team]) -> bool:
    """
    Whether a club a player moved to was one of `teams`.

    Matched on TransferMarkt's numeric club id.

    A youth or reserve side has an id of its own, so it never matches a first
    team - but a player in the under-21s is still available for Premier League
    selection, which is what `remove_youth_or_reserve_suffix` is for. For those,
    and only those, the stripped name is compared against the first team names as
    well; the transfer history abbreviates some of them ("man-city"), so this
    recognises a youth side of Arsenal but not one of Manchester City.
    """
    if any(club_id == team.club_id for team in teams):
        return True
    slug = _club_slug(club_url)
    senior = remove_youth_or_reserve_suffix(slug)
    if senior == slug:
        return False
    words = set(senior.split("-"))
    return any(words <= team.slug_words for team in teams)


# The earliest season a "1819"-style string can mean, because
# `season_str_to_year` reads one as 20xx. A career that starts before it cannot
# be walked, only joined part way through.
EARLIEST_SEASON = "0001"


def first_season_to_walk(first_transfer_season: str, end_season: str) -> str:
    """
    The season to start building a team history from.

    Usually the season of the player's first transfer. A career that began last
    century cannot be: "9899" reads as 2098/99, and stepping forwards from it
    runs 99 into "100" rather than wrapping. Those start at `EARLIEST_SEASON`
    instead.
    """
    if (
        len(first_transfer_season) != 4
        or not first_transfer_season.isdigit()
        or int(first_transfer_season[:2]) > int(end_season[:2])
    ):
        return EARLIEST_SEASON
    return first_transfer_season


def _nothing_known(
    season: str, start: pd.Timestamp, until: pd.Timestamp
) -> dict[str, object]:
    """
    A team history entry for a stretch we have no club for.

    Not marked as Premier League, so it reads as unavailable: a player we cannot
    place was not playing in the league as far as we know.
    """
    return {
        "season": season,
        "team": "Unknown",
        "team_tm": "",
        "team_url": "",
        "from": start,
        "until": until,
        "pl": False,
    }


def _carried_forward(
    previous: dict[str, object],
    season: str,
    start: pd.Timestamp,
    until: pd.Timestamp,
    pl_teams: list[Team],
) -> dict[str, object]:
    """
    A team history entry for a season the player stayed where they were.

    The Premier League flag is recomputed rather than copied: a club can be
    relegated under a player who never moved.
    """
    return {
        "season": season,
        "team": previous["team"],
        "team_tm": previous["team_tm"],
        "team_url": previous["team_url"],
        "from": start,
        "until": until,
        "pl": played_in_premier_league(
            str(previous["team_tm"]), str(previous["team_url"]), pl_teams
        ),
    }


def get_player_team_history(
    df: pd.DataFrame,
    pl_teams_in_season: dict[str, list[Team]] | None = None,
    end_season: str = CURRENT_SEASON,
) -> pd.DataFrame:
    """
    Turn a player's transfer data into a team history.

    Columns are season, team, from, until, and whether that team was in the
    Premier League.

    Args:
        df: Transfer data for one player, from `get_player_transfers`.
        pl_teams_in_season: Season in "1819" format to the teams that played in it,
            from `get_teams_for_season`. This is what decides the last column.
        end_season: The last season to produce history for.
    """
    if pl_teams_in_season is None:
        pl_teams_in_season = {}
    rows: list[dict[str, object]] = []
    current_season = first_season_to_walk(
        "".join(df.iloc[0]["season"].split("/")), end_season
    )
    while season_str_to_year(current_season) <= season_str_to_year(end_season):
        season_df = df[df["season"] == f"{current_season[:2]}/{current_season[2:]}"]
        start, end = get_start_end_dates_of_season(current_season)
        if current_season not in pl_teams_in_season:
            pl_teams_in_season[current_season] = get_teams_for_season(
                season_str_to_year(current_season)
            )
        pl_teams = pl_teams_in_season[current_season]

        if len(season_df) == 0:
            # if no transfer data, player continued at current club that year -
            # unless the walk has not reached their first transfer yet, in which
            # case there is no club to continue at and nothing is known.
            rows.append(
                _carried_forward(rows[-1], current_season, start, end, pl_teams)
                if rows
                else _nothing_known(current_season, start, end)
            )

        for i in range(len(season_df)):
            transfer_date = season_df.iloc[i]["date"]
            if i == 0 and transfer_date > start:
                day_before = transfer_date - pd.DateOffset(days=1)
                if len(rows) == 0:
                    # first team added and no data for before time
                    rows.append(_nothing_known(current_season, start, day_before))
                else:
                    # started the season at same club as previous entry
                    rows.append(
                        _carried_forward(
                            rows[-1], current_season, start, day_before, pl_teams
                        )
                    )

            # decide how long this player was at the club that season
            # by default, player will be until the end of the year unless they moved
            to_entry = end
            if i != len(season_df) - 1:
                # player left before the end of the year, so end this entry at this time
                to_date = season_df.iloc[i + 1]["date"] - pd.DateOffset(days=1)
                if to_date < end:
                    to_entry = to_date

            rows.append(
                {
                    "season": current_season,
                    "team": season_df.iloc[i]["new"],
                    "team_tm": season_df.iloc[i]["new_TM"],
                    "team_url": season_df.iloc[i]["new_link"],
                    "from": season_df.iloc[i]["date"],
                    "until": to_entry,
                    "pl": played_in_premier_league(
                        season_df.iloc[i]["new_TM"],
                        season_df.iloc[i]["new_link"],
                        pl_teams,
                    ),
                }
            )

        current_season = get_next_season(current_season)

    return pd.DataFrame(
        rows, columns=["season", "team", "team_tm", "team_url", "from", "until", "pl"]
    )


def get_player_transfer_unavailability(
    player_profile_url: str,
    pl_teams_in_season: dict[str, list[Team]] | None = None,
    end_season: str = CURRENT_SEASON,
) -> pd.DataFrame:
    """
    Spells a player was unavailable because they were at a non-league club.

    Columns are season, details, reason, from, until, days and games missed.

    Args:
        pl_teams_in_season: Season in "1819" format to the teams that played in it,
            from `get_teams_for_season`.
        end_season: The last season to produce unavailability for.
    """
    if pl_teams_in_season is None:
        pl_teams_in_season = {}
    logger.debug("getting player transfer unavailability for %s", player_profile_url)

    transfers = get_player_transfers(player_profile_url)
    if transfers.empty:
        # A player who has never moved has always been where they are now, and
        # `get_player_team_history` has no first season to start from.
        return empty_absences()

    transfer_history = get_player_team_history(
        df=transfers,
        pl_teams_in_season=pl_teams_in_season,
        end_season=end_season,
    )

    logger.debug("processing player transfer unavailability for %s", player_profile_url)

    unavailability = transfer_history[~transfer_history["pl"]]

    return pd.DataFrame(
        {
            "season": unavailability["season"],
            "details": "Transferred to " + unavailability["team"].astype(str),
            "reason": "Transfer",
            "from": unavailability["from"],
            "until": unavailability["until"],
            "days": pd.NA,
            "games": pd.NA,
        }
    )


def premier_league_absences(
    suspensions: pd.DataFrame, teams: list[Team]
) -> pd.DataFrame:
    """
    Absences from the non-injury table that cost the player league matches.

    Transfermarkt tags each row with what the games were missed from, so a
    Champions League ineligibility is dropped here.

    A national team call-up is tagged with the player's own club rather than with
    a competition - that cell carries a club crest where a suspension carries a
    competition logo - so the season's club names are kept alongside "Premier
    League" itself.
    """
    if "competition" not in suspensions.columns:
        return suspensions
    keep = {"Premier League", *(team.name for team in teams)}
    league_only = suspensions[suspensions["competition"].isin(keep)]
    return league_only.drop("competition", axis=1)


def get_season_absences(
    season: str, pl_teams_in_season: dict[str, list[Team]] | None = None
) -> pd.DataFrame:
    """
    Every absence for every player in a season, in one data frame.

    Injuries, suspensions, international call-ups and other unavailability from
    Transfermarkt's own two tables, plus time spent at a non-Premier League club.
    A player whose page cannot be scraped is counted and skipped.
    """
    if pl_teams_in_season is None:
        pl_teams_in_season = {}
    year = season_str_to_year(season)
    if season not in pl_teams_in_season:
        pl_teams_in_season[season] = get_teams_for_season(year)
    logger.info("Finding players...")

    players = get_players_for_season(year)
    absences = []
    failures: dict[str, int] = {}
    logger.info("Querying injuries, suspensions and transfers...")

    scrapers: tuple[tuple[str, Callable[[str], pd.DataFrame]], ...] = (
        ("injuries", lambda url: get_player_injuries(player_profile_url=url)),
        (
            "suspensions",
            lambda url: premier_league_absences(
                get_player_suspensions(player_profile_url=url),
                pl_teams_in_season.get(season, []),
            ),
        ),
        (
            "transfers",
            lambda url: get_player_transfer_unavailability(
                player_profile_url=url,
                pl_teams_in_season=pl_teams_in_season,
                end_season=season,
            ),
        ),
    )

    for player_name, player_url in track(players):
        for kind, scrape in scrapers:
            try:
                found = scrape(player_url)
            except (ValueError, IndexError, KeyError, RemoteError):
                failures[kind] = failures.get(kind, 0) + 1
                logger.warning(
                    "Could not read %s for %s", kind, player_url, exc_info=True
                )
                continue
            if found.empty:
                continue
            found["player"] = player_name
            found["url"] = player_url
            absences.append(found)

    for kind, _ in scrapers:
        if failures.get(kind):
            logger.error(
                "Failed to read %s for %s of %s players - Transfermarkt may have "
                "changed that page",
                kind,
                failures[kind],
                len(players),
            )

    if not absences:
        msg = f"Found no absences at all for {season}"
        raise RemoteError(msg)

    return filter_season(pd.concat(absences), season)


def scrape_transfermarkt(seasons: list[str]) -> None:
    """Get all player injury and suspension data for several seasons."""
    repo_home = data_dir()

    # get the teams that played in each season we want to scrape
    pl_teams = {}
    for s in seasons:
        pl_teams[s] = get_teams_for_season(season_str_to_year(s))

    for season in track(seasons):
        logger.info("-" * 50)
        logger.info("Season: %s", season)

        absences = get_season_absences(season, pl_teams_in_season=pl_teams)
        logger.info(
            "%s absences for %s: %s",
            len(absences),
            season,
            absences["reason"].value_counts().to_dict(),
        )
        absences.to_csv(os.path.join(repo_home, f"absences_{season}.csv"), index=False)
