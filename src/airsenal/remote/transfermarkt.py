"""
Get player injury, suspension and availability data from TransferMarkt
"""

import contextlib
import os
from cmath import nan
from io import StringIO

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

logger = get_logger(__name__)

TRANSFERMARKT_URL = "https://www.transfermarkt.co.uk"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)"
        "Chrome/47.0.2526.106 Safari/537.36"
    )
}


def _get(url: str) -> requests.Response:
    """
    Fetch a Transfermarkt page, failing the way the rest of `remote` fails.

    Checks the status, so an error page is a `RemoteError` here rather than
    whatever the HTML parser makes of it further away.
    """
    try:
        page = requests.get(url, headers=HEADERS)
    except requests.exceptions.RequestException as e:
        msg = f"Unable to reach Transfermarkt at {url}"
        raise RemoteConnectionError(msg) from e
    try:
        page.raise_for_status()
    except requests.exceptions.HTTPError as e:
        msg = f"Transfermarkt returned {page.status_code} for {url}"
        raise RemoteHTTPError(msg, page.status_code) from e
    return page


def get_teams_for_season(season: int) -> list[tuple[str, str, str, set[str]]]:
    """
    Get the names and TransferMarkt URLs for all the teams in this season.

    Args:
        season: The year the season started, not the usual four-digit string.

    Returns:
        Per team: name, relative URL, TransferMarkt's identifier for it, and that
        identifier split on '-', which is what team-played-this-season checks
        match against.
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
        (
            str(r.a.get("title")),
            str(r.a.get("href")),
            str(r.a.get("href")).split("/")[1],
            set(str(r.a.get("href")).split("/")[1].split("-")),
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


def tidy_df(df: pd.DataFrame, days_name: str = "days") -> pd.DataFrame:
    """Clean column names, data types, and missing data for injury/suspension data."""
    df.columns = df.columns.str.lower()
    df = df.rename(columns={"games missed": "games"})

    with contextlib.suppress(AttributeError):
        # can fail with AttributeError if all values are missing
        df["season"] = df["season"].str.replace("/", "")
    df = df.replace({"-": np.nan, f"? {days_name}": np.nan, "?": np.nan})
    df["from"] = pd.to_datetime(df["from"], format="%d/%m/%Y", errors="coerce")
    df["until"] = pd.to_datetime(df["until"], format="%d/%m/%Y", errors="coerce")

    with contextlib.suppress(AttributeError):
        # can fail with AttributeError if all values are missing
        df["days"] = df["days"].str.replace(f" {days_name}", "")
    df["days"] = df["days"].astype("float").astype("Int32")
    df["games"] = df["games"].astype("float").astype("Int32")

    return df.convert_dtypes()


def filter_season(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Extract the rows for one season, matching on `df`'s "1819"-format column."""
    return df[df["season"] == season]


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

    injuries = pd.read_html(StringIO(str(page.content)), match="Injury")[0]
    injuries = injuries.rename(columns={"Injury": "Details"})
    injuries["Reason"] = "injury"

    return tidy_df(injuries, days_name="days")


def get_reason(details: str) -> str:
    """get suspension/absence reason category (not for injuries)"""
    return "suspension" if "suspen" in details.lower() else "absence"


def get_player_suspensions(
    player_profile_url: str,
) -> pd.DataFrame:
    """
    Get a player's non-injury unavailability: reason, competition, date, length
    and games missed.

    Scrapes pages like
    https://www.transfermarkt.co.uk/kyle-walker/ausfaelle/spieler/95424
    """
    logger.debug("getting player suspensions for %s", player_profile_url)

    p = _get(
        f"{TRANSFERMARKT_URL}{player_profile_url.replace('/profil/', '/ausfaelle/')}"
    )

    logger.debug("processing player suspensions for %s", player_profile_url)

    suspended = pd.read_html(StringIO(str(p.content)), match="Absence/Suspension")[0]
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

    return tidy_df(suspended, days_name="Tage")


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


def get_player_transfers(
    player_profile_url: str,
) -> pd.DataFrame:
    """
    Get a player's transfer history: season, date, old team and new team.

    Scrapes pages like
    https://www.transfermarkt.co.uk/kyle-walker/transfers/spieler/95424
    """
    logger.debug("getting player transfer history for %s", player_profile_url)

    page = _get(
        f"{TRANSFERMARKT_URL}{player_profile_url.replace('/profil/', '/transfers/')}"
    )

    logger.debug("processing player transfer history for %s", player_profile_url)

    soup = BeautifulSoup(page.text, "lxml")
    raw = pd.DataFrame()
    n_transfers = len(
        soup.find_all("div", class_="tm-player-transfer-history-grid__season")
    )

    for i in range(1, n_transfers):
        # obtain season and date of transfer
        season = " ".join(
            soup.find_all("div", class_="tm-player-transfer-history-grid__season")[i]
            .getText()
            .split()
        )
        date = " ".join(
            soup.find_all("div", class_="tm-player-transfer-history-grid__date")[i]
            .getText()
            .split()
        )
        # old club details
        old = soup.find_all("div", class_="tm-player-transfer-history-grid__old-club")[
            i
        ]
        old_club = " ".join(old.getText().split())
        if old.a is None:
            logger.warning("Old club link is missing")
            continue
        old_link = old.a.get("href")
        if not isinstance(old_link, str):
            logger.warning("Old club link returned type %s", type(old_link))
            continue
        old_tm_identifier = old_link.split("/")[1]
        # new club details
        new = soup.find_all("div", class_="tm-player-transfer-history-grid__new-club")[
            i
        ]
        new_club = " ".join(new.getText().split())
        if new.a is None:
            logger.warning("New club link is missing")
            continue
        new_link = new.a.get("href")
        if not isinstance(new_link, str):
            logger.warning("New club link returned type %s", type(new_link))
            continue
        new_tm_identifier = new_link.split("/")[1]
        raw = pd.concat(
            [
                raw,
                pd.DataFrame(
                    [
                        [
                            season,
                            date,
                            old_club,
                            new_club,
                            remove_youth_or_reserve_suffix(old_tm_identifier),
                            remove_youth_or_reserve_suffix(new_tm_identifier),
                            old_link,
                            new_link,
                        ]
                    ]
                ),
            ]
        )

    raw.columns = [
        "season",
        "date",
        "old",
        "new",
        "old_TM",
        "new_TM",
        "old_link",
        "new_link",
    ]
    raw["date"] = pd.to_datetime(raw["date"], format="%d/%m/%Y", errors="coerce")

    return raw.iloc[::-1]


def get_start_end_dates_of_season(season: str) -> list[pd.Timestamp]:
    """
    Obtains rough start and end dates for the season.
    Takes into account the shorter and longer seasons in 19/20 and 20/21.

    Here rather than in `season.py` because this is its only caller, and it is
    the only thing in that module that needs pandas: everything else there is
    string arithmetic over season names.
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


def get_player_team_history(
    df: pd.DataFrame,
    pl_teams_in_season: dict[str, list[set[str]]] | None = None,
    end_season: str = CURRENT_SEASON,
) -> pd.DataFrame:
    """
    Turn a player's transfer data into a team history: season, team, from, until,
    and whether that team was in the Premier League.

    Args:
        df: Transfer data for one player, from `get_player_transfers`.
        pl_teams_in_season: Season in "1819" format to the teams that played in it,
            from `get_teams_for_season`. This is what decides the last column.
        end_season: The last season to produce history for.
    """
    if pl_teams_in_season is None:
        pl_teams_in_season = {}
    teams_df = pd.DataFrame()
    current_season = "".join(df.iloc[0]["season"].split("/"))
    diff = int(current_season[:2]) - int(end_season[2:])
    for _ in range(abs(diff)):
        season_df = df[df["season"] == f"{current_season[:2]}/{current_season[2:]}"]
        start, end = get_start_end_dates_of_season(current_season)
        if current_season not in pl_teams_in_season:
            teams = get_teams_for_season(season_str_to_year(current_season))
            pl_teams_in_season[current_season] = [teams[i][3] for i in range(20)]

        if len(season_df) == 0:
            # if no transfer data, player continued at current club that year
            teams_df = pd.concat(
                [
                    teams_df,
                    pd.DataFrame(
                        [
                            [
                                current_season,
                                teams_df.iloc[-1][1],
                                teams_df.iloc[-1][2],
                                start,
                                end,
                                set(teams_df.iloc[-1][2].split("-"))
                                in pl_teams_in_season[current_season],
                            ]
                        ]
                    ),
                ]
            )

        for i in range(len(season_df)):
            transfer_date = season_df.iloc[i]["date"]
            if i == 0 and transfer_date > start:
                if len(teams_df) == 0:
                    # first team added and no data for before time
                    teams_df = pd.concat(
                        [
                            teams_df,
                            pd.DataFrame(
                                [
                                    [
                                        current_season,
                                        "Unknown",
                                        "unknown",
                                        start,
                                        transfer_date - pd.DateOffset(days=1),
                                        False,
                                    ]
                                ]
                            ),
                        ]
                    )
                else:
                    # started the season at same club as previous entry
                    teams_df = pd.concat(
                        [
                            teams_df,
                            pd.DataFrame(
                                [
                                    [
                                        current_season,
                                        teams_df.iloc[-1][1],
                                        teams_df.iloc[-1][2],
                                        start,
                                        transfer_date - pd.DateOffset(days=1),
                                        set(teams_df.iloc[-1][2].split("-"))
                                        in pl_teams_in_season[current_season],
                                    ]
                                ]
                            ),
                        ]
                    )

            # decide how long this player was at the club that season
            # by default, player will be until the end of the year unless they moved
            to_entry = end
            if i != len(season_df) - 1:
                # player left before the end of the year, so end this entry at this time
                to_date = season_df.iloc[i + 1]["date"] - pd.DateOffset(days=1)
                if to_date < end:
                    to_entry = to_date

            teams_df = pd.concat(
                [
                    teams_df,
                    pd.DataFrame(
                        [
                            [
                                current_season,
                                season_df.iloc[i]["new"],
                                season_df.iloc[i]["new_TM"],
                                season_df.iloc[i]["date"],
                                to_entry,
                                set(season_df.iloc[i]["new_TM"].split("-"))
                                in pl_teams_in_season[current_season],
                            ]
                        ]
                    ),
                ]
            )

        current_season = get_next_season(current_season)

    teams_df.columns = ["season", "team", "team_tm", "from", "until", "pl"]

    return teams_df


def get_player_transfer_unavailability(
    player_profile_url: str,
    pl_teams_in_season: dict[str, list[set[str]]] | None = None,
    end_season: str = CURRENT_SEASON,
) -> pd.DataFrame:
    """
    Spells a player was unavailable because they were at a non-Premier League club:
    season, details, reason, from, until, days and games missed.

    Args:
        pl_teams_in_season: Season in "1819" format to the teams that played in it,
            from `get_teams_for_season`.
        end_season: The last season to produce unavailability for.
    """
    if pl_teams_in_season is None:
        pl_teams_in_season = {}
    logger.debug("getting player transfer unavailability for %s", player_profile_url)

    transfer_history = get_player_team_history(
        df=get_player_transfers(player_profile_url),
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
            "days": nan,
            "games": nan,
        }
    )


def get_season_absences(
    season: str, pl_teams_in_season: dict[str, list[set[str]]] | None = None
) -> pd.DataFrame:
    """
    Every absence - injury, suspension, or time at a non-Premier League club - for
    every player in a season, in one data frame.

    A player whose page cannot be scraped is skipped rather than failing the run.
    """
    if pl_teams_in_season is None:
        pl_teams_in_season = {}
    year = season_str_to_year(season)
    logger.info("Finding players...")

    players = get_players_for_season(year)
    absences = []
    logger.info("Querying injuries, suspensions and transfers...")

    for player_name, player_url in track(players):
        with contextlib.suppress(ValueError, IndexError, RemoteError):
            inj = get_player_injuries(player_profile_url=player_url)
            inj["player"] = player_name
            inj["url"] = player_url
            absences.append(inj)
        with contextlib.suppress(ValueError, IndexError, RemoteError):
            sus = get_player_suspensions(player_profile_url=player_url)
            sus = sus[sus["competition"] == "Premier League"]
            sus = sus.drop("competition", axis=1)
            sus["player"] = player_name
            sus["url"] = player_url
            absences.append(sus)
        with contextlib.suppress(ValueError, IndexError, RemoteError):
            tran = get_player_transfer_unavailability(
                player_profile_url=player_url,
                pl_teams_in_season=pl_teams_in_season,
                end_season=season,
            )
            tran["player"] = player_name
            tran["url"] = player_url
            absences.append(tran)

    return filter_season(pd.concat(absences), season)


def scrape_transfermarkt(seasons: list[str]) -> None:
    """Get all player injury and suspension data for several seasons."""
    repo_home = data_dir()

    # get the teams that played in each season we want to scrape
    pl_teams = {}
    for s in seasons:
        teams_in_s = get_teams_for_season(season_str_to_year(s))
        pl_teams[s] = [teams_in_s[i][3] for i in range(20)]

    for season in track(seasons):
        logger.info("-" * 50)
        logger.info("Season: %s", season)

        absences = get_season_absences(season, pl_teams_in_season=pl_teams)
        absences.to_csv(os.path.join(repo_home, f"absences_{season}.csv"), index=False)
