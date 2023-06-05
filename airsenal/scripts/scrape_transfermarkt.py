"""
Get player injury, suspension and availability data from TransferMarkt
"""

import argparse
import contextlib
import os
from cmath import nan
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from airsenal.framework.season import CURRENT_SEASON, season_str_to_year
from airsenal.framework.utils import get_next_season, get_start_end_dates_of_season

TRANSFERMARKT_URL = "https://www.transfermarkt.co.uk"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)"
        "Chrome/47.0.2526.106 Safari/537.36"
    )
}


def get_teams_for_season(
    season: int, verbose: bool = False
) -> List[Tuple[str, str, str, set]]:
    """Get the names and TransferMarkt URLs for all the teams in this season.

    Parameters
    ----------
    season : str
        season to query - the year the season started (int), rather than usual str
        representation
    verbose : bool
        Whether or not to print out information on progress

    Returns
    -------
    List[Tuple[str, str, str, set]]
        List of name, relative URL, team identifier used by TransferMarkt
        and the set of the team identifier split by '-' (useful for checking
        if a team played in this season) for each team
    """
    if verbose:
        print(
            f"getting teams that played in {str(season)[2:]}/{str(season+1)[2:]} season"
        )

    # get list of teams
    url_season = (
        f"{TRANSFERMARKT_URL}/premier-league/startseite/wettbewerb/GB1/plus/"
        f"?saison_id={season}"
    )
    page = requests.get(url_season, headers=HEADERS)
    soup = BeautifulSoup(page.content, features="lxml")
    rows = soup.find_all("td", {"class": "zentriert no-border-rechts"})

    return [
        (
            r.a.get("title"),
            r.a.get("href"),
            r.a.get("href").split("/")[1],
            set(r.a.get("href").split("/")[1].split("-")),
        )
        for r in rows
    ][:20]


def get_team_players(team_season_url: str) -> List[Tuple[str, str]]:
    """Get all the players in a team's squad for a season.
    Example TransferMarkt page:
    https://www.transfermarkt.co.uk/manchester-city/startseite/verein/281/saison_id/2021

    Parameters
    ----------
    team_season_url : str
        Relative URL to the season homepage for this team

    Returns
    -------
    List[Tuple[str, str]]
        List of player name and relative URL
    """
    page = requests.get(f"{TRANSFERMARKT_URL}{team_season_url}", headers=HEADERS)
    team_soup = BeautifulSoup(page.content, features="lxml")
    player_rows = team_soup.find_all("td", {"class": "posrela"})

    return [
        (r.find_all("a")[-1].get("title"), r.find_all("a")[-1].get("href"))
        for r in player_rows
    ]


def tidy_df(df: pd.DataFrame, days_name: str = "days") -> pd.DataFrame:
    """Clean column names, data types, and missing data for injury/suspension data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw injury or suspension data from TransferMarkt

    Returns
    -------
    pd.DataFrame
        Cleaned data
    """
    df.columns = df.columns.str.lower()
    df = df.rename(columns={"games missed": "games"})

    with contextlib.suppress(AttributeError):
        # can fail with AttributeError if all values are missing
        df["season"] = df["season"].str.replace("/", "")
    df = df.replace({"-": np.nan, f"? {days_name}": np.nan, "?": np.nan})
    df["from"] = pd.to_datetime(df["from"], format="%b %d, %Y", errors="coerce")
    df["until"] = pd.to_datetime(df["until"], format="%b %d, %Y", errors="coerce")

    with contextlib.suppress(AttributeError):
        # can fail with AttributeError if all values are missing
        df["days"] = df["days"].str.replace(f" {days_name}", "")
    df["days"] = df["days"].astype("float").astype("Int32")
    df["games"] = df["games"].astype("float").astype("Int32")

    return df.convert_dtypes()


def filter_season(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Extract rows for a given season from a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame with season column in "1819" format
    season : str
        Season to extract, in "1819" format

    Returns
    -------
    pd.DataFrame
        Rows of data frame for specified season
    """
    return df[df["season"] == season]


def get_player_injuries(player_profile_url: str, verbose: bool = False) -> pd.DataFrame:
    """Get a player's injury history.
    Example TransferMarkt page:
    https://www.transfermarkt.co.uk/kyle-walker/verletzungen/spieler/95424

    Parameters
    ----------
    player_profile_url : str
        Relative URL to the player's homepage
    verbose : bool
        Whether or not to print out information on progress

    Returns
    -------
    pd.DataFrame
        Player injuries: type, date, length and games missed
    """
    if verbose:
        print(f"getting player injuries for {player_profile_url}")

    page = requests.get(
        (
            f"{TRANSFERMARKT_URL}"
            f"{player_profile_url.replace('/profil/', '/verletzungen/')}"
        ),
        headers=HEADERS,
    )
    if verbose:
        print(f"processing player injuries for {player_profile_url}")

    injuries = pd.read_html(page.content, match="Injury")[0]
    injuries = injuries.rename(columns={"Injury": "Details"})
    injuries["Reason"] = "injury"

    return tidy_df(injuries, days_name="days")


def get_reason(details: str) -> str:
    """get suspension/absence reason category (not for injuries)"""
    return "suspension" if "suspen" in details.lower() else "absence"


def get_player_suspensions(
    player_profile_url: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """Get a players non-injury unavailability history (suspensions and other reasons)
    Example TransferMarkt page:
    https://www.transfermarkt.co.uk/kyle-walker/ausfaelle/spieler/95424

    Parameters
    ----------
    player_profile_url : str
        Relative URL to the player's homepage
    verbose : bool
        Whether or not to print out information on progress

    Returns
    -------
    pd.DataFrame
        Player unavailability: reason, competition, date, length, games missed
    """
    if verbose:
        print(f"getting player suspensions for {player_profile_url}")

    p = requests.get(
        f"{TRANSFERMARKT_URL}{player_profile_url.replace('/profil/', '/ausfaelle/')}",
        headers=HEADERS,
    )

    if verbose:
        print(f"processing player suspensions for {player_profile_url}")

    suspended = pd.read_html(p.content, match="Absence/Suspension")[0]
    player_soup = BeautifulSoup(p.content, features="lxml")

    comp = []
    for row in player_soup.find_all("table")[0].find_all("tr")[1:]:
        try:
            comp.append(row.find_all("img")[0].get("title"))
        except IndexError:
            comp.append("")
    suspended["Competition"] = comp
    suspended = suspended.rename(columns={"Absence/Suspension": "Details"})
    suspended["Reason"] = [get_reason(detail) for detail in suspended["Details"]]

    return tidy_df(suspended, days_name="Tage")


def get_players_for_season(season: int) -> List[Tuple[str, str]]:
    """Get all the players at any premier league club in a season

    Parameters
    ----------
    season : int
        season to query - the year the season started (int), rather than usual str
        representation

    Returns
    -------
    List[Tuple[str, str]]
        List of player name and relative URL
    """
    teams = get_teams_for_season(season)
    players = set()
    for _, team_url, __, ___ in tqdm(teams):
        players.update(get_team_players(team_url))

    return list(players)


def remove_youth_or_reserve_suffix(team_name: str) -> str:
    """Removes any youth or reserve team suffixes,
    e.g. "-youth", "-b", "-u18", "-u21", "-u23", etc.

    This may cause confusion later to say a player is unavailable
    just because they're in the youth team. This does not in
    practice mean they are unable to play in a premier league game

    Parameters
    ----------
    team_name : str
        team name given by TM, e.g. arsenal-fc or arsenal-fc-u21

    Returns
    -------
    str
        team name but with youth or reserve team suffixes removed
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
    verbose: bool = False,
) -> pd.DataFrame:
    """Get a player's transfer history.
    Example TransferMarkt page:
    https://www.transfermarkt.co.uk/kyle-walker/transfers/spieler/95424

    Parameters
    ----------
    player_profile_url : str
        Relative URL to the player's homepage
    verbose : bool
        Whether or not to print out information on progress

    Returns
    -------
    pd.DataFrame
        Player transfers: season, date, old team and new team
    """
    if verbose:
        print(f"getting player transfer history for {player_profile_url}")

    page = requests.get(
        (
            f"{TRANSFERMARKT_URL}"
            f"{player_profile_url.replace('/profil/', '/transfers/')}"
        ),
        headers=HEADERS,
    )

    if verbose:
        print(f"processing player transfer history for {player_profile_url}")

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
        old_link = old.a.get("href")
        old_tm_identifier = old_link.split("/")[1]
        # new club details
        new = soup.find_all("div", class_="tm-player-transfer-history-grid__new-club")[
            i
        ]
        new_club = " ".join(new.getText().split())
        new_link = new.a.get("href")
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
    raw["date"] = pd.to_datetime(raw["date"], format="%b %d, %Y", errors="coerce")

    return raw.iloc[::-1]


def get_player_team_history(
    df: pd.DataFrame, pl_teams_in_season: dict = {}, end_season: str = CURRENT_SEASON
) -> pd.DataFrame:
    """Get a player's team/club history given their transfer data.
    Example TransferMarkt page:
    https://www.transfermarkt.co.uk/kyle-walker/transfers/spieler/95424

    Parameters
    ----------
    df : pd.DataFrame
        Transfer data for a player obtained with get_player_transfers()
    pl_teams_in_season : dict
        Dictionary with keys as season (in str format) and items as the
        teams that played in that season (obtained with get_teams_for_season())
    end_season : str
        Where to stop the function getting data for. Default set to CURRENT_SEASON

    Returns
    -------
    pd.DataFrame
        Player team history: season, team, from, until, in the premier league or not
    """
    teams_df = pd.DataFrame()
    current_season = "".join(df.iloc[0]["season"].split("/"))
    diff = int(current_season[:2]) - int(end_season[2:])
    for i in range(abs(diff)):
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
    pl_teams_in_season: dict = {},
    end_season: str = CURRENT_SEASON,
    verbose: bool = False,
) -> pd.DataFrame:
    """Get a player's unavailability from transfers
    Example TransferMarkt page:
    https://www.transfermarkt.co.uk/kyle-walker/transfers/spieler/95424

    Parameters
    ----------
    player_profile_url : str
        Relative URL to the player's homepage
    pl_teams_in_season : dict
        Dictionary with keys as season (in str format) and items as the
        teams that played in that season (obtained with get_teams_for_season())
    end_season : str
        Where to stop the function getting data for. Default set to CURRENT_SEASON
    verbose : bool
        Whether or not to print out information on progress

    Returns
    -------
    pd.DataFrame
        Player's unavailability due to transfers: season,
        details, reason, from, until, days, games missed
    """
    if verbose:
        print(f"getting player transfer unavailability for {player_profile_url}")

    transfer_history = get_player_team_history(
        df=get_player_transfers(player_profile_url),
        pl_teams_in_season=pl_teams_in_season,
        end_season=end_season,
    )

    if verbose:
        print(f"processing player transfer unavailability for {player_profile_url}")

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
    season: str,
    pl_teams_in_season: dict = {},
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Get injury and suspension data for a season

    Parameters
    ----------
    season : str
        Season to query in "1819" format (for 2018/19 season)
    verbose : bool
        Whether or not to print out information on progress

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Injury and suspension data frames for all players in this season
    """
    year = season_str_to_year(season)
    if verbose:
        print("Finding players...")

    players = get_players_for_season(year)
    absences = []
    if verbose:
        print("Querying injuries, suspensions and transfers...")

    for player_name, player_url in tqdm(players):
        with contextlib.suppress(ValueError, IndexError):
            inj = get_player_injuries(player_profile_url=player_url, verbose=verbose)
            inj["player"] = player_name
            inj["url"] = player_url
            absences.append(inj)
        with contextlib.suppress(ValueError, IndexError):
            sus = get_player_suspensions(player_profile_url=player_url, verbose=verbose)
            sus = sus[sus["competition"] == "Premier League"]
            sus = sus.drop("competition", axis=1)
            sus["player"] = player_name
            sus["url"] = player_url
            absences.append(sus)
        with contextlib.suppress(ValueError, IndexError):
            tran = get_player_transfer_unavailability(
                player_profile_url=player_url,
                pl_teams_in_season=pl_teams_in_season,
                end_season=season,
                verbose=verbose,
            )
            tran["player"] = player_name
            tran["url"] = player_url
            absences.append(tran)

    absences = pd.concat(absences)
    return filter_season(absences, season)


def scrape_transfermarkt(seasons: List[str], verbose: bool = False):
    """Get all player injury and suspension data for mutiple seasons

    Parameters
    ----------
    seasons : List[str]
        seasons to query in format "1819" (for 2018/19 season)
    verbose : bool
        Whether or not to print out information on progress
    """
    REPO_HOME = os.path.join(os.path.dirname(__file__), "..", "data")

    # get the teams that played in each season we want to scrape
    pl_teams = {}
    for s in seasons:
        teams_in_s = get_teams_for_season(season_str_to_year(s), verbose=verbose)
        pl_teams[s] = [teams_in_s[i][3] for i in range(20)]

    for season in tqdm(seasons):
        if verbose:
            print("-" * 50)
            print(f"Season: {season}")

        absences = get_season_absences(
            season, pl_teams_in_season=pl_teams, verbose=verbose
        )
        absences.to_csv(os.path.join(REPO_HOME, f"absences_{season}.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Get injury, suspension and other absence data from Transfermarkt"
    )
    parser.add_argument(
        "--season",
        help=(
            "Which season(s) to update (comma separated, e.g. 2021,2122 "
            "for 2020/21 and 2021/22 seasons)"
        ),
        type=str,
        default=CURRENT_SEASON,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help=(
            "Which season(s) to update (comma separated, e.g. 2021,2122 "
            "for 2020/21 and 2021/22 seasons)"
        ),
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    seasons = args.season.split(",")
    seasons = [s.strip() for s in seasons]
    scrape_transfermarkt(seasons, args.verbose)


if __name__ == "__main__":
    main()
