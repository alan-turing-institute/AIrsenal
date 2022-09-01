"""
Get player injury, suspension and availability data from TransferMarkt
"""

import contextlib
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from airsenal.framework.season import CURRENT_SEASON

TRANSFERMARKT_URL = "https://www.transfermarkt.co.uk"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)"
        "Chrome/47.0.2526.106 Safari/537.36"
    )
}


def get_teams_for_season(season: int) -> List[Tuple[str, str]]:
    """Get the names and TransferMarkt URLs for all the teams in this season.

    Parameters
    ----------
    season : str
        season to query - the year the season started (int), rather than usual str
        representation

    Returns
    -------
    List[Tuple[str, str]]
        List of name and relative URL for each team
    """
    # get list of teams
    url_season = (
        f"{TRANSFERMARKT_URL}/premier-league/startseite/wettbewerb/GB1/plus/"
        f"?saison_id={season}"
    )
    page = requests.get(url_season, headers=HEADERS)
    soup = BeautifulSoup(page.content, features="lxml")
    rows = soup.find_all("td", {"class": "zentriert no-border-rechts"})
    return [(r.a.get("title"), r.a.get("href")) for r in rows][:20]


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


def tidy_df(df: pd.DataFrame, days_name="days") -> pd.DataFrame:
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


def get_player_injuries(player_profile_url: str) -> pd.DataFrame:
    """Get a player's injury history.
    Example TransferMarkt page:
    https://www.transfermarkt.co.uk/kyle-walker/verletzungen/spieler/95424

    Parameters
    ----------
    player_profile_url : str
        Relative URL to the player's homepage

    Returns
    -------
    pd.DataFrame
        Player injuries: type, date, length and games missed
    """
    page = requests.get(
        (
            f"{TRANSFERMARKT_URL}"
            f"{player_profile_url.replace('/profil/', '/verletzungen/')}"
        ),
        headers=HEADERS,
    )
    injuries = pd.read_html(page.content, match="Injury")[0]
    return tidy_df(injuries, days_name="days")


def get_player_suspensions(player_profile_url: str) -> pd.DataFrame:
    """Get a players non-injury unavailability history (suspensions and other reasons)
    Example TransferMarkt page:
    https://www.transfermarkt.co.uk/kyle-walker/ausfaelle/spieler/95424

    Parameters
    ----------
    player_profile_url : str
        Relative URL to the player's homepage

    Returns
    -------
    pd.DataFrame
        Player unavailability: reason, competition, date, length, games missed
    """
    p = requests.get(
        f"{TRANSFERMARKT_URL}{player_profile_url.replace('/profil/', '/ausfaelle/')}",
        headers=HEADERS,
    )
    suspended = pd.read_html(p.content, match="Absence/Suspension")[0]

    player_soup = BeautifulSoup(p.content, features="lxml")
    comp = []
    for row in player_soup.find_all("table")[0].find_all("tr")[1:]:
        try:
            comp.append(row.find_all("img")[0].get("title"))
        except IndexError:
            comp.append("")
    suspended["competition"] = comp
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
    for _, team_url in tqdm(teams):
        players.update(get_team_players(team_url))
    return list(players)


def season_str_to_year(season: str) -> int:
    """Convert season in "1819" format to the year the season started (2018)

    Parameters
    ----------
    season : str
        Season string in "1819" format (for 2018/19 season)

    Returns
    -------
    int
        Year season started
    """
    return int(f"20{season[:2]}")


def get_season_injuries_suspensions(season: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get injury and suspension data for a season

    Parameters
    ----------
    season : str
        Season to query in "1819" format (for 2018/19 season)

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Injury and suspension data frames for all players in this season
    """
    year = season_str_to_year(season)
    print("Finding players...")
    players = get_players_for_season(year)
    injuries = []
    suspensions = []
    print("Querying injuries and suspensions...")
    for player_name, player_url in tqdm(players):
        with contextlib.suppress(ValueError, IndexError):
            inj = get_player_injuries(player_url)
            inj["player"] = player_name
            inj["url"] = player_url
            injuries.append(inj)
        with contextlib.suppress(ValueError, IndexError):
            sus = get_player_suspensions(player_url)
            sus["player"] = player_name
            sus["url"] = player_url
            suspensions.append(sus)

    injuries = pd.concat(injuries)
    suspensions = pd.concat(suspensions)
    return filter_season(injuries, season), filter_season(suspensions, season)


def main(seasons: List[str]):
    """Get all player injury and suspension data for multiple seasons

    Parameters
    ----------
    seasons : List[str]
        seasons to query in format "1819" (for 2018/19 season)
    """
    REPO_HOME = os.path.join(os.path.dirname(__file__), "..", "data")

    for season in tqdm(seasons):
        print(f"Season: {season}")
        injuries, suspensions = get_season_injuries_suspensions(season)
        injuries.to_csv(os.path.join(REPO_HOME, f"injuries_{season}.csv"), index=False)
        suspensions.to_csv(
            os.path.join(REPO_HOME, f"suspensions_{season}.csv"), index=False
        )


if __name__ == "__main__":
    main([CURRENT_SEASON])
