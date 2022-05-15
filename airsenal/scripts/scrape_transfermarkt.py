"""
Get player injury, suspension and availability data from TransferMarkt
"""
import os
from time import sleep
from typing import List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

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
    return pd.read_html(page.content, match="Injury")[0]


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
    return suspended


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
        sleep(1)
    return list(players)


def main(seasons: List[int]):
    """Get all player injury and suspension data for multiple seasons

    Parameters
    ----------
    seasons : List[int]
        seasons to query - list of years the season started (int), rather than usual str
        representation
    """
    print("Finding players...")
    players = set()
    for season in seasons:
        print(season)
        players.update(get_players_for_season(season))

    print("Querying injuries and suspensions...")
    injuries = []
    suspensions = []
    for player_name, player_url in tqdm(players):
        try:
            inj = get_player_injuries(player_url)
            inj["player"] = player_name
            inj["url"] = player_url
            injuries.append(inj)
        except (ValueError, IndexError):
            pass
        try:
            sus = get_player_suspensions(player_url)
            sus["player"] = player_name
            sus["url"] = player_url
            suspensions.append(sus)
        except (ValueError, IndexError):
            pass
        sleep(1)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    injuries = pd.concat(injuries)
    injuries.to_csv(os.path.join(data_dir, "injuries.csv"), index=False)
    suspensions = pd.concat(suspensions)
    suspensions.to_csv(os.path.join(data_dir, "suspensions.csv"), index=False)


if __name__ == "__main__":
    main([2015, 2016, 2017, 2018, 2019, 2020, 2021])
