"""Find the alternative names each FPL team goes by in the other data sources."""

import json

from thefuzz import fuzz

from airsenal.core.logging import get_logger
from airsenal.remote.fpl_api import FPLDataFetcher

logger = get_logger(__name__)


def find_best_match(fpl_teams: list[str], team: str) -> tuple[str | None, int]:
    """Fuzzy-match a historical team name against the current FPL team names."""
    best_ratio = 0
    best_match = None
    for t in fpl_teams:
        if fuzz.partial_ratio(t, team) > best_ratio:
            best_ratio = fuzz.partial_ratio(t, team)
            best_match = t
    logger.debug("Best match %s/%s, score %s", best_match, team, best_ratio)
    return best_match, best_ratio


if __name__ == "__main__":
    # get the team names as used in FPL
    df = FPLDataFetcher()
    teamdata = df.get_current_team_data()
    teamdict = {teamdata[k]["name"]: [teamdata[k]["short_name"]] for k in teamdata}

    #    teamdicts = [{teamdata[k]['name']:[teamdata[k]['short_name']]} \
    #                for k in teamdata.keys()]
    fpl_teams = list(teamdict.keys())
    # get the team names from the results csv
    missing = set()
    matched = set()
    history_teams = set()
    for season in ["1415", "1516", "1617", "1718"]:
        filename = f"../data/results_{season}.csv"
        with open(filename) as f:
            lines = f.readlines()
        for line in lines[1:]:
            history_teams.add(line.split(",")[1])
            history_teams.add(line.split(",")[2])

    for team in history_teams:
        if team in fpl_teams:
            matched.add(team)
        else:
            t, score = find_best_match(fpl_teams, team)
            if score == 100:
                teamdict[t].append(team)
                matched.add(team)
            # ugh, ok, do the last few by hand
            elif team == "Manchester United":
                teamdict["Man Utd"].append(team)
                matched.add(team)
            elif team == "Manchester City":
                teamdict["Man City"].append(team)
                matched.add(team)
            elif team == "Tottenham Hotspur":
                teamdict["Spurs"].append(team)
                matched.add(team)
            else:
                missing.add(team)
    # matched teams should be all except promoted ones that haven't
    # been in the prem recently
    logger.info("Num matched: %s", len(matched))

    # print missing teams (should be the relegated ones
    logger.warning("Teams not in this seasons FPL: %s", missing)

    with open("../data/alternative_team_names.json", "w") as outfile:
        outfile.write(json.dumps(teamdict))
