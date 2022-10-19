from typing import List

from sqlalchemy.orm.session import Session

from airsenal.framework.schema import PlayerScore
from airsenal.framework.season import get_teams_for_season
from airsenal.framework.utils import (
    CURRENT_SEASON,
    get_fixtures_for_season,
    get_past_seasons,
    get_player_scores,
    session,
)

CHECK_SEASONS = [CURRENT_SEASON] + get_past_seasons(3)
SEPARATOR = "\n" + ("=" * 50) + "\n"  # used to separate groups of print statements


def result_string(n_error: int) -> str:
    """make string representing check result

    Arguments:
        n_error {int} -- number of errors encountered during check
    """
    if n_error == 0:
        return "OK!"
    else:
        return f"FAIL! {n_error} errors."


def season_num_teams(
    seasons: List[str] = CHECK_SEASONS, session: Session = session
) -> int:
    """Check whether each season has 20 teams.

    Keyword Arguments:
        seasons {list} -- seasons to check (default: {CHECK_SEASONS})
    """
    print("Checking seasons have 20 teams...\n")
    n_error = 0
    for season in seasons:
        teams = get_teams_for_season(season, session)
        if len(teams) != 20:
            n_error += 1
            print(f"Number of teams in {season} season is {len(teams)} (not 20)")

    print("\n", result_string(n_error))
    return n_error


def season_num_new_teams(
    seasons: List[str] = CHECK_SEASONS, session: Session = session
) -> int:
    """Check each season has 3 new teams.

    Keyword Arguments:
        seasons {list} -- seasons to check (default: {CHECK_SEASONS})
    """
    print("Checking seasons have 3 new teams...\n")
    n_error = 0

    teams = [get_teams_for_season(season, session) for season in seasons]
    for i in range(1, len(teams)):
        new_teams = [team for team in teams[i] if team not in teams[i - 1]]
        if len(new_teams) != 3:
            n_error += 1
            print(
                f"Number of teams changed between {seasons[i - 1]} "
                f"and {seasons[i]} is {len(new_teams)} (not 3)"
            )

    print("\n", result_string(n_error))
    return n_error


def season_num_fixtures(
    seasons: List[str] = CHECK_SEASONS, session: Session = session
) -> int:
    """Check each season has 380 fixtures.

    Keyword Arguments:
        seasons {list} -- seasons to check (default: CHECK_SEASONS)
        session {SQLAlchemy session} -- DB session (default:
        airsenal.framework.schema.session)
    """
    print("Checking seasons have 380 fixtures...\n")
    n_error = 0

    for season in seasons:
        fixtures = get_fixtures_for_season(season=season)
        if len(fixtures) != 380:
            n_error += 1
            print(f"Number of fixtures in {season} season is {len(fixtures)} (not 380)")

    print("\n", result_string(n_error))
    return n_error


def fixture_player_teams(
    seasons: List[str] = CHECK_SEASONS, session: Session = session
) -> int:
    """Check players who played in a match are labelled as playing for either
    the home team or the away team.

    Keyword Arguments:
        seasons {[type]} -- seasons to check (default: {CHECK_SEASONS})
        session {SQLAlchemy session} -- DB session (default:
        airsenal.framework.schema.session)
    """
    print("Checking player teams match fixture teams...\n")
    n_error = 0

    for season in seasons:
        fixtures = get_fixtures_for_season(season=season)

        for fixture in fixtures:
            if fixture.result:
                player_scores = get_player_scores(fixture=fixture)

                for score in player_scores:
                    if score.player_team not in [
                        fixture.home_team,
                        fixture.away_team,
                    ]:
                        n_error += 1
                        msg = (
                            f"{fixture}: {score.player} in player_scores but labelled "
                            f"as playing for {score.player_team}."
                        )
                        print(msg)

    print("\n", result_string(n_error))
    return n_error


def fixture_num_players(
    seasons: List[str] = CHECK_SEASONS, session: Session = session
) -> int:
    """Check each fixture has between 11 and 14 players  with at least 1 minute
    in player_scores. For season 19/20 it can be up to 16 players.

    Keyword Arguments:
        seasons {[type]} -- seasons to check (default: {CHECK_SEASONS})
        session {SQLAlchemy session} -- DB session (default:
        airsenal.framework.schema.session)
    """
    print(
        "Checking 11 to 14 players play per team in each fixture...\n"
        "Note:\n"
        "- 2019/20: 5 subs allowed after Covid-19 lockdown (accounted for in checks)\n"
        "- From 2020/21: Concussion subs allowed (may cause false errors)\n"
        "- From 2022/22: 5 subs allowed due to rule change (accounted for in checks)\n"
    )
    n_error = 0

    for season in seasons:
        fixtures = get_fixtures_for_season(season=season)

        for fixture in fixtures:
            result = fixture.result

            if result:
                home_scores = (
                    session.query(PlayerScore)
                    .filter_by(fixture=fixture, player_team=fixture.home_team)
                    .filter(PlayerScore.minutes > 0)
                    .all()
                )

                away_scores = (
                    session.query(PlayerScore)
                    .filter_by(fixture=fixture, player_team=fixture.away_team)
                    .filter(PlayerScore.minutes > 0)
                    .all()
                )

                # Rule change due to shorter season and
                if (fixture.season == "1920" and int(fixture.gameweek) >= 39) or (
                    int(fixture.season[:2]) >= 22
                ):
                    upper_team_limit = 16
                else:
                    upper_team_limit = 14

                if not (
                    (len(home_scores) > 10) and (len(home_scores) <= upper_team_limit)
                ):
                    n_error += 1
                    print(
                        (
                            f"{result}: {len(home_scores)} "
                            "players with minutes > 0 for home team."
                        )
                    )

                if not (
                    (len(away_scores) > 10) and (len(away_scores) <= upper_team_limit)
                ):
                    n_error += 1
                    print(
                        (
                            f"{result}: {len(away_scores)} "
                            "players with minutes > 0 for away team."
                        )
                    )

    print("\n", result_string(n_error))
    return n_error


def fixture_num_goals(
    seasons: List[str] = CHECK_SEASONS, session: Session = session
) -> int:
    """Check individual player goals sum to match result for each fixture.

    Keyword Arguments:
        seasons {[type]} -- seasons to check (default: {CHECK_SEASONS})
        session {SQLAlchemy session} -- DB session (default:
        airsenal.framework.schema.session)
    """
    print("Checking sum of player goals equals match results...\n")
    n_error = 0

    for season in seasons:
        fixtures = get_fixtures_for_season(season=season)

        for fixture in fixtures:
            result = fixture.result

            if result:
                home_scores = (
                    session.query(PlayerScore)
                    .filter_by(fixture=fixture, player_team=fixture.home_team)
                    .all()
                )

                away_scores = (
                    session.query(PlayerScore)
                    .filter_by(fixture=fixture, player_team=fixture.away_team)
                    .all()
                )

                home_goals = sum(score.goals for score in home_scores) + sum(
                    score.own_goals for score in away_scores
                )

                away_goals = sum(score.goals for score in away_scores) + sum(
                    score.own_goals for score in home_scores
                )

                if home_goals != result.home_score:
                    n_error += 1
                    msg = (
                        f"{result}: Player scores sum to {home_goals} "
                        f"but {result.home_score} goals in result for home team"
                    )
                    print(msg)

                if away_goals != result.away_score:
                    n_error += 1
                    msg = (
                        f"{result}: Player scores sum to {away_goals} but "
                        f"{result.away_score} goals in result for away team"
                    )
                    print(msg)

    print("\n", result_string(n_error))
    return n_error


def fixture_num_assists(
    seasons: List[str] = CHECK_SEASONS, session: Session = session
) -> int:
    """Check number of assists is less than or equal to number of goals
    for home and away team in each fixture.
    Less than or equal to as some goals do not result in an assist being
    awarded.

    Keyword Arguments:
        seasons {[type]} -- seasons to check (default: {CHECK_SEASONS})
        session {SQLAlchemy session} -- DB session (default:
        airsenal.framework.schema.session)
    """
    print("Checking no. assists less than or equal to no. goals...\n")
    n_error = 0

    for season in seasons:
        fixtures = get_fixtures_for_season(season=season)

        for fixture in fixtures:
            result = fixture.result
            if result:
                home_scores = (
                    session.query(PlayerScore)
                    .filter_by(fixture=fixture, player_team=fixture.home_team)
                    .all()
                )

                away_scores = (
                    session.query(PlayerScore)
                    .filter_by(fixture=fixture, player_team=fixture.away_team)
                    .all()
                )

                home_assists = sum(score.assists for score in home_scores)
                away_assists = sum(score.assists for score in away_scores)

                if home_assists > result.home_score:
                    n_error += 1
                    msg = (
                        f"{result}: Player assists sum to {home_assists} but "
                        f"{result.home_score} goals in result for home team"
                    )
                    print(msg)

                if away_assists > result.away_score:
                    n_error += 1
                    msg = (
                        f"{result}: Player assists sum to {away_assists} but "
                        f"{result.away_score} goals in result for away team"
                    )
                    print(msg)

    print("\n", result_string(n_error))
    return n_error


def fixture_num_conceded(
    seasons: List[str] = CHECK_SEASONS, session: Session = session
) -> int:
    """Check number of goals concdeded equals goals scored by opposition if
    player played whole match (90 minutes).
    NB: only checks max of player conceded values to avoid potential issues
    with substitutes and goals in stoppage time.

    Keyword Arguments:
        seasons {[type]} -- seasons to check (default: {CHECK_SEASONS})
        session {SQLAlchemy session} -- DB session (default:
        airsenal.framework.schema.session)
    """
    print("Checking no. goals conceded matches goals scored by opponent...\n")
    n_error = 0

    for season in seasons:
        fixtures = get_fixtures_for_season(season=season)

        for fixture in fixtures:
            result = fixture.result
            if result:
                home_scores = (
                    session.query(PlayerScore)
                    .filter_by(
                        fixture=fixture, player_team=fixture.home_team, minutes=90
                    )
                    .all()
                )

                away_scores = (
                    session.query(PlayerScore)
                    .filter_by(
                        fixture=fixture, player_team=fixture.away_team, minutes=90
                    )
                    .all()
                )

                home_conceded = max(score.conceded for score in home_scores)
                away_conceded = max(score.conceded for score in away_scores)

                if home_conceded != result.away_score:
                    n_error += 1
                    msg = (
                        f"{result}: Player conceded {home_conceded} but "
                        f"{result.away_score} goals in result for home team"
                    )
                    print(msg)

                if away_conceded != result.home_score:
                    n_error += 1
                    msg = (
                        f"{result}: Player conceded {away_conceded} but "
                        f"{result.home_score} goals in result for away team"
                    )
                    print(msg)

    print("\n", result_string(n_error))
    return n_error


def run_all_checks(seasons: List[str] = CHECK_SEASONS) -> None:
    print("Running checks for seasons:", seasons)
    print(SEPARATOR)

    functions = {
        "season_num_teams": season_num_teams,
        "season_num_new_teams": season_num_new_teams,
        "season_num_fixtures": season_num_fixtures,
        "fixture_player_teams": fixture_player_teams,
        "fixture_num_players": fixture_num_players,
        "fixture_num_goals": fixture_num_goals,
        "fixture_num_assists": fixture_num_assists,
        "fixture_num_conceded": fixture_num_conceded,
    }
    results = {}

    for name, fn in functions.items():
        results[name] = fn(seasons)
        print(SEPARATOR)

    print("SUMMARY\n-------")
    print("Seasons:", seasons, "\n")
    for name, res in results.items():
        print(f"{name}: {result_string(res)}")

    n_tests = len(functions)
    n_passed = sum(1 for _, r in results.items() if r == 0)
    n_total_errors = sum(r for _, r in results.items())
    print(
        (
            f"\nOVERALL: Passed {n_passed} out of {n_tests} tests with "
            f"{n_total_errors} errors."
        )
    )


if __name__ == "__main__":
    run_all_checks()
