from sqlalchemy import select
from sqlalchemy.orm.session import Session

from airsenal.core.output import get_logger
from airsenal.db.models import PlayerScore
from airsenal.db.session import get_session
from airsenal.domain.season import get_teams_for_season
from airsenal.framework.utils import (
    CURRENT_SEASON,
    get_fixtures_for_season,
    get_past_seasons,
    get_player_scores,
)

logger = get_logger(__name__)

CHECK_SEASONS = [CURRENT_SEASON, *get_past_seasons(3)]


def result_string(n_error: int) -> str:
    """make string representing check result

    Arguments:
        n_error {int} -- number of errors encountered during check
    """
    if n_error == 0:
        return "OK!"
    return f"FAIL! {n_error} errors."


def season_num_teams(
    seasons: list[str] = CHECK_SEASONS, dbsession: Session | None = None
) -> int:
    """Check whether each season has 20 teams.

    Keyword Arguments:
        seasons {list} -- seasons to check (default: {CHECK_SEASONS})
    """
    dbsession = dbsession if dbsession is not None else get_session()
    logger.info("Checking seasons have 20 teams...")
    n_error = 0
    for season in seasons:
        teams = get_teams_for_season(season, dbsession)
        if len(teams) != 20:
            n_error += 1
            logger.warning(
                "Number of teams in %s season is %s (not 20)", season, len(teams)
            )

    logger.info(result_string(n_error))
    return n_error


def season_num_new_teams(
    seasons: list[str] = CHECK_SEASONS, dbsession: Session | None = None
) -> int:
    """Check each season has 3 new teams.

    Keyword Arguments:
        seasons {list} -- seasons to check (default: {CHECK_SEASONS})
    """
    dbsession = dbsession if dbsession is not None else get_session()
    logger.info("Checking seasons have 3 new teams...")
    n_error = 0

    teams = [get_teams_for_season(season, dbsession) for season in seasons]
    for i in range(1, len(teams)):
        new_teams = [team for team in teams[i] if team not in teams[i - 1]]
        if len(new_teams) != 3:
            n_error += 1
            logger.warning(
                "Number of teams changed between %s and %s is %s (not 3)",
                seasons[i - 1],
                seasons[i],
                len(new_teams),
            )

    logger.info(result_string(n_error))
    return n_error


def season_num_fixtures(
    seasons: list[str] = CHECK_SEASONS, dbsession: Session | None = None
) -> int:
    """Check each season has 380 fixtures.

    Keyword Arguments:
        seasons {list} -- seasons to check (default: CHECK_SEASONS)
        dbsession {SQLAlchemy session} -- DB session (default:
        airsenal.db.session.get_session())
    """
    dbsession = dbsession if dbsession is not None else get_session()
    logger.info("Checking seasons have 380 fixtures...")
    n_error = 0

    for season in seasons:
        fixtures = get_fixtures_for_season(season=season, dbsession=dbsession)
        if len(fixtures) != 380:
            n_error += 1
            logger.warning(
                "Number of fixtures in %s season is %s (not 380)",
                season,
                len(fixtures),
            )

    logger.info(result_string(n_error))
    return n_error


def fixture_player_teams(
    seasons: list[str] = CHECK_SEASONS, dbsession: Session | None = None
) -> int:
    """Check players who played in a match are labelled as playing for either
    the home team or the away team.

    Keyword Arguments:
        seasons {[type]} -- seasons to check (default: {CHECK_SEASONS})
        dbsession {SQLAlchemy session} -- DB session (default:
        airsenal.db.session.get_session())
    """
    dbsession = dbsession if dbsession is not None else get_session()
    logger.info("Checking player teams match fixture teams...")
    n_error = 0

    for season in seasons:
        fixtures = get_fixtures_for_season(season=season, dbsession=dbsession)

        for fixture in fixtures:
            if fixture.result:
                player_scores = get_player_scores(fixture=fixture, dbsession=dbsession)
                if player_scores is None:
                    logger.warning("Fixture %s has no player scores", fixture)
                    continue
                if isinstance(player_scores, PlayerScore):
                    player_scores = [player_scores]

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
                        logger.warning(msg)

    logger.info(result_string(n_error))
    return n_error


def fixture_num_players(
    seasons: list[str] = CHECK_SEASONS, dbsession: Session | None = None
) -> int:
    """Check each fixture has between 11 and 14 players  with at least 1 minute
    in player_scores. For season 19/20 it can be up to 16 players.

    Keyword Arguments:
        seasons {[type]} -- seasons to check (default: {CHECK_SEASONS})
        dbsession {SQLAlchemy session} -- DB session (default:
        airsenal.db.session.get_session())
    """
    dbsession = dbsession if dbsession is not None else get_session()
    logger.info(
        "Checking 11 to 14 players play per team in each fixture...\n"
        "Note:\n"
        "- 2019/20: 5 subs allowed after Covid-19 lockdown (accounted for in checks)\n"
        "- From 2020/21: Concussion subs allowed (may cause false errors)\n"
        "- From 2022/22: 5 subs allowed due to rule change (accounted for in checks)"
    )
    n_error = 0

    for season in seasons:
        fixtures = get_fixtures_for_season(season=season, dbsession=dbsession)

        for fixture in fixtures:
            result = fixture.result

            if result:
                home_scores = dbsession.scalars(
                    select(PlayerScore).where(
                        PlayerScore.fixture_id == fixture.fixture_id,
                        PlayerScore.player_team == fixture.home_team,
                        PlayerScore.minutes > 0,
                    )
                ).all()

                away_scores = dbsession.scalars(
                    select(PlayerScore).where(
                        PlayerScore.fixture_id == fixture.fixture_id,
                        PlayerScore.player_team == fixture.away_team,
                        PlayerScore.minutes > 0,
                    )
                ).all()

                # No. subs changes during Covid and later rule changes
                if (
                    fixture.season == "1920"
                    and (fixture.gameweek is not None and fixture.gameweek >= 39)
                ) or (int(fixture.season[:2]) >= 22):
                    upper_team_limit = 16
                else:
                    upper_team_limit = 14

                if not (
                    (len(home_scores) > 10) and (len(home_scores) <= upper_team_limit)
                ):
                    n_error += 1
                    logger.warning(
                        "%s: %s players with minutes > 0 for home team.",
                        result,
                        len(home_scores),
                    )

                if not (
                    (len(away_scores) > 10) and (len(away_scores) <= upper_team_limit)
                ):
                    n_error += 1
                    logger.warning(
                        "%s: %s players with minutes > 0 for away team.",
                        result,
                        len(away_scores),
                    )

    logger.info(result_string(n_error))
    return n_error


def fixture_num_goals(
    seasons: list[str] = CHECK_SEASONS, dbsession: Session | None = None
) -> int:
    """Check individual player goals sum to match result for each fixture.

    Keyword Arguments:
        seasons {[type]} -- seasons to check (default: {CHECK_SEASONS})
        dbsession {SQLAlchemy session} -- DB session (default:
        airsenal.db.session.get_session())
    """
    dbsession = dbsession if dbsession is not None else get_session()
    logger.info("Checking sum of player goals equals match results...")
    n_error = 0

    for season in seasons:
        fixtures = get_fixtures_for_season(season=season, dbsession=dbsession)

        for fixture in fixtures:
            result = fixture.result

            if result:
                home_scores = dbsession.scalars(
                    select(PlayerScore).where(
                        PlayerScore.fixture_id == fixture.fixture_id,
                        PlayerScore.player_team == fixture.home_team,
                    )
                ).all()

                away_scores = dbsession.scalars(
                    select(PlayerScore).where(
                        PlayerScore.fixture_id == fixture.fixture_id,
                        PlayerScore.player_team == fixture.away_team,
                    )
                ).all()

                home_goals = sum(score.goals for score in home_scores) + sum(
                    score.own_goals or 0 for score in away_scores
                )

                away_goals = sum(score.goals for score in away_scores) + sum(
                    score.own_goals or 0 for score in home_scores
                )

                if home_goals != result.home_score:
                    n_error += 1
                    msg = (
                        f"{result}: Player scores sum to {home_goals} "
                        f"but {result.home_score} goals in result for home team"
                    )
                    logger.warning(msg)

                if away_goals != result.away_score:
                    n_error += 1
                    msg = (
                        f"{result}: Player scores sum to {away_goals} but "
                        f"{result.away_score} goals in result for away team"
                    )
                    logger.warning(msg)

    logger.info(result_string(n_error))
    return n_error


def fixture_num_assists(
    seasons: list[str] = CHECK_SEASONS, dbsession: Session | None = None
) -> int:
    """Check number of assists is less than or equal to number of goals
    for home and away team in each fixture.
    Less than or equal to as some goals do not result in an assist being
    awarded.

    Keyword Arguments:
        seasons {[type]} -- seasons to check (default: {CHECK_SEASONS})
        dbsession {SQLAlchemy session} -- DB session (default:
        airsenal.db.session.get_session())
    """
    dbsession = dbsession if dbsession is not None else get_session()
    logger.info("Checking no. assists less than or equal to no. goals...")
    n_error = 0

    for season in seasons:
        fixtures = get_fixtures_for_season(season=season, dbsession=dbsession)

        for fixture in fixtures:
            result = fixture.result
            if result:
                home_scores = dbsession.scalars(
                    select(PlayerScore).where(
                        PlayerScore.fixture_id == fixture.fixture_id,
                        PlayerScore.player_team == fixture.home_team,
                    )
                ).all()

                away_scores = dbsession.scalars(
                    select(PlayerScore).where(
                        PlayerScore.fixture_id == fixture.fixture_id,
                        PlayerScore.player_team == fixture.away_team,
                    )
                ).all()

                home_assists = sum(score.assists for score in home_scores)
                away_assists = sum(score.assists for score in away_scores)

                if home_assists > result.home_score:
                    n_error += 1
                    msg = (
                        f"{result}: Player assists sum to {home_assists} but "
                        f"{result.home_score} goals in result for home team"
                    )
                    logger.warning(msg)

                if away_assists > result.away_score:
                    n_error += 1
                    msg = (
                        f"{result}: Player assists sum to {away_assists} but "
                        f"{result.away_score} goals in result for away team"
                    )
                    logger.warning(msg)

    logger.info(result_string(n_error))
    return n_error


def fixture_num_conceded(
    seasons: list[str] = CHECK_SEASONS, dbsession: Session | None = None
) -> int:
    """Check number of goals concdeded equals goals scored by opposition if
    player played whole match (90 minutes).
    NB: only checks max of player conceded values to avoid potential issues
    with substitutes and goals in stoppage time.

    Keyword Arguments:
        seasons {[type]} -- seasons to check (default: {CHECK_SEASONS})
        dbsession {SQLAlchemy session} -- DB session (default:
        airsenal.db.session.get_session())
    """
    dbsession = dbsession if dbsession is not None else get_session()
    logger.info("Checking no. goals conceded matches goals scored by opponent...")
    n_error = 0

    for season in seasons:
        fixtures = get_fixtures_for_season(season=season, dbsession=dbsession)

        for fixture in fixtures:
            result = fixture.result
            if result:
                home_scores = dbsession.scalars(
                    select(PlayerScore).where(
                        PlayerScore.fixture_id == fixture.fixture_id,
                        PlayerScore.player_team == fixture.home_team,
                        PlayerScore.minutes == 90,
                    )
                ).all()

                away_scores = dbsession.scalars(
                    select(PlayerScore).where(
                        PlayerScore.fixture_id == fixture.fixture_id,
                        PlayerScore.player_team == fixture.away_team,
                        PlayerScore.minutes == 90,
                    )
                ).all()

                home_conceded = max(score.conceded for score in home_scores)
                away_conceded = max(score.conceded for score in away_scores)

                if home_conceded != result.away_score:
                    n_error += 1
                    msg = (
                        f"{result}: Player conceded {home_conceded} but "
                        f"{result.away_score} goals in result for home team"
                    )
                    logger.warning(msg)

                if away_conceded != result.home_score:
                    n_error += 1
                    msg = (
                        f"{result}: Player conceded {away_conceded} but "
                        f"{result.home_score} goals in result for away team"
                    )
                    logger.warning(msg)

    logger.info(result_string(n_error))
    return n_error


def run_all_checks(seasons: list[str] = CHECK_SEASONS) -> None:
    logger.info("Running checks for seasons: %s", seasons)

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
    results = {name: fn(seasons) for name, fn in functions.items()}

    logger.info("[bold]SUMMARY[/bold]")
    logger.info("Seasons: %s", seasons)
    for name, res in results.items():
        logger.info("%s: %s", name, result_string(res))

    n_tests = len(functions)
    n_passed = sum(1 for _, r in results.items() if r == 0)
    n_total_errors = sum(r for _, r in results.items())
    logger.info(
        "OVERALL: Passed %s out of %s tests with %s errors.",
        n_passed,
        n_tests,
        n_total_errors,
    )


if __name__ == "__main__":
    run_all_checks()
