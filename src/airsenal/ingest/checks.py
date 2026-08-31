"""
Consistency checks over the ingested database.

Run by `airsenal db check`. Each check covers `CHECK_SEASONS` - the current
season plus the three before it - and logs what it found rather than raising, so
one bad season does not hide the rest.
"""

from sqlalchemy import select
from sqlalchemy.orm.session import Session

from airsenal.core.logging import get_logger
from airsenal.db.models import PlayerScore
from airsenal.db.queries.fixtures import get_fixtures_for_season
from airsenal.db.queries.scores import get_player_scores
from airsenal.db.queries.teams import get_teams_for_season
from airsenal.db.session import get_session
from airsenal.game.season import CURRENT_SEASON, get_past_seasons

logger = get_logger(__name__)

CHECK_SEASONS = [CURRENT_SEASON, *get_past_seasons(3)]


def result_string(n_error: int) -> str:
    """Summarise a check's outcome as "OK!" or a count of errors."""
    if n_error == 0:
        return "OK!"
    return f"FAIL! {n_error} errors."


def season_num_teams(
    seasons: list[str] = CHECK_SEASONS, dbsession: Session | None = None
) -> int:
    """Check whether each season has 20 teams."""
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
    """Check each season has 3 new teams."""
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
    """Check each season has 380 fixtures."""
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
    """Check every player in a match is labelled with one of the two teams playing."""
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
    """
    Check each fixture has 11 to 14 players with at least a minute played.

    19/20 allowed five substitutes, so up to 16 there.
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
    """Check individual player goals sum to match result for each fixture."""
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
    """
    Check each team's assists in a fixture do not exceed its goals.

    Fewer is normal - not every goal is credited with an assist.
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
    """
    Check goals conceded match the opposition's goals scored.

    Only the maximum across a team's players is checked, which sidesteps
    substitutes and goals in stoppage time. A team with nobody recorded as playing
    the full 90 has no figure to compare, and is reported rather than checked.
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

                # `default` rather than a bare max(): a fixture whose result has
                # landed before its player scores have - which is the state an
                # interrupted update leaves, and the one this check exists to
                # find - has no 90-minute players, and an empty max() ended the
                # run with a ValueError instead of reporting the fixture.
                for scores, conceded_by_opponent, side in (
                    (home_scores, result.away_score, "home"),
                    (away_scores, result.home_score, "away"),
                ):
                    conceded = max((score.conceded for score in scores), default=None)
                    if conceded is None:
                        n_error += 1
                        logger.warning(
                            "%s: no %s players recorded as playing 90 minutes, "
                            "so goals conceded cannot be checked",
                            result,
                            side,
                        )
                    elif conceded != conceded_by_opponent:
                        n_error += 1
                        logger.warning(
                            "%s: Player conceded %s but %s goals in result for %s team",
                            result,
                            conceded,
                            conceded_by_opponent,
                            side,
                        )

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
