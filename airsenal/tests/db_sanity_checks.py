from airsenal.framework.utils import (
    get_past_seasons,
    get_teams_for_season,
    session,
    CURRENT_SEASON,
    get_fixtures_for_season,
    get_result_for_fixture,
    get_player_scores_for_fixture
)
from airsenal.framework.schema import (
    Fixture,
    Result,
    PlayerScore
)

CHECK_SEASONS = [CURRENT_SEASON] + get_past_seasons(3)


def fixture_string(fixture, result=None):
    """Get a string with basic info about a fixture.
    
    Arguments:
        fixture {SQLAlchemy class object} -- fixture from the database.
        result {SQLAlchemy class object} -- result from the database. If given
        returned string contains the match score.
    
    Returns:
        [string] -- formatted string with id, season, gameweek, home team and
        away team.
    """
    
    if result:
        return "{} GW{} {} {}-{} {} (id {})".format(fixture.season,
                                                    fixture.gameweek,
                                                    fixture.home_team,
                                                    result.home_score,
                                                    result.away_score,
                                                    fixture.away_team,
                                                    fixture.fixture_id)

    else:
        return "{} GW{} {} vs {} (id {})".format(fixture.season,
                                                 fixture.gameweek,
                                                 fixture.home_team,
                                                 fixture.away_team,
                                                 fixture.fixture_id)

    
def season_num_teams(seasons=CHECK_SEASONS):
    """Check whether each season has 20 teams.
    
    Keyword Arguments:
        seasons {list} -- seasons to check (default: {CHECK_SEASONS})
    """
    print("Checking seasons have 20 teams...")
    for season in seasons:
        teams = get_teams_for_season(season)
        assert len(teams) == 20, \
            "Number of teams in {} season is {} (not 20)"\
            .format(season, len(teams))
    print("OK!")


def season_num_new_teams(seasons=CHECK_SEASONS):
    """Check each season has 3 new teams.
    
    Keyword Arguments:
        seasons {list} -- seasons to check (default: {CHECK_SEASONS})
    """
    print("Checking seasons have 3 new teams...")
    teams = [get_teams_for_season(season) for season in seasons]
    for i in range(1, len(teams)):
        new_teams = [team for team in teams[i] if team not in teams[i-1]]
        assert len(new_teams) == 3, \
            "Number of teams changed between {} and {} is {} (not 3)"\
            .format(seasons[i-1], seasons[i], len(new_teams))
    print("OK!")


def season_num_fixtures(seasons=CHECK_SEASONS, session=session):
    """Check each season has 380 fixtures.
    
    Keyword Arguments:
        seasons {list} -- seasons to check (default: CHECK_SEASONS)
        session {SQLAlchemy session} -- DB session (default:
        airsenal.framework.schema.session)
    """
    print("Checking seasons have 380 fixtures...")
    for season in seasons:
        fixtures = get_fixtures_for_season(season=season)
        assert len(fixtures) == 380, \
            "Number of fixtures in {} season is {} (not 380)"\
            .format(season, len(fixtures))
    print("OK!")


def fixture_player_teams(seasons=CHECK_SEASONS, session=session):
    """Check players who played in a match are labelled as playing for either
    the home team or the away team.
    
    Keyword Arguments:
        seasons {[type]} -- seasons to check (default: {CHECK_SEASONS})
        session {SQLAlchemy session} -- DB session (default:
        airsenal.framework.schema.session)
    """
    print("Checking player teams match fixture teams...")
    for season in seasons:
        fixtures = get_fixtures_for_season(season=season)
                        
        for fixture in fixtures:
            player_scores = get_player_scores_for_fixture(fixture)

            for score in player_scores:
                assert ((score.player_team == fixture.home_team) or
                        (score.player_team == fixture.away_team)), \
                    """{}:
                {} in player_scores but labelled as playing for {}."""\
                    .format(fixture_string(fixture),
                            score.player.name, score.player_team)
    print("OK!")


def fixture_num_players(seasons=CHECK_SEASONS, session=session):
    """Check each fixture has between 11 and 14 players  with at least 1 minute
    in player_scores.
    
    Keyword Arguments:
        seasons {[type]} -- seasons to check (default: {CHECK_SEASONS})
        session {SQLAlchemy session} -- DB session (default:
        airsenal.framework.schema.session)
    """
    print("Checking 11 to 14 players play per team in each fixture...")
    for season in seasons:
        fixtures = get_fixtures_for_season(season=season)
                        
        for fixture in fixtures:
            result = get_result_for_fixture(fixture)

            if result:
                result = result[0]
                home_scores = session.query(PlayerScore)\
                                     .filter_by(fixture=fixture,
                                                player_team=fixture.home_team)\
                                     .filter(PlayerScore.minutes > 0)\
                                     .all()
 
                away_scores = session.query(PlayerScore)\
                                     .filter_by(fixture=fixture,
                                                player_team=fixture.away_team)\
                                     .filter(PlayerScore.minutes > 0)\
                                     .all()
                                     
                assert ((len(home_scores) > 10) and (len(home_scores) < 15)), \
                    """{}:
                {} players with minutes > 0 for home team.
                    """.format(fixture_string(fixture, result),
                               len(home_scores))
                
                assert ((len(away_scores) > 10) and (len(away_scores) < 15)), \
                    """{}:
                {} players with minutes > 0 for away team.
                    """.format(fixture_string(fixture, result),
                               len(away_scores))
    print("OK!")


def fixture_num_goals(seasons=CHECK_SEASONS, session=session):
    """Check individual player goals sum to match result for each fixture.
    
    Keyword Arguments:
        seasons {[type]} -- seasons to check (default: {CHECK_SEASONS})
        session {SQLAlchemy session} -- DB session (default:
        airsenal.framework.schema.session)
    """
    print("Checking sum of player goals match results...")
    for season in seasons:
        fixtures = get_fixtures_for_season(season=season)
                        
        for fixture in fixtures:
            result = get_result_for_fixture(fixture)

            if result:
                result = result[0]
                home_scores = session.query(PlayerScore)\
                                     .filter_by(fixture=fixture,
                                                player_team=fixture.home_team)\
                                     .all()
 
                away_scores = session.query(PlayerScore)\
                                     .filter_by(fixture=fixture,
                                                player_team=fixture.away_team)\
                                     .all()
                
                home_goals = sum([score.goals for score in home_scores]) \
                    + sum([score.own_goals for score in away_scores])
                             
                away_goals = sum([score.goals for score in away_scores]) \
                    + sum([score.own_goals for score in home_scores])
             
                assert home_goals == result.home_score, \
                    """{}:
                Player scores sum to {} but {} goals in result for home team"""\
                    .format(fixture_string(fixture, result),
                            home_goals, result.home_score)
                       
                assert away_goals == result.away_score, \
                    """{}:
                Player scores sum to {} but {} goals in result for home team"""\
                    .format(fixture_string(fixture, result),
                            away_goals, result.away_score)
    print("OK!")


def run_all_checks(seasons=CHECK_SEASONS):
    season_num_teams(seasons)
    season_num_new_teams(seasons)
    season_num_fixtures(seasons)
    fixture_player_teams(seasons)
    fixture_num_players(seasons)
    fixture_num_goals(seasons)


if __name__ == '__main__':
    run_all_checks()
