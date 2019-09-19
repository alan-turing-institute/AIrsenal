from airsenal.framework.utils import *
from airsenal.framework.schema import *

CHECK_SEASONS = [CURRENT_SEASON] + get_past_seasons(3)


def season_num_teams(seasons=CHECK_SEASONS):
    """Check whether each season has 20 teams.
    
    Keyword Arguments:
        seasons {list} -- seasons to check (default: {CHECK_SEASONS})
    """
    for season in seasons:
        teams = get_teams_for_season(season)
        assert len(teams) == 20, \
            "Number of teams in {} season is {} (not 20)"\
            .format(season, len(teams))


def season_num_new_teams(seasons=CHECK_SEASONS):
    """Check each season has 3 new teams.
    
    Keyword Arguments:
        seasons {list} -- seasons to check (default: {CHECK_SEASONS})
    """
    teams = [get_teams_for_season(season) for season in seasons]
    for i in range(1, len(teams)):
        new_teams = [team for team in teams[i] if team not in teams[i-1]]
        assert len(new_teams) == 3, \
            "Number of teams changed between {} and {} is {} (not 3)"\
            .format(seasons[i-1], seasons[i], len(new_teams))


def season_num_fixtures(seasons=CHECK_SEASONS, session=session):
    """Check each season has 380 fixtures.
    
    Keyword Arguments:
        seasons {list} -- seasons to check (default: CHECK_SEASONS)
        session {SQLAlchemy session} -- DB session (default:
        airsenal.framework.schema.session)
    """
    for season in seasons:
        fixtures = session.query(Fixture)\
                          .filter_by(season=season)\
                          .all()
        assert len(fixtures) == 380, \
            "Number of fixtures in {} season is {} (not 380)"\
            .format(season, len(fixtures))


def fixture_num_goals(seasons=CHECK_SEASONS, session=session):
    """Check individual player goals sum to match result for each fixture.
    
    Keyword Arguments:
        seasons {[type]} -- [description] (default: {CHECK_SEASONS})
        session {SQLAlchemy session} -- DB session (default:
        airsenal.framework.schema.session)
    """
    for season in seasons:
        fixtures = session.query(Fixture)\
                          .filter_by(season=season)\
                          .all()
                        
        for fixture in fixtures:
            result = session.query(Result)\
                            .filter_by(fixture=fixture)\
                            .all()
            if result:
                home_scores = session.query(PlayerScore)\
                                     .filter_by(fixture=fixture)\
                                     .filter(PlayerScore.minutes > 0)\
                                     .filter_by(player_team=fixture.home_team)\
                                     .all()
 
                away_scores = session.query(PlayerScore)\
                                     .filter_by(fixture=fixture)\
                                     .filter(PlayerScore.minutes > 0)\
                                     .filter_by(player_team=fixture.away_team)\
                                     .all()
                
                home_goals = sum([score.goals for score in home_scores]) \
                    + sum([score.own_goals for score in away_scores])
                             
                away_goals = sum([score.goals for score in away_scores]) \
                    + sum([score.own_goals for score in home_scores])

                assert home_goals == result[0].home_score, \
                    """{} GW{} {} {} - {} {}:
                Player scores sum to {} but {} goals in result for home team"""\
                    .format(season, fixture.gameweek,
                            fixture.home_team, result[0].home_score,
                            result[0].away_score, fixture.away_team,
                            home_goals, result[0].home_score)
                         
                assert away_goals == result[0].away_score, \
                    """{}: GW{} {} {}-{} {}:
                Player scores sum to {} but {} goals in result for away team"""\
                    .format(season, fixture.gameweek,
                            fixture.home_team, result[0].home_score,
                            result[0].away_score, fixture.away_team,
                            away_goals, result[0].away_score)
                   

if __name__ == '__main__':
    season_num_teams()
    season_num_new_teams()
    season_num_fixtures()
    fixture_num_goals()
