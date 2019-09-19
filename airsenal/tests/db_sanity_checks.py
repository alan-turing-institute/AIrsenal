from airsenal.framework.utils import *
from airsenal.framework.schema import *

CHECK_SEASONS = [CURRENT_SEASON] + get_past_seasons(3)


def season_num_teams(seasons=CHECK_SEASONS):
    for season in seasons:
        teams = get_teams_for_season(season)
        assert len(teams) == 20, \
            "Number of teams in {} season is {} (not 20)"\
            .format(season, len(teams))


def season_num_new_teams(seasons=CHECK_SEASONS):
    teams = [get_teams_for_season(season) for season in seasons]
    for i in range(1, len(teams)):
        new_teams = [team for team in teams[i] if team not in teams[i-1]]
        assert len(new_teams) == 3, \
            "Number of teams changed between {} and {} is {} (not 3)"\
            .format(seasons[i-1], seasons[i], len(new_teams))


def season_num_fixtures(seasons=CHECK_SEASONS, session=session):
    for season in seasons:
        fixtures = session.query(Fixture)\
                        .filter_by(season=season)\
                        .all()
        assert len(fixtures) == 380, \
            "Number of fixtures in {} season is {} (not 380)"\
            .format(season, len(fixtures))
               

if __name__ == '__main__':
    season_num_teams()
    season_num_new_teams()
    season_num_fixtures()
