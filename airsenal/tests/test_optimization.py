"""
Test the optimization of transfers, generating a few simplified scenarios
and checking that the optimizer finds the expected outcome.
"""
import pytest
from unittest import mock
from operator import itemgetter

#from ..utils import get_predicted_points

from ..framework.optimization_utils import Team, make_optimum_transfer, make_optimum_double_transfer, get_predicted_points



def mock_add_player(pid):
    self.players += DummyPlayer(pid,"FWD",{})
    print("Mocking add player {}".format(pid))


class DummyPlayer(object):
    """
    fake player that we can add to a team, giving a specified expected score.
    """
    def __init__(self,player_id,position,points_dict):
        """
        we generate team to avoid >3-players-per-team problem,
        and set price to 0 to avoid overrunning budget.
        """
        self.player_id = player_id
        self.name = "player_{}".format(player_id)
        self.position = position
        self.team = "DUMMY_TEAM_{}".format(player_id)
        self.current_price = 0
        self.is_starting = True
        self.is_captain = False
        self.is_vice_captain = False
        self.predicted_points = {"DUMMY": points_dict}
        self.sub_position = None

    def calc_predicted_points(self,dummy):
        pass


def generate_dummy_team(player_points_dict=None):
    """
    Fill a team up with dummy players.
    player_points_dict is a dictionary
    { player_id: { gw: points,...} ,...}
    """
    if not player_points_dict:  # make a simple one
        player_points_dict = {}
        for i in range(15):
            player_points_dict[i] = {1: 2} # 2 points per game
    t=Team()
    for i in range(15):
        if i<2:
            position="GK"
        elif i<7:
            position="DEF"
        elif i<12:
            position="MID"
        else:
            position="FWD"
        t.add_player(DummyPlayer(i,position,player_points_dict[i]))
    return t


def predicted_point_mock_generator(point_dict):
    """
    return a function that will mock the get_predicted_points function
    the point_dict it is given should be keyed by position, i.e.
    {"GK" : {player_id: points, ...}, "DEF": {}, ... }
    """

    def mock_get_predicted_points(gameweek, tag, position, team=None):
        """
        return an ordered list in the same way as the real
        get_predicted_points func does. EXCEPT - we return dummy players rather than just ids
        (so the Team.add_player can add them)
        """
        output_pid_list = [(k,v) for k,v in point_dict[position].items()]
        output_pid_list.sort(key=itemgetter(1), reverse=True)
#        return output_pid_list
        if isinstance(gameweek, list):
            gameweek = gameweek[0]
        output_list = [(DummyPlayer(entry[0],position,{gameweek:entry[1]}),entry[1]) for entry in output_pid_list]
        return output_list
    return mock_get_predicted_points



def test_subs():
    """
    mock squads with some players predicted some points, and
    some predicted to score zero, and check we get the right starting 11.
    """
    points_dict = { 0: {1:0},
                    1: {1:2},
                    2: {1:2},
                    3: {1:2},
                    4: {1:0},
                    5: {1:2},
                    6: {1:2},
                    7: {1:2},
                    8: {1:2},
                    9: {1:0},
                    10: {1:2},
                    11: {1:4},
                    12: {1:0},
                    13: {1:2},
                    14: {1:3}
                    }
    ## should get 4,4,2, with players 0,4,9,12 on the bench,
    ## captain player 11, vice-captain player 14
    ## should have 29 points (9*2 + 3 + (2*4) )
    t = generate_dummy_team(points_dict)
    ep = t.get_expected_points(1,"DUMMY")
    assert(ep==29)
    assert(t.players[0].is_starting==False)
    assert(t.players[4].is_starting==False)
    assert(t.players[9].is_starting==False)
    assert(t.players[12].is_starting==False)
    assert(t.players[11].is_captain==True)
    assert(t.players[14].is_vice_captain==True)



def test_single_transfer():
    """
    mock squad with all players predicted 2 points, and potential transfers
    with higher scores, check we get the best transfer.
    """
    t = generate_dummy_team()
    position_points_dict = {"GK": {0:2, 1:2,   # in the orig team
                                   100:0, 101: 0,
                                   200:3, 201:2},
                            "DEF": {2: 2, 3: 2, 4: 2, 5:2, 6:2,# in the orig team
                                    103:0, 104:0, 105:5, 106:2, 107:2,
                                    203:0, 204:0, 205:1, 206:2, 207:2},
                            "MID": {7:2, 8:2, 9:2, 10:2, 11:2, # in the orig team
                                    108:2, 109:2, 110:3, 111:3, 112:0,
                                    208:2, 209:2, 210:3, 211:3, 212:0},
                            "FWD": {12:2, 13:2, 14:2, # in the orig team
                                    113: 6, 114:3, 115:7}
                            }
    mock_pred_points= predicted_point_mock_generator(position_points_dict)


    with mock.patch('airsenal.framework.optimization_utils.get_predicted_points', side_effect=mock_pred_points):
        new_team, pid_out, pid_in = make_optimum_transfer(t,"DUMMY",[1])
        ## we should expect - player 115 to be transfered in, and to be captain.
    assert(pid_in[0]==115)
    for p in new_team.players:
        if p.player_id == 115:
            assert(p.is_captain==True)
        else:
            assert(p.is_captain==False)
    ## expected points should be 10*2 + 7*2 = 34
    assert(new_team.get_expected_points(1,"DUMMY")==34)


def test_double_transfer():
    """
    mock squad with two players predicted low score, see if we get better players
    transferred in.
    """
    t=generate_dummy_team()
    position_points_dict = {"GK": {0:2, 1:2,   # in the orig team
                                   100:0, 101: 0,
                                   200:3, 201:7},
                            "DEF": {2:2, 3: 2, 2: 2, 5:2, 6:2,# in the orig team
                                    103:0, 104:0, 105:5, 106:2, 107:2,
                                    203:0, 204:0, 205:1, 206:2, 207:2},
                            "MID": {7:2, 8:2, 9:2, 10: 2, 11:2, # in the orig team
                                    108:2, 109:2, 110:3, 111:3, 112:0,
                                    208:2, 209:2, 210:3, 211:3, 212:0},
                            "FWD": {12:2, 13:2, 14:2, # in the orig team
                                    113: 6, 114:3, 115:8 }
                            }
    mock_pred_points= predicted_point_mock_generator(position_points_dict)


    with mock.patch('airsenal.framework.optimization_utils.get_predicted_points', side_effect=mock_pred_points):
        new_team, pid_out, pid_in = make_optimum_double_transfer(t,"DUMMY",[1])
        ## we should expect 201 and 115 to be transferred in, and 1,15 to
        ## be transferred out.   115 should be captain
        assert(201 in pid_in)
        assert(115 in pid_in)
        print(new_team)
        for p in new_team.players:
            if p.player_id == 115:
                assert(p.is_captain==True)
            else:
                assert(p.is_captain==False)
