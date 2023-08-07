"""
Functions used by the AIrsenal API
"""
from flask import jsonify
from sqlalchemy.orm import scoped_session

from airsenal.framework.optimization_transfers import (
    make_optimum_double_transfer,
    make_optimum_single_transfer,
)
from airsenal.framework.schema import Player, SessionBudget, SessionSquad, session
from airsenal.framework.squad import Squad
from airsenal.framework.utils import (
    CURRENT_SEASON,
    NEXT_GAMEWEEK,
    fetcher,
    get_fixtures_for_player,
    get_last_finished_gameweek,
    get_latest_prediction_tag,
    get_next_fixture_for_player,
    get_player,
    get_predicted_points_for_player,
    get_recent_scores_for_player,
    list_players,
    list_teams,
)

DBSESSION = scoped_session(session)


def remove_db_session(dbsession=DBSESSION):
    dbsession.remove()


def create_response(orig_response, dbsession=DBSESSION):
    """
    Add headers to the response
    """
    response = jsonify(orig_response)
    response.headers.add(
        "Access-Control-Allow-Headers",
        "Origin, X-Requested-With, Content-Type, Accept, x-auth",
    )
    return response


def reset_session_squad(session_id, dbsession=DBSESSION):
    """
    remove any rows with the given session ID and add a new budget of
    100M
    """
    # remove all players with this session id
    dbsession.query(SessionSquad).filter_by(session_id=session_id).delete()
    dbsession.commit()
    # now remove the budget, and make a new one
    dbsession.query(SessionBudget).filter_by(session_id=session_id).delete()
    dbsession.commit()
    sb = SessionBudget(session_id=session_id, budget=1000)
    dbsession.add(sb)
    dbsession.commit()
    return True


def list_players_for_api(team, position, dbsession=DBSESSION):
    """
    List players.  Just pass on to utils.list_players but
    specify the dbsession.
    """
    return list_players(team=team, position=position, dbsession=dbsession)


def list_teams_for_api(dbsession=DBSESSION):
    """
    List teams.  Just pass on to utils.list_teams but
    specify the season and  dbsession.
    """
    all_teams = [{"name": "all", "full_name": "all"}]
    all_teams += list_teams(season=CURRENT_SEASON, dbsession=dbsession)
    return all_teams


def combine_player_info(player_id, dbsession=DBSESSION):
    """
    Get player's name, club, recent scores, upcoming fixtures, and
    upcoming predictions if available
    """
    info_dict = {"player_id": player_id}
    p = get_player(player_id, dbsession=dbsession)
    info_dict["player_name"] = p.name
    team = p.team(CURRENT_SEASON, NEXT_GAMEWEEK)
    info_dict["team"] = team
    # get recent scores for the player
    rs = get_recent_scores_for_player(p, dbsession=dbsession)
    recent_scores = [{"gameweek": k, "score": v} for k, v in rs.items()]
    info_dict["recent_scores"] = recent_scores
    # get upcoming fixtures
    fixtures = get_fixtures_for_player(p, dbsession=dbsession)[:3]
    info_dict["fixtures"] = []
    for f in fixtures:
        home_or_away = "home" if f.home_team == team else "away"
        opponent = f.away_team if home_or_away == "home" else f.home_team
        info_dict["fixtures"].append(
            {"gameweek": f.gameweek, "opponent": opponent, "home_or_away": home_or_away}
        )
    try:
        tag = get_latest_prediction_tag(dbsession=dbsession)
        predicted_points = get_predicted_points_for_player(p, tag, dbsession=dbsession)
        info_dict["predictions"] = predicted_points
    except RuntimeError:
        pass
    return info_dict


def add_session_player(player_id, session_id, dbsession=DBSESSION):
    """
    Add a row in the SessionSquad table.
    """
    pids = [p["id"] for p in get_session_players(session_id, dbsession)]
    if player_id in pids:  # don't add the same player twice!
        return False
    st = SessionSquad(session_id=session_id, player_id=player_id)
    dbsession.add(st)
    dbsession.commit()
    return True


def remove_session_player(player_id, session_id, dbsession=DBSESSION):
    """
    Remove row from SessionSquad table.
    """
    pids = [p["id"] for p in get_session_players(session_id, dbsession)]
    player_id = int(player_id)
    if player_id not in pids:  # player not there
        return False
    (
        dbsession.query(SessionSquad)
        .filter_by(session_id=session_id, player_id=player_id)
        .delete()
    )
    dbsession.commit()
    return True


def list_players_teams_prices(
    position="all", team="all", dbsession=DBSESSION, gameweek=NEXT_GAMEWEEK
):
    """
    Return a list of players, each with their current team and price
    """
    return [
        (
            f"{p.name} "
            f"({p.team(CURRENT_SEASON, NEXT_GAMEWEEK)}): "
            f"{p.price(CURRENT_SEASON, NEXT_GAMEWEEK)}"
        )
        for p in list_players(
            position=position, team=team, dbsession=dbsession, gameweek=gameweek
        )
    ]


def get_session_budget(session_id, dbsession=DBSESSION):
    """
    query the sessionbudget table in the db - there should hopefully
    be one and only one row for this session_id
    """

    sb = dbsession.query(SessionBudget).filter_by(session_id=session_id).all()
    if len(sb) != 1:
        raise RuntimeError(f"{len(sb)}  SessionBudgets for session key {session_id}")
    return sb[0].budget


def set_session_budget(budget, session_id, dbsession=DBSESSION):
    """
    delete the existing entry for this session_id in the sessionbudget table,
    then enter a new row
    """
    print("Deleting old budget")
    dbsession.query(SessionBudget).filter_by(session_id=session_id).delete()
    dbsession.commit()
    print(f"Setting budget for {session_id} to {budget}")
    sb = SessionBudget(session_id=session_id, budget=budget)
    dbsession.add(sb)
    dbsession.commit()
    return True


def get_session_players(session_id, dbsession=DBSESSION):
    """
    query the dbsession for the list of players with the requested player_id
    """
    players = dbsession.query(SessionSquad).filter_by(session_id=session_id).all()
    return [
        {
            "id": p.player_id,
            "name": dbsession.query(Player)
            .filter_by(player_id=p.player_id)
            .first()
            .name,
        }
        for p in players
    ]


def validate_session_squad(session_id, dbsession=DBSESSION):
    """
    get the list of player_ids for this session_id, and see if we can
    make a valid 15-player squad out of it
    """
    budget = get_session_budget(session_id, dbsession)

    players = get_session_players(session_id, dbsession)
    if len(players) != 15:
        return False
    t = Squad(budget)
    for p in players:
        added_ok = t.add_player(p["id"], dbsession=dbsession)
        if not added_ok:
            return False
    return True


def fill_session_squad(team_id, session_id, dbsession=DBSESSION):
    """
    Use the FPL API to get list of players in an FPL squad with id=team_id,
    then fill the session squad with these players.
    """
    # first reset the squad
    reset_session_squad(session_id, dbsession)
    # now query the API
    players = fetcher.get_fpl_team_data(get_last_finished_gameweek(), team_id)["picks"]
    player_ids = [p["element"] for p in players]
    for pid in player_ids:
        add_session_player(pid, session_id, dbsession)
    team_history = fetcher.get_fpl_team_history_data()["current"]
    index = (
        get_last_finished_gameweek() - 1
    )  # as gameweek starts counting from 1 but list index starts at 0
    budget = team_history[index]["value"]
    set_session_budget(budget, session_id)
    return player_ids


def get_session_prediction(
    player_id, session_id, gw=None, pred_tag=None, dbsession=DBSESSION
):
    """
    Query the fixture and predictedscore tables for a specified player
    """
    if not gw:
        gw = NEXT_GAMEWEEK
    if not pred_tag:
        pred_tag = get_latest_prediction_tag()
    return {
        "predicted_score": get_predicted_points_for_player(
            player_id, pred_tag, CURRENT_SEASON, dbsession
        )[gw],
        "fixture": get_next_fixture_for_player(player_id, CURRENT_SEASON, dbsession),
    }


def get_session_predictions(session_id, dbsession=DBSESSION):
    """
    Query the fixture and predictedscore tables for all
    players in our session squad
    """
    pids = [p["id"] for p in get_session_players(session_id, dbsession)]
    pred_tag = get_latest_prediction_tag()
    gw = NEXT_GAMEWEEK
    return {
        pid: get_session_prediction(pid, session_id, gw, pred_tag, dbsession)
        for pid in pids
    }


def best_transfer_suggestions(n_transfer, session_id, dbsession=DBSESSION):
    """
    Use our predicted playerscores to suggest the best transfers.
    """
    n_transfer = int(n_transfer)
    if n_transfer not in range(1, 3):
        raise RuntimeError("Need to choose 1 or 2 transfers")
    if not validate_session_squad(session_id, dbsession):
        raise RuntimeError("Cannot suggest transfer without complete squad")

    budget = get_session_budget(session_id, dbsession)
    players = [p["id"] for p in get_session_players(session_id, dbsession)]
    t = Squad(budget)
    for p in players:
        added_ok = t.add_player(p)
        if not added_ok:
            raise RuntimeError(f"Cannot add player {p}")
    pred_tag = get_latest_prediction_tag()
    if n_transfer == 1:
        _, pid_out, pid_in = make_optimum_single_transfer(t, pred_tag)
    elif n_transfer == 2:
        _, pid_out, pid_in = make_optimum_double_transfer(t, pred_tag)
    return {"transfers_out": pid_out, "transfers_in": pid_in}
