#!/usr/bin/env python

"""
Fill the "player_prediction" table with score predictions
Usage:
python fill_predictedscore_table.py --weeks_ahead <nweeks>
Generates a "tag" string which is stored so it can later be used by team-optimizers to
get consistent sets of predictions from the database.
"""
import argparse
from multiprocessing import Process, Queue
from uuid import uuid4

from airsenal.framework.bpl_interface import (
    get_fitted_team_model,
    get_goal_probabilities_for_fixtures,
)
from airsenal.framework.multiprocessing_utils import set_multiprocessing_start_method
from airsenal.framework.player_model import ConjugatePlayerModel, NumpyroPlayerModel
from airsenal.framework.prediction_utils import (
    MAX_GOALS,
    calc_predicted_points_for_player,
    fit_bonus_points,
    fit_card_points,
    fit_save_points,
    get_all_fitted_player_data,
)
from airsenal.framework.schema import session_scope
from airsenal.framework.utils import (
    CURRENT_SEASON,
    NEXT_GAMEWEEK,
    get_fixtures_for_gameweek,
    get_gameweeks_array,
    get_top_predicted_points,
    list_players,
)


def allocate_predictions(
    queue,
    gw_range,
    fixture_goal_probs,
    df_player,
    df_bonus,
    df_saves,
    df_cards,
    season,
    tag,
    dbsession,
):
    """
    Take positions off the queue and call function to calculate predictions
    """
    while True:
        player = queue.get()
        if player == "DONE":
            print("Finished processing")
            break

        predictions = calc_predicted_points_for_player(
            player,
            fixture_goal_probs,
            df_player,
            df_bonus,
            df_saves,
            df_cards,
            season,
            gw_range=gw_range,
            tag=tag,
            dbsession=dbsession,
        )
        for p in predictions:
            dbsession.add(p)
        dbsession.commit()


def calc_all_predicted_points(
    gw_range,
    season,
    include_bonus=True,
    include_cards=True,
    include_saves=True,
    num_thread=4,
    tag="",
    player_model=ConjugatePlayerModel(),
    dbsession=None,
):
    """
    Do the full prediction for players.
    """
    model_team = get_fitted_team_model(
        season, gameweek=min(gw_range), dbsession=dbsession
    )
    print("Calculating fixture score probabilities...")
    fixtures = get_fixtures_for_gameweek(gw_range, season=season, dbsession=dbsession)
    fixture_goal_probs = get_goal_probabilities_for_fixtures(
        fixtures, model_team, max_goals=MAX_GOALS
    )

    df_player = get_all_fitted_player_data(
        season, gw_range[0], model=player_model, dbsession=dbsession
    )

    if include_bonus:
        df_bonus = fit_bonus_points(gameweek=gw_range[0], season=season)
    else:
        df_bonus = None
    if include_saves:
        df_saves = fit_save_points(gameweek=gw_range[0], season=season)
    else:
        df_saves = None
    if include_cards:
        df_cards = fit_card_points(gameweek=gw_range[0], season=season)
    else:
        df_cards = None

    players = list_players(season=season, gameweek=gw_range[0], dbsession=dbsession)

    if num_thread is not None and num_thread > 1:
        queue = Queue()
        procs = []
        for _ in range(num_thread):
            processor = Process(
                target=allocate_predictions,
                args=(
                    queue,
                    gw_range,
                    fixture_goal_probs,
                    df_player,
                    df_bonus,
                    df_saves,
                    df_cards,
                    season,
                    tag,
                    dbsession,
                ),
            )
            processor.daemon = True
            processor.start()
            procs.append(processor)

        for p in players:
            queue.put(p.player_id)
        for _ in range(num_thread):
            queue.put("DONE")

        for _, p in enumerate(procs):
            p.join()
    else:
        # single threaded
        for player in players:
            predictions = calc_predicted_points_for_player(
                player,
                fixture_goal_probs,
                df_player,
                df_bonus,
                df_saves,
                df_cards,
                season,
                gw_range=gw_range,
                tag=tag,
                dbsession=dbsession,
            )
            for p in predictions:
                dbsession.add(p)
        dbsession.commit()
        print("Finished adding predictions to db")


def make_predictedscore_table(
    gw_range=None,
    season=CURRENT_SEASON,
    num_thread=4,
    include_bonus=True,
    include_cards=True,
    include_saves=True,
    tag_prefix=None,
    player_model=ConjugatePlayerModel(),
    dbsession=None,
):
    tag = tag_prefix or ""
    tag += str(uuid4())
    if not gw_range:
        gw_range = list(range(NEXT_GAMEWEEK, NEXT_GAMEWEEK + 3))
    calc_all_predicted_points(
        gw_range,
        season,
        include_bonus=include_bonus,
        include_cards=include_cards,
        include_saves=include_saves,
        num_thread=num_thread,
        tag=tag,
        player_model=player_model,
        dbsession=dbsession,
    )
    return tag


def make_predictions(
    weeks_ahead: int,
    gameweek_start: int,
    gameweek_end: int,
    season: int,
    num_threads: int,
    no_bonus: bool,
    no_cards: bool,
    no_saves: bool,
    sampling: bool,
):
    """
    fill the player_prediction db table
    """
    if weeks_ahead and (gameweek_start or gameweek_end):
        print("Please specify either gameweek_start and gameweek_end, OR weeks_ahead")
        raise RuntimeError("Inconsistent arguments")
    if weeks_ahead and season != CURRENT_SEASON:
        print("For past seasons, please specify gameweek_start and gameweek_end")
        raise RuntimeError("Inconsistent arguments")
    if weeks_ahead:
        gw_range = get_gameweeks_array(weeks_ahead)
    elif gameweek_start and gameweek_end:
        gw_range = list(range(gameweek_start, gameweek_end))
    elif gameweek_start:  # by default go three weeks ahead
        gw_range = list(range(gameweek_start, gameweek_start + 3))
    else:
        gw_range = list(range(NEXT_GAMEWEEK, NEXT_GAMEWEEK + 3))
    num_threads = num_threads or None
    include_bonus = not no_bonus
    include_cards = not no_cards
    include_saves = not no_saves
    if sampling:
        player_model = NumpyroPlayerModel()
    else:
        player_model = ConjugatePlayerModel()

    set_multiprocessing_start_method(num_threads)

    with session_scope() as session:
        session.expire_on_commit = False

        tag = make_predictedscore_table(
            gw_range=gw_range,
            season=season,
            num_thread=num_threads,
            include_bonus=include_bonus,
            include_cards=include_cards,
            include_saves=include_saves,
            player_model=player_model,
            dbsession=session,
        )

        # print players with top predicted points
        get_top_predicted_points(
            gameweek=gw_range,
            tag=tag,
            season=season,
            per_position=True,
            n_players=5,
            dbsession=session,
        )
