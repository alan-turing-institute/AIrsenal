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

from bpl import ExtendedDixonColesMatchPredictor, NeutralDixonColesMatchPredictor
from pandas import Series
from sqlalchemy.orm.session import Session

from airsenal.framework.bpl_interface import (
    DEFAULT_TEAM_EPSILON,
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
from airsenal.framework.random_team_model import RandomMatchPredictor
from airsenal.framework.schema import session, session_scope
from airsenal.framework.utils import (
    CURRENT_SEASON,
    NEXT_GAMEWEEK,
    get_fixtures_for_gameweek,
    get_gameweeks_array,
    get_top_predicted_points,
    list_players,
)


def allocate_predictions(
    queue: Queue,
    gw_range: list[int],
    fixture_goal_probs: dict,
    df_player: dict,
    df_bonus: tuple,
    df_saves: Series,
    df_cards: Series,
    season: str,
    tag: str,
    dbsession: Session,
) -> None:
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
    gw_range: list[int],
    season: str,
    dbsession: Session,
    include_bonus: bool = True,
    include_cards: bool = True,
    include_saves: bool = True,
    num_thread: int = 4,
    tag: str = "",
    player_model: NumpyroPlayerModel | ConjugatePlayerModel | None = None,
    team_model: ExtendedDixonColesMatchPredictor
    | NeutralDixonColesMatchPredictor
    | RandomMatchPredictor
    | None = None,
    team_model_args: dict | None = None,
) -> None:
    """
    Do the full prediction for players.
    """
    if team_model_args is None:
        team_model_args = {"epsilon": DEFAULT_TEAM_EPSILON}
    model_team = get_fitted_team_model(
        season=season,
        gameweek=min(gw_range),
        dbsession=dbsession,
        model=team_model,
        **team_model_args,
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

    if num_thread > 1:
        queue: Queue = Queue()
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

        for _, pr in enumerate(procs):
            pr.join()
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
            for pred in predictions:
                dbsession.add(pred)
        dbsession.commit()
        print("Finished adding predictions to db")


def make_predictedscore_table(
    gw_range: list[int] | None = None,
    season: str = CURRENT_SEASON,
    num_thread: int = 4,
    include_bonus: bool = True,
    include_cards: bool = True,
    include_saves: bool = True,
    tag_prefix: str | None = None,
    player_model: NumpyroPlayerModel | ConjugatePlayerModel | None = None,
    team_model: ExtendedDixonColesMatchPredictor
    | NeutralDixonColesMatchPredictor
    | RandomMatchPredictor
    | None = None,
    team_model_args: dict | None = None,
    dbsession: Session = session,
) -> str:
    if team_model_args is None:
        team_model_args = {"epsilon": DEFAULT_TEAM_EPSILON}
    tag = tag_prefix or ""
    tag += str(uuid4())
    if not gw_range:
        gw_range = list(range(NEXT_GAMEWEEK, NEXT_GAMEWEEK + 3))
    calc_all_predicted_points(
        gw_range=gw_range,
        season=season,
        dbsession=dbsession,
        include_bonus=include_bonus,
        include_cards=include_cards,
        include_saves=include_saves,
        num_thread=num_thread,
        tag=tag,
        player_model=player_model,
        team_model=team_model,
        team_model_args=team_model_args,
    )
    return tag


def main():
    """
    fill the player_prediction db table
    """
    parser = argparse.ArgumentParser(description="fill player predictions")
    parser.add_argument("--weeks_ahead", help="how many weeks ahead to fill", type=int)
    parser.add_argument("--gameweek_start", help="first gameweek to look at", type=int)
    parser.add_argument("--gameweek_end", help="last gameweek to look at", type=int)
    parser.add_argument("--ep_filename", help="csv filename for FPL expected points")
    parser.add_argument(
        "--season", help="season, in format e.g. '1819'", default=CURRENT_SEASON
    )
    parser.add_argument(
        "--num_thread",
        help="number of threads to parallelise over",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--no_bonus",
        help="don't include bonus points",
        action="store_true",
    )
    parser.add_argument(
        "--no_cards",
        help="don't include points lost to yellow and red cards",
        action="store_true",
    )
    parser.add_argument(
        "--no_saves",
        help="don't include save points for goalkeepers",
        action="store_true",
    )
    parser.add_argument(
        "--sampling",
        help="If set use fit the model using sampling with numpyro",
        action="store_true",
    )
    parser.add_argument(
        "--team_model",
        help="which team model to fit",
        type=str,
        choices=["extended", "neutral", "random"],
        default="extended",
    )
    parser.add_argument(
        "--epsilon",
        help="how much to downweight games by in exponential time weighting",
        type=float,
        default=DEFAULT_TEAM_EPSILON,
    )

    args = parser.parse_args()
    gw_range = get_gameweeks_array(
        weeks_ahead=args.weeks_ahead,
        gameweek_start=args.gameweek_start,
        gameweek_end=args.gameweek_end,
        season=args.season,
    )
    num_thread: int = args.num_thread
    include_bonus = not args.no_bonus
    include_cards = not args.no_cards
    include_saves = not args.no_saves
    player_model = NumpyroPlayerModel() if args.sampling else ConjugatePlayerModel()
    if args.team_model == "extended":
        team_model = ExtendedDixonColesMatchPredictor()
    elif args.team_model == "neutral":
        team_model = NeutralDixonColesMatchPredictor()
    elif args.team_model == "random":
        team_model = RandomMatchPredictor()
    else:
        msg = f"Unknown team model: {args.team_model}"
        raise ValueError(msg)

    set_multiprocessing_start_method()

    with session_scope() as session:
        session.expire_on_commit = False

        tag = make_predictedscore_table(
            gw_range=gw_range,
            season=args.season,
            num_thread=num_thread,
            include_bonus=include_bonus,
            include_cards=include_cards,
            include_saves=include_saves,
            player_model=player_model,
            team_model=team_model,
            team_model_args={"epsilon": args.epsilon},
            dbsession=session,
        )

        # print players with top predicted points
        get_top_predicted_points(
            gameweek=gw_range,
            tag=tag,
            season=args.season,
            per_position=True,
            n_players=5,
            dbsession=session,
        )
