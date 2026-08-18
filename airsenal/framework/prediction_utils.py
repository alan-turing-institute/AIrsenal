"""
Use the BPL models to predict scores for upcoming fixtures.
"""

import logging
import os
import uuid
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd
from scipy.stats import multinomial
from sqlalchemy import and_, select
from sqlalchemy.orm import selectinload
from sqlalchemy.orm.session import Session

from airsenal.framework.FPL_scoring_rules import (
    def_cons_required,
    get_appearance_points,
    points_for_assist,
    points_for_cs,
    points_for_def_cons,
    points_for_goal,
    points_for_red_card,
    points_for_yellow_card,
    saves_for_point,
)
from airsenal.framework.player_model import (
    DEFAULT_N_GOALS_PRIOR,
    DEFAULT_PLAYER_EPSILON,
    ConjugatePlayerModel,
    NumpyroPlayerModel,
    get_empirical_bayes_estimates,
)
from airsenal.framework.schema import (
    Absence,
    Fixture,
    Player,
    PlayerAttributes,
    PlayerPrediction,
    PlayerScore,
)
from airsenal.framework.utils import (
    CURRENT_SEASON,
    NEXT_GAMEWEEK,
    fastcopy,
    fetcher,
    get_fixtures_for_gameweek,
    get_fixtures_for_player,
    get_max_matches_per_player,
    get_player,
    get_player_from_api_id,
    get_recent_minutes_for_player,
    is_future_gameweek,
    list_players,
    session,
    was_historic_absence,
)

logger = logging.getLogger(__name__)

np.random.seed(42)

# Global Sabitler
MAX_GOALS = 10
MIN_MINUTES_SHORT = 30
MIN_MINUTES_FULL = 60
MAX_MINUTES_MATCH = 90


def check_absence(
    player: Player,
    gameweek: int,
    season: str,
    dbsession: Session = session,
) -> tuple[str | list[str] | None, str | list[str] | None]:
    """
    Query the Absence table for a given player and season to see if the
    gameweek is within the period of absence. If so, return the details of absence.
    """
    absence = dbsession.scalars(
        select(Absence).where(
            Absence.season == season,
            Absence.player_id == player.player_id,
            Absence.gw_from < gameweek,
            Absence.gw_until > gameweek,
        )
    ).all()

    reasons = [ab.reason for ab in absence] if len(absence) > 0 else None
    details = [ab.details for ab in absence] if len(absence) > 0 else None

    if reasons is not None:
        reasons = reasons[0] if len(reasons) == 1 else reasons
    if details is not None:
        details = details[0] if len(details) == 1 else details

    return reasons, details


def get_player_history_df(
    position="all",
    all_players=False,
    fill_blank=True,
    season=CURRENT_SEASON,
    gameweek=NEXT_GAMEWEEK,
    dbsession=session,
) -> pd.DataFrame:
    col_names = [
        "player_id",
        "player_name",
        "match_id",
        "date",
        "season",
        "gameweek",
        "goals",
        "assists",
        "minutes",
        "team_goals",
        "expected_goals",
        "expected_assists",
        "absence_reason",
        "absence_detail",
    ]
    player_data = []

    if all_players:
        q = dbsession.scalars(
            select(PlayerAttributes).options(selectinload(PlayerAttributes.player))
        )
        players = []
        seen_player_ids = set()
        for p in q:
            if p.player_id in seen_player_ids:
                continue
            seen_player_ids.add(p.player_id)
            players.append(p.player)
    else:
        players = list_players(
            position=position, season=season, gameweek=gameweek, dbsession=dbsession
        )

    player_ids = [p.player_id for p in players]
    scores_by_player = defaultdict(list)
    absences_by_player_season = defaultdict(list)

    if player_ids:
        all_scores = dbsession.scalars(
            select(PlayerScore)
            .options(
                selectinload(PlayerScore.fixture),
                selectinload(PlayerScore.result),
            )
            .where(PlayerScore.player_id.in_(player_ids))
        ).all()
        for score in all_scores:
            scores_by_player[score.player_id].append(score)

        score_seasons = {
            score.fixture.season
            for score in all_scores
            if score.fixture is not None and score.fixture.season is not None
        }
        if score_seasons:
            absences = dbsession.scalars(
                select(Absence)
                .where(
                    Absence.player_id.in_(player_ids),
                    Absence.season.in_(score_seasons),
                )
                .order_by(Absence.id)
            ).all()
            for absence in absences:
                if absence.player_id is None:
                    continue
                absences_by_player_season[(absence.player_id, absence.season)].append(
                    absence
                )

    max_matches_per_player = get_max_matches_per_player(
        position, season=season, gameweek=gameweek, dbsession=dbsession
    )

    for counter, player in enumerate(players):
        # Her adımda print etmek yerine logging ve periyodik bildirim kullanımı
        if counter % 50 == 0 or counter == len(players) - 1:
            logger.info(f"Filling history dataframe for {player}: {counter}/{len(players)} done")

        results = scores_by_player.get(player.player_id, [])
        row_count = 0
        for row in results:
            if is_future_gameweek(
                row.fixture.season,
                row.fixture.gameweek,
                current_season=season,
                next_gameweek=gameweek,
            ):
                continue

            match_id = row.result_id
            if not match_id:
                logger.warning(f"Couldn't find result for {row.fixture}")
                continue

            minutes = row.minutes
            goals = row.goals
            assists = row.assists
            match_result = row.result
            match_date = row.fixture.date

            if row.fixture.home_team == row.opponent:
                team_goals = match_result.away_score
            elif row.fixture.away_team == row.opponent:
                team_goals = match_result.home_score
            else:
                logger.warning("Unknown opponent!")
                team_goals = -1

            expected_goals = row.expected_goals
            expected_assists = row.expected_assists
            matching_absences = [
                ab
                for ab in absences_by_player_season.get(
                    (player.player_id, row.fixture.season), []
                )
                if ab.gw_until is not None
                and ab.gw_from < row.fixture.gameweek
                and ab.gw_until > row.fixture.gameweek
            ]
            absence_reason = (
                [ab.reason for ab in matching_absences] if matching_absences else None
            )
            absence_detail = (
                [ab.details for ab in matching_absences] if matching_absences else None
            )
            if absence_reason is not None and len(absence_reason) == 1:
                absence_reason = absence_reason[0]
            if absence_detail is not None and len(absence_detail) == 1:
                absence_detail = absence_detail[0]

            player_data.append(
                [
                    player.player_id,
                    player.name,
                    match_id,
                    match_date,
                    row.fixture.season,
                    row.fixture.gameweek,
                    goals,
                    assists,
                    minutes,
                    team_goals,
                    expected_goals,
                    expected_assists,
                    absence_reason,
                    absence_detail,
                ]
            )
            row_count += 1

        if fill_blank and row_count < max_matches_per_player:
            blank_row = [
                player.player_id,
                player.name,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                None, None,
            ]
            player_data.extend([list(blank_row) for _ in range(max_matches_per_player - row_count)])

    df = pd.DataFrame(player_data, columns=col_names)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.reset_index(drop=True, inplace=True)

    return df


def get_attacking_points(
    position: str,
    minutes: int | float,
    team_score_prob: dict[int, float],
    player_prob: pd.Series,
) -> float:
    if minutes == 0.0:
        return 0.0

    pr_score = (minutes / 90.0) * player_prob["prob_score"]
    pr_assist = (minutes / 90.0) * player_prob["prob_assist"]
    pr_neither = 1.0 - pr_score - pr_assist
    multinom_probs = (pr_score, pr_assist, pr_neither)

    def _get_partitions(n):
        partitions = []
        for i in range(n + 1):
            for j in range(n - i + 1):
                partitions.append([i, j, n - i - j])
        return partitions

    def _get_partition_score(partition):
        return (
            points_for_goal[position] * partition[0] + points_for_assist * partition[1]
        )

    exp_points = 0.0
    for ngoals, score_n_prob in team_score_prob.items():
        if ngoals > 0:
            partitions = _get_partitions(ngoals)
            probabilities = multinomial.pmf(
                partitions, n=[ngoals] * len(partitions), p=multinom_probs
            )
            scores = map(_get_partition_score, partitions)
            exp_score_inner = sum(
                pi * si for pi, si in zip(probabilities, scores, strict=False)
            )
            exp_points += exp_score_inner * score_n_prob
    return exp_points


def get_defending_points(
    position: str, minutes: int | float, team_concede_prob: dict[int, float]
) -> float:
    if position == "FWD" or minutes == 0.0:
        return 0.0

    defending_points = 0.0
    if minutes >= MIN_MINUTES_FULL:
        defending_points = points_for_cs[position] * team_concede_prob[0]

    if position in ["DEF", "GK"]:
        defending_points -= sum(
            (ngoals // 2) * (minutes / 90) * concede_n_prob
            for ngoals, concede_n_prob in team_concede_prob.items()
        )
    return defending_points


def get_bonus_points(
    player_id: int, minutes: int | float, df_bonus: tuple[pd.Series, pd.Series]
) -> float:
    """
    Sadeleştirilmiş ve güvenli bonus puan hesaplaması.
    """
    if minutes >= MIN_MINUTES_FULL:
        return float(df_bonus[0].get(player_id, 0.0))
    if minutes >= MIN_MINUTES_SHORT:
        return float(df_bonus[1].get(player_id, 0.0))
    return 0.0


def get_def_con_points(
    player_id: int, minutes: int | float, df_def_con: tuple[pd.Series, pd.Series]
) -> float:
    """
    Sadeleştirilmiş ve güvenli savunma katkısı puan hesaplaması.
    """
    if minutes >= MIN_MINUTES_FULL:
        return float(df_def_con[0].get(player_id, 0.0))
    if minutes >= MIN_MINUTES_SHORT:
        return float(df_def_con[1].get(player_id, 0.0))
    return 0.0


def get_save_points(
    position: str, player_id: int, minutes: int | float, df_saves: pd.Series
) -> float:
    if position != "GK":
        return 0.0
    if minutes >= MIN_MINUTES_FULL:
        return float(df_saves.get(player_id, 0.0))
    return 0.0


def get_card_points(player_id: int, minutes: int | float, df_cards: pd.Series) -> float:
    if minutes >= MIN_MINUTES_SHORT:
        return float(df_cards.get(player_id, 0.0))
    return 0.0


def calc_predicted_points_for_player(
    player: Player | str | int,
    fixture_goal_probs: dict,
    df_player: dict[str, pd.DataFrame],
    df_bonus: tuple[pd.Series, pd.Series] | None,
    df_saves: pd.Series | None,
    df_cards: pd.Series | None,
    df_def_con: tuple[pd.Series, pd.Series] | None,
    season: str,
    gw_range: list[int] | None = None,
    fixtures_behind: int | None = None,
    min_fixtures_behind: int = 3,
    tag: str = "",
    dbsession: Session = session,
) -> list[PlayerPrediction]:
    if isinstance(player, str | int):
        p = get_player(player, dbsession=dbsession)
        if p is None:
            msg = f"Player {player} not found in database"
            raise ValueError(msg)
        player = p

    message = f"Points prediction for player {player}"

    if not gw_range:
        gw_range = list(range(NEXT_GAMEWEEK, min(NEXT_GAMEWEEK + 3, 38)))

    if fixtures_behind is None:
        fixtures_behind = len(gw_range)

    fixtures_behind = max(fixtures_behind, min_fixtures_behind)

    team = player.team(season, gw_range[0])
    position = player.position(season)
    if position is None or team is None:
        msg = f"Player {player} has missing team or position for season {season}"
        raise ValueError(msg)

    fixtures = get_fixtures_for_player(
        player, season, gw_range=gw_range, dbsession=dbsession
    )

    player_prob = df_player[position].loc[player.player_id]
    if not isinstance(player_prob, pd.Series):
        msg = f"player_prob for {player} is not a Series, but {type(player_prob)}"
        raise RuntimeError(msg)

    recent_minutes = get_recent_minutes_for_player(
        player,
        num_match_to_use=fixtures_behind,
        season=season,
        last_gw=min(gw_range) - 1,
        dbsession=dbsession,
    )
    if len(recent_minutes) == 0:
        msg = "Recent minutes is empty."
        raise ValueError(msg)

    expected_points = defaultdict(float)
    predictions = []

    for fixture in fixtures:
        gameweek = fixture.gameweek
        if gameweek is None:
            logger.info(f"Skipping fixture {fixture} with no gameweek")
            continue

        is_home = fixture.home_team == team
        opponent = fixture.away_team if is_home else fixture.home_team
        home_or_away = "at home" if is_home else "away"
        message += f"\ngameweek: {gameweek} vs {opponent}  {home_or_away}"
        team_score_prob = fixture_goal_probs[fixture.fixture_id][team]
        team_concede_prob = fixture_goal_probs[fixture.fixture_id][opponent]

        points = 0.0
        expected_points[gameweek] = points

        if sum(recent_minutes) == 0:
            points = 0.0
        elif player.is_injured_or_suspended(season, gw_range[0], gameweek):
            points = 0.0
        elif was_historic_absence(
            player,
            gameweek=gameweek,
            season=season,
            dbsession=dbsession,
        ):
            points = 0.0
        else:
            points = 0
            for mins in recent_minutes:
                points += (
                    get_appearance_points(mins)
                    + get_attacking_points(
                        position,
                        mins,
                        team_score_prob,
                        player_prob,
                    )
                    + get_defending_points(position, mins, team_concede_prob)
                )
                if df_bonus is not None:
                    points += get_bonus_points(player.player_id, mins, df_bonus)
                if df_cards is not None:
                    points += get_card_points(player.player_id, mins, df_cards)
                if df_saves is not None:
                    points += get_save_points(
                        position, player.player_id, mins, df_saves
                    )
                if df_def_con is not None:
                    points += get_def_con_points(player.player_id, mins, df_def_con)

            points /= len(recent_minutes)

        if np.isnan(points):
            msg = f"nan points for {player} {fixture} {points} {tag}"
            raise ValueError(msg)

        predictions.append(make_prediction(player, fixture, points, tag))
        expected_points[gameweek] += points
        message += f"\nExpected points: {points:.2f}"

    logger.debug(message)
    return predictions


def calc_predicted_points_for_pos(
    pos: str,
    fixture_goal_probs: dict,
    df_bonus: tuple[pd.Series, pd.Series] | None,
    df_saves: pd.Series | None,
    df_cards: pd.Series | None,
    df_def_con: tuple[pd.Series, pd.Series] | None,
    season: str,
    gw_range: list[int],
    tag: str,
    model: NumpyroPlayerModel | ConjugatePlayerModel | None = None,
    dbsession: Session = session,
) -> dict[int, list[PlayerPrediction]]:
    df_player = {pos: fit_player_data(pos, season, min(gw_range), model, dbsession)}
    return {
        player.player_id: calc_predicted_points_for_player(
            player=player,
            fixture_goal_probs=fixture_goal_probs,
            df_player=df_player,
            df_bonus=df_bonus,
            df_saves=df_saves,
            df_cards=df_cards,
            df_def_con=df_def_con,
            season=season,
            gw_range=gw_range,
            tag=tag,
            dbsession=dbsession,
        )
        for player in list_players(
            position=pos, season=season, gameweek=min(gw_range), dbsession=dbsession
        )
    }


def make_prediction(
    player: Player, fixture: Fixture, points: float, tag: str
) -> PlayerPrediction:
    pp = PlayerPrediction()
    pp.predicted_points = points
    pp.tag = tag
    pp.player = player
    pp.fixture = fixture
    return pp


def fill_ep(csv_filename: str, dbsession: Session = session) -> None:
    if not os.path.exists(csv_filename):
        with open(csv_filename, "w") as outfile:
            outfile.write("player_id,gameweek,EP\n")

    tag = f"EP-{uuid.uuid4()!s}"
    summary_data = fetcher.get_player_summary_data()
    gameweek = NEXT_GAMEWEEK

    with open(csv_filename, "a") as outfile:
        for k, v in summary_data.items():
            player = get_player_from_api_id(k)
            if player is None:
                logger.warning(f"Player with API ID {k} not found in database")
                continue

            player_id = player.player_id
            outfile.write(f"{player_id},{gameweek},{v['ep_next']}\n")

            pp = PlayerPrediction()
            pp.player_id = player_id
            pp.fixture.gameweek = gameweek
            pp.predicted_points = v["ep_next"]
            pp.tag = tag
            dbsession.add(pp)

    dbsession.commit()


def process_player_data(
    prefix: str,
    season: str = CURRENT_SEASON,
    gameweek: int = NEXT_GAMEWEEK,
    dbsession: Session = session,
) -> dict:
    df = get_player_history_df(
        prefix, season=season, gameweek=gameweek, dbsession=dbsession
    )
    df["neither"] = df["team_goals"] - df["goals"] - df["assists"]
    df.loc[(df["neither"] < 0), ["neither", "team_goals", "goals", "assists"]] = [
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    alpha = get_empirical_bayes_estimates(df)

    # .values yerine modern Pandas standardı olan .to_numpy() kullanımı
    y = df.sort_values("player_id")[["goals", "assists", "neither"]].to_numpy().reshape(
        (
            df["player_id"].nunique(),
            df.groupby("player_id").count().iloc[0]["player_name"],
            3,
        )
    )

    minutes = df.sort_values("player_id")[["minutes"]].to_numpy().reshape(
        (
            df["player_id"].nunique(),
            df.groupby("player_id").count().iloc[0]["player_name"],
        )
    )

    nplayer = df["player_id"].nunique()
    nmatch = df.groupby("player_id").count().iloc[0]["player_name"]
    player_ids = np.sort(df["player_id"].unique())

    now_date = np.array(
        [
            pd.Timestamp(f.date).replace(tzinfo=None).date()
            for f in get_fixtures_for_gameweek(gameweek, season, dbsession)
            if f.date is not None
        ]
    ).min()

    match_date = df["date"].fillna(df["date"].min()).dt.date
    df["time_diff"] = (now_date - match_date) / pd.Timedelta(days=365)
    time_diff = df.sort_values("player_id")[["time_diff"]].to_numpy().reshape(
        (
            df["player_id"].nunique(),
            df.groupby("player_id").count().iloc[0]["player_name"],
        )
    )

    return {
        "player_ids": player_ids,
        "nplayer": nplayer,
        "nmatch": nmatch,
        "minutes": minutes.astype("int64"),
        "y": y.astype("int64"),
        "alpha": alpha,
        "time_diff": time_diff,
    }


def fit_player_data(
    position: str,
    season: str,
    gameweek: int,
    model: NumpyroPlayerModel | ConjugatePlayerModel | None = None,
    dbsession: Session = session,
    epsilon=DEFAULT_PLAYER_EPSILON,
    n_goals_prior=DEFAULT_N_GOALS_PRIOR,
) -> pd.DataFrame:
    if model is None:
        model = ConjugatePlayerModel()

    data = process_player_data(position, season, gameweek, dbsession)
    logger.info(f"Fitting player model for {position} ...")

    model = fastcopy(model)
    fitted_model = model.fit(data, epsilon=epsilon, n_goals_prior=n_goals_prior)
    df = pd.DataFrame(fitted_model.get_probs())

    df["pos"] = position
    return (
        df.rename(columns={"index": "player_id"})
        .sort_values("player_id")
        .set_index("player_id")
    )


def get_all_fitted_player_data(
    season: str,
    gameweek: int,
    model: NumpyroPlayerModel | ConjugatePlayerModel | None = None,
    dbsession: Session = session,
    epsilon=DEFAULT_PLAYER_EPSILON,
    n_goals_prior=DEFAULT_N_GOALS_PRIOR,
) -> dict[str, pd.DataFrame]:
    return {
        pos: fit_player_data(
            pos,
            season,
            gameweek,
            model,
            dbsession,
            epsilon=epsilon,
            n_goals_prior=n_goals_prior,
        )
        for pos in ["GK", "DEF", "MID", "FWD"]
    }


def get_player_scores(
    season: str,
    gameweek: int,
    min_minutes: int = 0,
    max_minutes: int = MAX_MINUTES_MATCH,
    position: str | None = None,
    dbsession: Session = session,
) -> pd.DataFrame:
    query = (
        select(PlayerScore, Fixture.season, Fixture.gameweek, PlayerAttributes.position)
        .where(PlayerScore.minutes >= min_minutes)
        .where(PlayerScore.minutes <= max_minutes)
        .join(Fixture)
        .join(
            PlayerAttributes,
            and_(
                PlayerAttributes.player_id == PlayerScore.player_id,
                PlayerAttributes.season == Fixture.season,
                PlayerAttributes.gameweek == Fixture.gameweek,
            ),
        )
        .order_by(Fixture.season, Fixture.gameweek, PlayerAttributes.player_id)
    )
    if position:
        query = query.where(PlayerAttributes.position == position)

    df = pd.read_sql(query, dbsession.connection())

    is_fut = partial(is_future_gameweek, current_season=season, next_gameweek=gameweek)
    exclude = df.apply(lambda r: is_fut(r["season"], r["gameweek"]), axis=1)
    return df[~exclude]


def mean_group_prior(
    df: pd.DataFrame,
    group_col: str,
    mean_col: str,
    n_prior: int = 10,
    prior_by_position: bool = False,
) -> pd.Series:
    group_counts = df.groupby(group_col)[mean_col].count()
    group_sums = df.groupby(group_col)[mean_col].sum()
    group_position = (
        df.sort_values(by=["season", "gameweek"]).groupby(group_col)["position"].last()
    )

    if prior_by_position:
        prior_sum = df.groupby("position")[mean_col].mean() * n_prior
        return (group_sums + prior_sum.loc[group_position].values) / (
            group_counts + n_prior
        )

    prior_sum = n_prior * df[mean_col].mean()
    return (group_sums + prior_sum) / (group_counts + n_prior)


def fit_bonus_points(
    gameweek: int = NEXT_GAMEWEEK,
    season: str = CURRENT_SEASON,
    n_prior: int = 10,
    dbsession: Session = session,
) -> tuple[pd.Series, pd.Series]:
    def get_bonus_df(min_minutes, max_minutes):
        df = get_player_scores(
            season,
            gameweek,
            min_minutes=min_minutes,
            max_minutes=max_minutes,
            dbsession=dbsession,
        )
        return mean_group_prior(
            df, "player_id", "bonus", n_prior=n_prior, prior_by_position=True
        )

    df_90 = get_bonus_df(MIN_MINUTES_FULL, MAX_MINUTES_MATCH)
    df_60 = get_bonus_df(MIN_MINUTES_SHORT, MIN_MINUTES_FULL - 1)

    return (df_90, df_60)


def fit_save_points(
    gameweek: int = NEXT_GAMEWEEK,
    season: str = CURRENT_SEASON,
    n_prior: int = 10,
    min_minutes: int = MAX_MINUTES_MATCH,
    dbsession: Session = session,
) -> pd.Series:
    df = get_player_scores(
        season, gameweek, min_minutes=min_minutes, position="GK", dbsession=dbsession
    )

    df["save_pts"] = (df["saves"] / saves_for_point).astype(int)

    return mean_group_prior(df, "player_id", "save_pts", n_prior=n_prior)


def fit_card_points(
    gameweek: int = NEXT_GAMEWEEK,
    season: str = CURRENT_SEASON,
    n_prior: int = 10,
    min_minutes: int = 1,
    dbsession: Session = session,
) -> pd.Series:
    df = get_player_scores(
        season, gameweek, min_minutes=min_minutes, dbsession=dbsession
    )

    df["card_pts"] = (
        points_for_yellow_card * df["yellow_cards"]
        + points_for_red_card * df["red_cards"]
    )

    return mean_group_prior(
        df, "player_id", "card_pts", n_prior=n_prior, prior_by_position=False
    )


def fit_def_con(
    gameweek: int = NEXT_GAMEWEEK,
    season: str = CURRENT_SEASON,
    n_prior: int = 10,
    dbsession: Session = session,
) -> tuple[pd.Series, pd.Series]:
    def get_def_con_df(min_minutes, max_minutes):
        dfs = []
        for position in ["DEF", "MID", "FWD"]:
            df = get_player_scores(
                season,
                gameweek,
                min_minutes=min_minutes,
                max_minutes=max_minutes,
                position=position,
                dbsession=dbsession,
            ).dropna(subset="defensive_contribution")
            df["def_con_pts"] = (
                df["defensive_contribution"] >= def_cons_required[position]
            ).astype(int) * points_for_def_cons
            dfs.append(df)

        return mean_group_prior(
            pd.concat(dfs),
            "player_id",
            "def_con_pts",
            n_prior=n_prior,
            prior_by_position=True,
        )

    df_90 = get_def_con_df(MIN_MINUTES_FULL, MAX_MINUTES_MATCH)
    df_60 = get_def_con_df(MIN_MINUTES_SHORT, MIN_MINUTES_FULL - 1)

    return (df_90, df_60)
