"""Commands for running the full AIrsenal pipeline."""

from airsenal.cli import options
from airsenal.optimization.moves import ChipWeeks
from airsenal.optimization.protocols import (
    DEFAULT_MAX_OPT_TRANSFERS,
    DEFAULT_MAX_TOTAL_HIT,
    TransferConstraints,
)
from airsenal.optimization.squad_optimizers import (
    DEFAULT_SQUAD_OPTIMIZER,
    build_squad_optimizer,
)
from airsenal.optimization.squad_score import SquadScoringConfig
from airsenal.optimization.transfer_optimizers import (
    DEFAULT_TRANSFER_OPTIMIZER,
    build_transfer_optimizer,
)
from airsenal.pipeline import AIrsenalPipeline, DatabaseSettings, PipelineSettings
from airsenal.pipeline.settings import (
    DEFAULT_N_GAMEWEEKS,
    DEFAULT_N_PREVIOUS,
    StaleDatabase,
)
from airsenal.prediction.player_models import DEFAULT_PLAYER_MODEL, build_player_model
from airsenal.prediction.team_models import DEFAULT_TEAM_MODEL, build_team_model
from airsenal.squad.squad import SubWeights


def run(
    fpl_team_id: options.FplTeamId = None,
    n_gameweeks: options.WeeksAhead = DEFAULT_N_GAMEWEEKS,
    gameweek_start: options.GameweekStart = None,
    # --- database ---
    clean: options.Clean = False,
    n_previous: options.NPrevious = DEFAULT_N_PREVIOUS,
    current_season: options.CurrentSeason = True,
    refresh_database: options.RefreshDatabase = True,
    on_stale: options.OnStale = StaleDatabase.ASK,
    # --- prediction ---
    player_model: options.PlayerModel = DEFAULT_PLAYER_MODEL,
    team_model: options.TeamModel = DEFAULT_TEAM_MODEL,
    epsilon: options.Epsilon = None,
    # --- optimisation ---
    transfer_optimizer: options.TransferOptimizer = DEFAULT_TRANSFER_OPTIMIZER,
    squad_optimizer: options.SquadOptimizer = DEFAULT_SQUAD_OPTIMIZER,
    num_thread: options.NumThread = None,
    max_transfers: options.MaxTransfers = DEFAULT_MAX_OPT_TRANSFERS,
    max_hit: options.MaxHit = DEFAULT_MAX_TOTAL_HIT,
    allow_unused: options.AllowUnused = False,
    subs: options.Subs = True,
    wildcard_week: options.WildcardWeek = -1,
    free_hit_week: options.FreeHitWeek = -1,
    triple_captain_week: options.TripleCaptainWeek = -1,
    bench_boost_week: options.BenchBoostWeek = -1,
    # --- output ---
    apply_transfers: options.ApplyTransfers = False,
    yes: options.Yes = False,
    save_absences: options.SaveAbsences = False,
) -> None:
    """Run the full AIrsenal pipeline."""
    AIrsenalPipeline(
        team_model=build_team_model(team_model, epsilon),
        player_model=build_player_model(player_model),
        transfer_optimizer=build_transfer_optimizer(
            transfer_optimizer, num_thread=num_thread
        ),
        squad_optimizer=build_squad_optimizer(squad_optimizer),
        constraints=TransferConstraints(
            max_total_hit=max_hit,
            allow_unused_transfers=allow_unused,
            max_opt_transfers=max_transfers,
        ),
        scoring=SquadScoringConfig(
            sub_weights=SubWeights() if subs else SubWeights.none()
        ),
        settings=PipelineSettings(
            fpl_team_id=fpl_team_id,
            n_gameweeks=n_gameweeks,
            gameweek_start=gameweek_start,
            chips=ChipWeeks(
                wildcard=wildcard_week,
                free_hit=free_hit_week,
                triple_captain=triple_captain_week,
                bench_boost=bench_boost_week,
            ),
            database=DatabaseSettings(
                clean=clean,
                n_previous=n_previous,
                include_current_season=current_season,
            ),
            refresh_database=refresh_database,
            on_stale_database=on_stale,
            apply_transfers=apply_transfers,
            skip_confirmation=yes,
            save_absences=save_absences,
        ),
    ).run()
