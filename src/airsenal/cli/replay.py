"""Commands for replaying historical seasons."""

from typing import Annotated

import typer

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
from airsenal.pipeline import (
    AIrsenalPipeline,
    PipelineSettings,
    ReplaySettings,
    run_replays,
)
from airsenal.pipeline.settings import DEFAULT_N_GAMEWEEKS
from airsenal.prediction.player_models import DEFAULT_PLAYER_MODEL, build_player_model
from airsenal.prediction.team_models import DEFAULT_TEAM_MODEL, build_team_model
from airsenal.squad.squad import SubWeights


def replay(
    season: Annotated[str, typer.Option(help="Season in the form 2526.")],
    gameweek_start: Annotated[
        int, typer.Option(min=1, help="First gameweek to replay.")
    ] = 1,
    gameweek_end: Annotated[
        int | None, typer.Option(help="Last gameweek to replay.")
    ] = None,
    n_gameweeks: options.NGameweeks = DEFAULT_N_GAMEWEEKS,
    fpl_team_id: options.FplTeamId = None,
    resume: Annotated[
        bool, typer.Option(help="Resume an existing replay team.")
    ] = False,
    loop: Annotated[
        int, typer.Option(help="Replay count; -1 repeats indefinitely.")
    ] = 1,
    num_thread: options.NumThread = None,
    num_iterations: options.NumIterations = None,
    num_generations: options.NumGenerations = None,
    population_size: options.PopulationSize = None,
    num_free_transfers: options.NumFreeTransfers = None,
    player_model: options.PlayerModel = DEFAULT_PLAYER_MODEL,
    team_model: options.TeamModel = DEFAULT_TEAM_MODEL,
    epsilon: options.Epsilon = None,
    transfer_optimizer: options.TransferOptimizer = DEFAULT_TRANSFER_OPTIMIZER,
    squad_optimizer: options.SquadOptimizer = DEFAULT_SQUAD_OPTIMIZER,
    max_transfers: options.MaxTransfers = DEFAULT_MAX_OPT_TRANSFERS,
    max_hit: options.MaxHit = DEFAULT_MAX_TOTAL_HIT,
    allow_unused: options.AllowUnused = False,
    wildcard_week: options.WildcardWeek = -1,
    free_hit_week: options.FreeHitWeek = -1,
    triple_captain_week: options.TripleCaptainWeek = -1,
    bench_boost_week: options.BenchBoostWeek = -1,
    subs: options.Subs = True,
    output_dir: options.OutputDir = None,
    tag_prefix: options.TagPrefix = "",
) -> None:
    """Replay a historical FPL season."""
    run_replays(
        AIrsenalPipeline(
            team_model=build_team_model(team_model, epsilon),
            player_model=build_player_model(player_model),
            transfer_optimizer=build_transfer_optimizer(
                transfer_optimizer,
                num_thread=num_thread,
                num_iterations=num_iterations,
            ),
            squad_optimizer=build_squad_optimizer(
                squad_optimizer,
                num_generations=num_generations,
                population_size=population_size,
            ),
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
                num_free_transfers=num_free_transfers,
                season=season,
                chips=ChipWeeks(
                    wildcard=wildcard_week,
                    free_hit=free_hit_week,
                    triple_captain=triple_captain_week,
                    bench_boost=bench_boost_week,
                ),
                # replay never touches the real entry or the live API
                refresh_database=False,
                apply_transfers=False,
            ),
        ),
        ReplaySettings(
            gameweek_start=gameweek_start,
            gameweek_end=gameweek_end,
            tag_prefix=tag_prefix,
            loop=loop,
            resume=resume,
            output_dir=output_dir,
        ),
    )
