"""The whole pipeline, as one configured object."""

from dataclasses import dataclass, field, replace
from typing import Any

from sqlalchemy.orm.session import Session

from airsenal.apply.lineup import set_lineup
from airsenal.apply.transfers import make_transfers
from airsenal.core.concurrency import set_multiprocessing_start_method
from airsenal.core.console import confirm
from airsenal.core.logging import get_logger
from airsenal.core.lookup import ConfigError
from airsenal.db.queries.gameweeks import get_gameweeks_array
from airsenal.db.queries.tags import check_tag_valid
from airsenal.db.session import session_scope
from airsenal.export.absences import save_expected_absences
from airsenal.game.season import CURRENT_SEASON
from airsenal.ingest.init_db import check_clean_db, make_init_db
from airsenal.ingest.update import update_db
from airsenal.optimization.plan import Plan
from airsenal.optimization.protocols import (
    SquadOptimizer,
    TransferConstraints,
    TransferOptimizer,
)
from airsenal.optimization.run_squad import build_new_squad
from airsenal.optimization.run_transfers import run_optimization
from airsenal.optimization.squad_optimizers import GeneticSquadOptimizer
from airsenal.optimization.squad_score import SquadScoringConfig
from airsenal.optimization.transfer_optimizers import TreeSearchOptimizer
from airsenal.pipeline.settings import PipelineSettings, StaleDatabase
from airsenal.prediction.player_models import (
    build_player_model,
)
from airsenal.prediction.points import PointsConfig
from airsenal.prediction.protocols import PlayerModel, TeamModel
from airsenal.prediction.run import make_predictedscore_table
from airsenal.prediction.team_models import (
    build_team_model,
)
from airsenal.remote.errors import RemoteError
from airsenal.remote.fpl_api import get_fetcher, require_fpl_team_id
from airsenal.reporting.top_players import get_top_predicted_points
from airsenal.squad.squad import Squad
from airsenal.squad.state import get_entry_start_gameweek

logger = get_logger(__name__)


class StaleDatabaseError(RuntimeError):
    """The database could not be updated and the run was told not to continue."""


@dataclass(frozen=True, kw_only=True)
class AIrsenalPipeline:
    """A configured run: what to predict and optimise with, and what to do with it."""

    team_model: TeamModel = field(default_factory=build_team_model)
    player_model: PlayerModel = field(default_factory=build_player_model)
    transfer_optimizer: TransferOptimizer = field(default_factory=TreeSearchOptimizer)
    squad_optimizer: SquadOptimizer = field(default_factory=GeneticSquadOptimizer)
    constraints: TransferConstraints = field(default_factory=TransferConstraints)
    scoring: SquadScoringConfig = field(default_factory=SquadScoringConfig)
    points: PointsConfig = field(default_factory=PointsConfig)
    settings: PipelineSettings = field(default_factory=PipelineSettings)

    def with_settings(self, **changes: Any) -> "AIrsenalPipeline":
        """The same components, with some of the settings changed."""
        return replace(self, settings=replace(self.settings, **changes))

    # ------------------------------------------------------------------ stages

    def gameweeks(
        self, dbsession: Session | None = None, gameweek_start: int | None = None
    ) -> list[int]:
        """The gameweek window this run covers."""
        gameweek_end = self.settings.gameweek_end
        return get_gameweeks_array(
            n_gameweeks=None if gameweek_end is not None else self.settings.n_gameweeks,
            gameweek_start=(
                gameweek_start
                if gameweek_start is not None
                else self.settings.gameweek_start
            ),
            gameweek_end=gameweek_end,
            season=self.settings.season,
            dbsession=dbsession,
        )

    def predict(
        self, gameweeks: list[int], dbsession: Session, tag_prefix: str = ""
    ) -> str:
        """
        Predict points for every player.

        Returns the tag they were written under in the database.
        """
        return make_predictedscore_table(
            gameweeks=gameweeks,
            season=self.settings.season,
            points=self.points,
            tag_prefix=tag_prefix,
            player_model=self.player_model,
            team_model=self.team_model,
            dbsession=dbsession,
        )

    def optimize(
        self,
        gameweeks: list[int],
        tag: str,
        fpl_team_id: int,
        is_replay: bool = False,
    ) -> tuple[Squad, Plan | None]:
        """
        Choose a squad: build one from scratch, or transfer into the current one.

        Returns the squad and the plan that produced it. The plan is None when
        the squad was built from scratch: there was nothing to transfer from, so
        there is no sequence of moves to describe.
        """
        self._require_predictions(gameweeks, tag)
        set_multiprocessing_start_method()
        if self._is_new_squad(fpl_team_id, gameweeks):
            logger.info("[bold]Generating Squad[/bold]")
            return build_new_squad(
                tag=tag,
                gameweeks=gameweeks,
                season=self.settings.season,
                fpl_team_id=fpl_team_id,
                optimizer=self.squad_optimizer,
                scoring=self.scoring,
                remove_zero=self.settings.remove_zero_points_players,
                chips=self.settings.chips,
                is_replay=is_replay,
            ), None

        logger.info("[bold]Optimising Transfers[/bold]")
        return run_optimization(
            gameweeks=gameweeks,
            tag=tag,
            season=self.settings.season,
            fpl_team_id=fpl_team_id,
            chips=self.settings.chips,
            num_free_transfers=self.settings.num_free_transfers,
            constraints=self.constraints,
            optimizer=self.transfer_optimizer,
            squad_optimizer=self.squad_optimizer,
            scoring=self.scoring,
            save_plans=self.settings.save_plans,
            is_replay=is_replay,
        )

    # -------------------------------------------------------------------- run

    def run(self) -> None:
        """Set up the database, predict, optimise, and optionally apply."""
        fpl_team_id = require_fpl_team_id(self.settings.fpl_team_id)
        logger.info("Running for FPL Team ID %s", fpl_team_id)
        set_multiprocessing_start_method()

        with session_scope() as dbsession:
            if self.settings.refresh_database:
                self._refresh_database(fpl_team_id, dbsession)

            gameweeks = self.gameweeks(dbsession)

            logger.info("[bold]Points Prediction[/bold]")
            tag = self.predict(gameweeks, dbsession)
            get_top_predicted_points(
                gameweeks=gameweeks,
                tag=tag,
                season=self.settings.season,
                per_position=True,
                n_players=5,
                dbsession=dbsession,
            )
            logger.info("[green]Prediction complete![/green]")

            self.optimize(gameweeks, tag, fpl_team_id)
            logger.info("[green]Optimization complete![/green]")

            if self.settings.apply_transfers:
                self._apply(fpl_team_id)
            if self.settings.save_absences:
                logger.info("[bold]Saving Absences[/bold]")
                save_expected_absences()
            logger.info("[green]Pipeline finished![/green]")

    # ---------------------------------------------------------------- private

    def _require_predictions(self, gameweeks: list[int], tag: str) -> None:
        """
        Refuse to optimise against predictions that do not cover the window.

        Raises `ConfigError` rather than a plain error so `main_cli` reports it
        as a bad option rather than as a crash.
        """
        if check_tag_valid(tag, gameweeks, season=self.settings.season):
            return
        msg = (
            f"Prediction tag '{tag}' does not cover gameweeks "
            f"{gameweeks[0]}-{gameweeks[-1]} of season {self.settings.season}. "
            "Run `airsenal predict` first, for the same gameweeks and season."
        )
        raise ConfigError(msg)

    def _is_new_squad(self, fpl_team_id: int, gameweeks: list[int]) -> bool:
        """Whether there is no squad yet to transfer from."""
        if self.settings.new_squad is not None:
            return self.settings.new_squad
        if gameweeks[0] == 1:
            return True
        return (
            get_entry_start_gameweek(fpl_team_id, get_fetcher(fpl_team_id))
            == gameweeks[0]
        )

    def _refresh_database(self, fpl_team_id: int, dbsession: Session) -> None:
        if check_clean_db(self.settings.database.clean, dbsession):
            logger.info("[bold]Database Setup[/bold]")
            self._create_database(fpl_team_id, dbsession)
            logger.info("[green]Database setup complete![/green]")
            update_attributes = False
        else:
            logger.debug("Found pre-existing AIrsenal database.")
            update_attributes = True

        logger.info("[bold]Updating database[/bold]")
        self._update_database(fpl_team_id, update_attributes, dbsession)

    def _create_database(self, fpl_team_id: int, dbsession: Session) -> None:
        if not make_init_db(fpl_team_id, self.settings.database.seasons(), dbsession):
            msg = "Problem setting up initial db"
            raise RuntimeError(msg)

    def _update_database(
        self, fpl_team_id: int, attributes: bool, dbsession: Session
    ) -> None:
        try:
            updated = update_db(CURRENT_SEASON, attributes, fpl_team_id, dbsession)
        except RemoteError:
            logger.warning("Database update failed.", exc_info=True)
            updated = False

        if updated:
            logger.info("[green]Database update complete![/green]")
            return

        message = (
            "The database update failed. AIrsenal can continue using the latest "
            "status of its database but the results may be outdated or invalid."
        )
        if self.settings.on_stale_database is StaleDatabase.CONTINUE:
            logger.warning(message)
            return
        if self.settings.on_stale_database is StaleDatabase.ASK:
            logger.warning(message)
            if confirm("Do you want to continue?"):
                return

        raise StaleDatabaseError(message)

    def _apply(self, fpl_team_id: int) -> None:
        if self.settings.season != CURRENT_SEASON:
            msg = (
                f"Refusing to apply transfers for season {self.settings.season}: "
                f"the FPL entry is playing {CURRENT_SEASON}."
            )
            raise RuntimeError(msg)
        skip_check = self.settings.skip_confirmation
        logger.info("[bold]Applying Transfers[/bold]")
        applied = make_transfers(fpl_team_id, skip_check=skip_check)
        if applied is None:
            logger.info("No transfers to apply.")
        elif applied is False:
            logger.info("Transfers not applied.")
        logger.info("[bold]Setting Lineup[/bold]")
        set_lineup(fpl_team_id, skip_check=skip_check)
