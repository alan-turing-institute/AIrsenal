"""
The whole pipeline, as one configured object.

`run_pipeline` took twenty arguments - the repo's lint ceiling was set to
accommodate it - and mixed three unrelated things: which models and algorithms
to use, which database to build, and what to do with the answer. Worse, the
choice of model reached it as a string, so the only way to run the pipeline with
something the registry did not know about was not to use the pipeline.

`AIrsenalPipeline` holds the four swappable components as objects, so a model or
an optimizer written anywhere can be dropped in, and `from_names` does the
registry lookups for the command line. Settings that belong to a component - the
GA's population, the search's thread count, a model's epsilon - live on that
component and not here.
"""

from dataclasses import dataclass, field, replace
from typing import Any

from curl_cffi import requests
from sqlalchemy.orm.session import Session

from airsenal.apply.lineup import set_lineup
from airsenal.apply.transfers import make_transfers
from airsenal.core.concurrency import set_multiprocessing_start_method
from airsenal.core.console import confirm
from airsenal.core.logging import get_logger
from airsenal.core.season import CURRENT_SEASON
from airsenal.db.queries.gameweeks import get_gameweeks_array, next_gameweek
from airsenal.db.session import session_scope
from airsenal.export.absences import main as save_expected_absences
from airsenal.fetch.fpl_api import get_fetcher, require_fpl_team_id
from airsenal.ingest.init_db import check_clean_db, make_init_db
from airsenal.ingest.update import update_db
from airsenal.optimization.moves import TransferConstraints
from airsenal.optimization.protocols import SquadOptimizer, TransferOptimizer
from airsenal.optimization.run_squad import fill_initial_squad
from airsenal.optimization.run_transfers import run_optimization
from airsenal.optimization.squad_optimizers import SQUAD_OPTIMIZERS
from airsenal.optimization.strategy import Strategy
from airsenal.optimization.transfer_optimizers import TRANSFER_OPTIMIZERS
from airsenal.pipeline.settings import PipelineSettings, StaleDatabase
from airsenal.prediction.protocols import ConfiguredTeamModel, PlayerModel
from airsenal.prediction.registry import (
    DEFAULT_PLAYER_MODEL,
    DEFAULT_TEAM_MODEL,
    PLAYER_MODELS,
    build_team_model,
)
from airsenal.prediction.run import make_predictedscore_table
from airsenal.reporting.top_players import get_top_predicted_points
from airsenal.squad.squad import Squad
from airsenal.squad.state import get_entry_start_gameweek

logger = get_logger(__name__)


class StaleDatabaseError(RuntimeError):
    """The database could not be updated and the run was told not to continue."""


@dataclass(frozen=True, kw_only=True)
class AIrsenalPipeline:
    """
    A configured run: what to predict and optimise with, and what to do with it.

    The components are objects rather than names so that anything satisfying the
    protocol can be used, including something defined in a notebook. Use
    `from_names` to build one from the strings a command line supplies.
    """

    team_model: ConfiguredTeamModel = field(default_factory=build_team_model)
    player_model: PlayerModel = field(
        default_factory=lambda: PLAYER_MODELS.create(DEFAULT_PLAYER_MODEL)
    )
    transfer_optimizer: TransferOptimizer = field(
        default_factory=lambda: TRANSFER_OPTIMIZERS.create("tree_search")
    )
    squad_optimizer: SquadOptimizer = field(
        default_factory=lambda: SQUAD_OPTIMIZERS.create("genetic")
    )
    constraints: TransferConstraints = field(default_factory=TransferConstraints)
    settings: PipelineSettings = field(default_factory=PipelineSettings)

    @classmethod
    def from_names(
        cls,
        *,
        team_model: str = DEFAULT_TEAM_MODEL,
        player_model: str = DEFAULT_PLAYER_MODEL,
        epsilon: float | None = None,
        team_options: dict[str, str] | None = None,
        player_options: dict[str, str] | None = None,
        transfer_optimizer: TransferOptimizer | None = None,
        squad_optimizer: SquadOptimizer | None = None,
        constraints: TransferConstraints | None = None,
        settings: PipelineSettings | None = None,
    ) -> "AIrsenalPipeline":
        """
        Build a pipeline from the names and options a command line supplies.

        The registry lookups happen here and nowhere else, so constructing a
        pipeline directly never reads a string and never raises ConfigError.
        Keyword-only because `team_model` and `player_model` are both bare
        strings, and getting them the wrong way round would otherwise be quiet.
        """
        return cls(
            team_model=build_team_model(team_model, team_options, epsilon),
            player_model=PLAYER_MODELS.create_with(player_model, player_options or {}),
            transfer_optimizer=(
                transfer_optimizer
                if transfer_optimizer is not None
                else TRANSFER_OPTIMIZERS.create("tree_search")
            ),
            squad_optimizer=(
                squad_optimizer
                if squad_optimizer is not None
                else SQUAD_OPTIMIZERS.create("genetic")
            ),
            constraints=(
                constraints if constraints is not None else TransferConstraints()
            ),
            settings=settings if settings is not None else PipelineSettings(),
        )

    def with_settings(self, **changes: Any) -> "AIrsenalPipeline":
        """The same components, with some of the settings changed."""
        return replace(self, settings=replace(self.settings, **changes))

    # ------------------------------------------------------------------ stages

    def gameweeks(
        self, dbsession: Session | None = None, gameweek_start: int | None = None
    ) -> list[int]:
        """The gameweek window this run covers."""
        return get_gameweeks_array(
            n_gameweeks=self.settings.n_gameweeks,
            gameweek_start=(
                gameweek_start
                if gameweek_start is not None
                else self.settings.gameweek_start
            ),
            season=self.settings.season,
            dbsession=dbsession,
        )

    def predict(
        self, gameweeks: list[int], dbsession: Session, tag_prefix: str = ""
    ) -> str:
        """
        Predict points for every player, and return the tag they were written under.

        The tag is returned rather than looked up again afterwards: the optimiser
        used to re-find it with `get_latest_prediction_tag`, which is a race
        between two runs and needless in any case.
        """
        return make_predictedscore_table(
            gameweeks=gameweeks,
            season=self.settings.season,
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
    ) -> tuple[Squad, Strategy | None]:
        """
        Choose a squad: build one from scratch, or transfer into the current one.

        Which of the two is the decision this object exists to make, and it is
        what makes both optimizer components live rather than one of them dead.

        Returns the squad and the strategy that produced it. The strategy is None
        when the squad was built from scratch: there was nothing to transfer
        from, so there is no sequence of moves to describe.
        """
        if self._is_new_squad(fpl_team_id):
            logger.info("[bold]Generating Squad[/bold]")
            return fill_initial_squad(
                tag=tag,
                gameweeks=gameweeks,
                season=self.settings.season,
                fpl_team_id=fpl_team_id,
                optimizer=self.squad_optimizer,
                chip_gameweeks=self.settings.chips.as_dict(),
                is_replay=is_replay,
            ), None

        logger.info("[bold]Optimising Transfers[/bold]")
        return run_optimization(
            gameweeks=gameweeks,
            tag=tag,
            season=self.settings.season,
            fpl_team_id=fpl_team_id,
            chip_gameweeks=self.settings.chips.as_dict(),
            constraints=self.constraints,
            optimizer=self.transfer_optimizer,
            squad_optimizer=self.squad_optimizer,
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

    def _is_new_squad(self, fpl_team_id: int) -> bool:
        """Whether there is no squad yet to transfer from."""
        if self.settings.new_squad is not None:
            return self.settings.new_squad
        return get_entry_start_gameweek(fpl_team_id, get_fetcher()) == next_gameweek()

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
        except requests.exceptions.RequestException:
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
        # Raising rather than sys.exit: this is a library function, and a
        # function that exits the interpreter cannot be tested or reused.
        raise StaleDatabaseError(message)

    def _apply(self, fpl_team_id: int) -> None:
        if self.settings.season != CURRENT_SEASON:
            msg = (
                f"Refusing to apply transfers for season {self.settings.season}: "
                f"the FPL entry is playing {CURRENT_SEASON}."
            )
            raise RuntimeError(msg)
        logger.info("[bold]Applying Transfers[/bold]")
        if not make_transfers(fpl_team_id):
            msg = "Problem applying the transfers"
            raise RuntimeError(msg)
        logger.info("[bold]Setting Lineup[/bold]")
        set_lineup(fpl_team_id)
