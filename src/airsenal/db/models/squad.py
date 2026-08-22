"""The user's squad: transactions, suggestions and interactive session state."""

from sqlalchemy.orm import Mapped, mapped_column

from airsenal.db.models.base import Base, intpk, str100, str100_optional


class Transaction(Base):
    __tablename__ = "transaction"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    player_id: Mapped[int]
    gameweek: Mapped[int]
    bought_or_sold: Mapped[int]  # +1 for bought, -1 for sold
    season: Mapped[str100]
    time: Mapped[str100]
    tag: Mapped[str100]
    price: Mapped[int]
    free_hit: Mapped[int]  # 1 if transfer on Free Hit, 0 otherwise
    fpl_team_id: Mapped[int]

    def __repr__(self):
        trans_str = f"{self.season} GW{self.gameweek}: Team {self.fpl_team_id} "
        if self.bought_or_sold == 1:
            trans_str += f"bought player {self.player_id}"
        else:
            trans_str += f"sold player {self.player_id}"
        if self.free_hit:
            trans_str += " (FREE HIT)"
        return trans_str


class TransferSuggestion(Base):
    __tablename__ = "transfer_suggestion"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    player_id: Mapped[int]
    in_or_out: Mapped[int]  # +1 for buy, -1 for sell
    gameweek: Mapped[int]
    points_gain: Mapped[float]
    timestamp: Mapped[str100]  # use this to group suggestions
    season: Mapped[str100]
    fpl_team_id: Mapped[int]  # to identify team to apply transfers.
    chip_played: Mapped[str100_optional]

    def __repr__(self):
        sugg_str = f"{self.season} GW{self.gameweek}: Suggest "
        if self.in_or_out == 1:
            sugg_str += f"buying {self.player_id} to gain {self.points_gain:.2f} pts"
        else:
            sugg_str += f"selling {self.player_id} to gain {self.points_gain:.2f} pts"
        return sugg_str


class SessionSquad(Base):
    __tablename__ = "sessionteam"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    session_id: Mapped[str100]
    player_id: Mapped[int]


class SessionBudget(Base):
    __tablename__ = "sessionbudget"
    id: Mapped[intpk] = mapped_column(autoincrement=True)
    session_id: Mapped[str100]
    budget: Mapped[int]
