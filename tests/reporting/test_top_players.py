"""Filtering the top-predicted-points table."""

from airsenal.reporting import top_players
from airsenal.reporting.top_players import get_top_predicted_points, within_price


class FakePlayer:
    def __init__(self, name, price):
        self.name = name
        self._price = price

    def price(self, gameweek, season):  # noqa: ARG002
        return self._price

    def __repr__(self):
        return self.name


def predictions(*prices):
    return [(FakePlayer(f"p{i}", price), 1.0) for i, price in enumerate(prices)]


def test_no_cap_keeps_everyone():
    pts = predictions(50, 100, 150)
    assert within_price(pts, None, 1, "2526") == pts


def test_consecutive_players_over_the_cap_are_all_dropped():
    """Two adjacent players over the cap are both dropped, not just the first."""
    kept = within_price(
        predictions(150, 160, 50), max_price=100, gameweek=1, season="2526"
    )
    assert [p.name for p, _ in kept] == ["p2"]


def test_everyone_over_the_cap_is_dropped():
    assert within_price(predictions(150, 160, 170), 100, 1, "2526") == []


def test_a_player_exactly_on_the_cap_is_kept():
    kept = within_price(predictions(100), 100, 1, "2526")
    assert len(kept) == 1


def test_a_player_with_no_price_is_kept():
    """None means unknown, not free - dropping them would hide a real player."""
    kept = within_price(predictions(None, 150), 100, 1, "2526")
    assert [p.name for p, _ in kept] == ["p0"]


def test_the_latest_predictions_are_those_for_the_season_asked_for(monkeypatch):
    """Without a tag, the latest one is looked up in the season being shown."""
    dbsession = object()
    lookups = []

    def record_latest_tag(season="unset", tag_prefix="", dbsession="unset"):
        lookups.append((season, dbsession))
        return "latest"

    monkeypatch.setattr(top_players, "get_latest_prediction_tag", record_latest_tag)
    monkeypatch.setattr(top_players, "get_webhook_url", lambda: None)
    monkeypatch.setattr(top_players, "post_webhook", lambda _payload, _url: None)
    monkeypatch.setattr(top_players, "get_predicted_points", lambda *a, **k: [])

    get_top_predicted_points(gameweeks=[1], season="2425", dbsession=dbsession)

    assert lookups == [("2425", dbsession)]
