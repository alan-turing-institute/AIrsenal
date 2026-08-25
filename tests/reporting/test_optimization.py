"""
Tests for the renderers the transfer search and the from-scratch squad build
share.
"""

from airsenal.reporting.optimization import (
    GameweekRow,
    TransferRow,
    discord_payload,
    print_plan_table,
    print_result_panel,
    print_transfer_table,
)


def gameweek_row(gw, chip=None, points_hit=0):
    return GameweekRow(
        gameweek=gw,
        transfers="1",
        chip=chip,
        points_hit=points_hit,
        predicted_points=50.0,
    )


def transfer_row(gw, out_name, in_name):
    return TransferRow(
        gameweek=gw,
        player_out=out_name,
        position_out="MID",
        team_out="ARS",
        sale_price=100,
        player_in=in_name,
        position_in="MID",
        team_in="LIV",
        purchase_price=105,
    )


# --------------------------- the discord payload ---------------------------


def test_transfers_are_grouped_by_the_gameweek_they_are_made_in():
    payload = discord_payload(
        [gameweek_row(3), gameweek_row(4)],
        [
            transfer_row(3, "out-three", "in-three"),
            transfer_row(4, "out-four", "in-four"),
        ],
        ["a lineup"],
    )
    fields = payload["embeds"][0]["fields"]
    by_name = {f["name"]: f["value"] for f in fields}
    assert by_name["GW3 transfers out:"] == "out-three"
    assert by_name["GW3 transfers in:"] == "in-three"
    assert by_name["GW4 transfers out:"] == "out-four"
    assert by_name["GW4 transfers in:"] == "in-four"


def test_a_gameweek_with_no_transfers_still_gets_a_field():
    payload = discord_payload([gameweek_row(3)], [], ["a lineup"])
    fields = payload["embeds"][0]["fields"]
    assert {f["name"] for f in fields} == {
        "GW3 chips:",
        "GW3 transfers out:",
        "GW3 transfers in:",
    }
    assert all(f["value"].strip() in ("", "Chips played:  None") for f in fields)


def test_the_description_names_every_gameweek_in_the_plan():
    payload = discord_payload(
        [gameweek_row(3), gameweek_row(4), gameweek_row(5)], [], []
    )
    assert "3,4,5" in payload["embeds"][0]["description"]


def test_the_lineup_is_the_message_body():
    payload = discord_payload([], [], ["first", "second"])
    assert payload["content"] == "first\nsecond"
    assert payload["username"] == "AIrsenal"


# --------------------------- the tables ---------------------------
# These write to a Rich console, so what is asserted is that they render at all
# for the shapes both callers produce - including the ones only one of them hits.


def test_the_result_panel_omits_the_comparison_when_there_is_no_baseline(capsys):
    """The from-scratch squad build has nothing to compare against."""
    print_result_panel(gameweeks=[1, 2], fpl_team_id=1, optimised_score=60.0)
    out = capsys.readouterr().out
    assert "Optimised Score" in out
    assert "Baseline" not in out
    assert "Points Gained" not in out


def test_the_result_panel_shows_the_gain_when_there_is_a_baseline(capsys):
    print_result_panel(
        gameweeks=[3],
        fpl_team_id=1,
        optimised_score=60.0,
        baseline_score=50.0,
        points_hit=4,
        chips=("wildcard",),
    )
    out = capsys.readouterr().out
    assert "Gameweek: 3" in out
    assert "Baseline Score" in out
    assert "+10.0" in out
    assert "wildcard" in out


def test_an_empty_transfer_table_says_so_rather_than_printing_a_header(capsys):
    print_transfer_table([])
    assert "no transfers made" in capsys.readouterr().out


def test_a_gameweek_with_no_chip_renders_a_dash(capsys):
    print_plan_table([gameweek_row(3), gameweek_row(4, chip="wildcard")])
    out = capsys.readouterr().out
    assert "wildcard" in out
    assert "-" in out
