"""
Tests for the DEAP-based optimization implementation.
"""

from unittest.mock import Mock, patch


def test_deap_import():
    """Test that DEAP optimization can be imported if available."""
    try:
        from airsenal.framework.optimization_deap import (
            SquadOptDEAP,
            make_new_squad_deap,
        )

        # If we get here, DEAP is available and imports work
        assert SquadOptDEAP is not None
        assert make_new_squad_deap is not None
        print("DEAP optimization imports successfully")
    except ModuleNotFoundError as e:
        if "deap" in str(e).lower():
            print("DEAP not installed - skipping test")
            # This is expected if DEAP isn't installed
            pass
        else:
            # Some other import error
            raise


def test_deap_optimization_basic():
    """Basic test of DEAP optimization if available."""
    try:
        from airsenal.framework.optimization_deap import SquadOptDEAP

        # Mock the dependencies that require database access
        with patch(
            "airsenal.framework.optimization_deap.list_players"
        ) as mock_list_players:
            # Create simple mock players
            mock_players = []
            for i in range(40):  # 40 total players
                player = Mock()
                player.player_id = f"player_{i}"
                player.name = f"Player {i}"
                if i < 5:
                    position = "GK"
                elif i < 20:
                    position = "DEF"
                elif i < 35:
                    position = "MID"
                else:
                    position = "FWD"
                player.position.return_value = position
                player.team.return_value = f"Team {i % 10}"
                player.price.return_value = 50  # £5.0m
                mock_players.append(player)

            # Return appropriate players per position
            def mock_list_players_side_effect(position=None, **kwargs):
                return [p for p in mock_players if p.position() == position]

            mock_list_players.side_effect = mock_list_players_side_effect

            # Mock get_predicted_points_for_player
            with patch(
                "airsenal.framework.optimization_deap.get_predicted_points_for_player"
            ) as mock_get_points:
                mock_get_points.return_value = {1: 5.0, 2: 4.0, 3: 6.0}  # Some points

                # Initialize optimizer
                optimizer = SquadOptDEAP(
                    gw_range=[1, 2, 3],
                    tag="test_tag",
                    budget=1000,
                    players_per_position={"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
                )

                # Check basic properties
                assert optimizer.gw_range == [1, 2, 3]
                assert optimizer.tag == "test_tag"
                assert optimizer.budget == 1000
                assert optimizer.n_opt_players == 15  # 2+5+5+3

                print("DEAP optimization basic test passed")

    except ModuleNotFoundError as e:
        if "deap" in str(e).lower():
            print("DEAP not installed - skipping test")
        else:
            raise


if __name__ == "__main__":
    test_deap_import()
    test_deap_optimization_basic()
