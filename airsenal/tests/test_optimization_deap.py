"""
Tests for the DEAP-based optimization implementation.
"""

from unittest.mock import Mock, patch

from airsenal.framework.optimization_deap import SquadOptDEAP


def test_deap_class():
    """Basic test of DEAP optimization."""

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
            assert optimizer.n_opt_players == 15


def test_deap_optimization_creates_valid_squad():
    """Test that DEAP optimization creates a valid squad that meets all constraints."""

    # Mock the dependencies that require database access
    with patch(
        "airsenal.framework.optimization_deap.list_players"
    ) as mock_list_players:
        # Create mock players with varying prices and teams to test constraints
        mock_players = []
        for i in range(60):  # More players to give algorithm choice
            player = Mock()
            player.player_id = f"player_{i}"
            player.name = f"Player {i}"
            if i < 10:
                position = "GK"
            elif i < 30:
                position = "DEF"
            elif i < 50:
                position = "MID"
            else:
                position = "FWD"

            # Create methods that can be called with or without parameters
            def make_position_func(pos):
                return lambda season=None: pos

            def make_team_func(team_name):
                return lambda season=None, gameweek=None: team_name

            def make_price_func(price):
                return lambda season=None, gameweek=None: price

            player.position = make_position_func(position)
            # Spread players across different teams (max 3 per team constraint)
            team_name = f"Team {i // 3}"
            player.team = make_team_func(team_name)

            # Vary prices - some expensive, some cheap
            if i % 10 == 0:
                price = 130  # £13.0m premium player
            elif i % 5 == 0:
                price = 80  # £8.0m mid-price player
            else:
                price = 45  # £4.5m budget player
            player.price = make_price_func(price)
            mock_players.append(player)

        # Return appropriate players per position
        def mock_list_players_side_effect(position=None, **kwargs):
            return [p for p in mock_players if p.position() == position]

        mock_list_players.side_effect = mock_list_players_side_effect

        # Mock get_predicted_points_for_player with realistic variance
        with patch(
            "airsenal.framework.optimization_deap.get_predicted_points_for_player"
        ) as mock_get_points:
            # Premium players get more points
            def mock_points_side_effect(player, tag, season=None, dbsession=None):
                if player.price() == 130:  # Premium players
                    return {1: 8.0, 2: 7.5, 3: 8.5}
                elif player.price() == 80:  # Mid-price players
                    return {1: 5.5, 2: 5.0, 3: 6.0}
                else:  # Budget players
                    return {1: 3.0, 2: 2.5, 3: 3.5}

            mock_get_points.side_effect = mock_points_side_effect

            # Mock the _evaluate_individual method at the class level
            def mock_evaluate_individual(self, individual):
                # Check if individual is valid length
                if len(individual) != 15:
                    return (0.0,)

                # Check if all indices are valid
                for idx in individual:
                    if idx < 0 or idx >= len(self.players):
                        return (0.0,)

                # Get selected players
                selected_players = [self.players[int(idx)] for idx in individual]

                # Check position constraints
                position_counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
                for player in selected_players:
                    position_counts[player.position()] += 1

                if (
                    position_counts["GK"] != 2
                    or position_counts["DEF"] != 5
                    or position_counts["MID"] != 5
                    or position_counts["FWD"] != 3
                ):
                    return (0.0,)

                # Check budget constraint
                total_cost = sum(player.price() for player in selected_players)
                if total_cost > 1000:
                    return (0.0,)

                # Check team constraint (max 3 players per team)
                team_counts = {}
                for player in selected_players:
                    team = player.team()
                    team_counts[team] = team_counts.get(team, 0) + 1
                    if team_counts[team] > 3:
                        return (0.0,)

                # Check for duplicates
                player_ids = [p.player_id for p in selected_players]
                if len(player_ids) != len(set(player_ids)):
                    return (0.0,)

                # Calculate fitness based on predicted points
                total_points = 0.0
                for player in selected_players:
                    points_dict = mock_get_points.side_effect(player, "test_tag")
                    total_points += sum(points_dict.values())

                return (total_points,)

            # Patch the _evaluate_individual method at the class level
            with patch.object(
                SquadOptDEAP, "_evaluate_individual", mock_evaluate_individual
            ):
                # Initialize optimizer
                optimizer = SquadOptDEAP(
                    gw_range=[1, 2, 3],
                    tag="test_tag",
                    budget=1000,  # £100.0m budget
                    players_per_position={"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
                )

                # Run optimization with small parameters for fast test
                best_individual, best_fitness = optimizer.optimize(
                    population_size=20,
                    generations=10,
                    verbose=False,
                    random_state=42,  # For reproducible results
                )

                # Verify optimization returns valid results
                assert best_individual is not None
                assert len(best_individual) == 15
                assert best_fitness > 0

                # Check that all indices are valid
                for idx in best_individual:
                    assert 0 <= idx < len(optimizer.players)

                # Verify selected players meet constraints by examining them directly
                selected_players = [
                    optimizer.players[int(idx)] for idx in best_individual
                ]

                # Check position constraints
                position_counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
                for player in selected_players:
                    position_counts[player.position()] += 1

                assert position_counts["GK"] == 2, "Should have exactly 2 goalkeepers"
                assert position_counts["DEF"] == 5, "Should have exactly 5 defenders"
                assert position_counts["MID"] == 5, "Should have exactly 5 midfielders"
                assert position_counts["FWD"] == 3, "Should have exactly 3 forwards"

                # Check budget constraint
                total_cost = sum(player.price() for player in selected_players)
                assert total_cost <= 1000, (
                    f"Total cost {total_cost} exceeds budget of 1000"
                )

                # Check team constraint (max 3 players per team)
                team_counts = {}
                for player in selected_players:
                    team = player.team()
                    team_counts[team] = team_counts.get(team, 0) + 1
                    assert team_counts[team] <= 3, f"Too many players from {team}"

                # Verify no duplicate players
                player_ids = [p.player_id for p in selected_players]
                assert len(player_ids) == len(set(player_ids)), (
                    "Squad contains duplicate players"
                )

                # Verify the optimization found a reasonable solution (non-zero fitness)
                assert best_fitness > 0, (
                    "Optimization should find a solution with positive fitness"
                )
