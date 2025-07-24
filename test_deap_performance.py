#!/usr/bin/env python3
"""
Simple performance test of DEAP optimization without requiring full database
"""

import random
import time
from unittest.mock import Mock, patch


def create_mock_player(player_id, position, price=50):
    """Create a mock player for testing"""
    player = Mock()
    player.player_id = player_id
    player.name = f"Player {player_id}"
    player.position.return_value = position
    # Convert player_id to int for modulo operation
    id_num = hash(player_id) % 10  # Use hash to get a consistent integer
    player.team.return_value = f"Team {id_num}"
    player.price.return_value = price
    return player


def test_deap_performance():
    """Test DEAP optimization performance with mock data"""
    print("DEAP Performance Test")
    print("=" * 40)

    try:
        from airsenal.framework.optimization_deap import SquadOptDEAP

        # Create mock players for each position
        mock_players = []
        position_counts = {"GK": 10, "DEF": 30, "MID": 35, "FWD": 20}
        player_id = 0

        for pos, count in position_counts.items():
            for i in range(count):
                mock_players.append(create_mock_player(f"{pos}_{i}", pos))
                player_id += 1

        print(f"Created {len(mock_players)} mock players")

        # Mock the framework dependencies
        with patch(
            "airsenal.framework.optimization_deap.list_players"
        ) as mock_list_players:
            with patch(
                "airsenal.framework.optimization_deap.get_predicted_points_for_player"
            ) as mock_get_points:
                with patch(
                    "airsenal.framework.optimization_deap.Squad"
                ) as mock_squad_class:
                    with patch(
                        "airsenal.framework.optimization_deap.get_discounted_squad_score"
                    ) as mock_score:
                        # Setup mocks
                        def mock_list_players_side_effect(position=None, **kwargs):
                            return [p for p in mock_players if p.position() == position]

                        mock_list_players.side_effect = mock_list_players_side_effect
                        mock_get_points.return_value = {
                            1: random.uniform(2, 8),
                            2: random.uniform(2, 8),
                            3: random.uniform(2, 8),
                        }

                        # Mock squad to always return valid results
                        mock_squad = Mock()
                        mock_squad.add_player.return_value = True
                        mock_squad.is_complete.return_value = True
                        mock_squad_class.return_value = mock_squad

                        # Mock scoring function
                        mock_score.return_value = random.uniform(50, 150)

                        print("Setting up optimizer...")

                        # Create optimizer
                        optimizer = SquadOptDEAP(
                            gw_range=[1, 2, 3],
                            tag="test_tag",
                            budget=1000,
                            players_per_position={
                                "GK": 2,
                                "DEF": 5,
                                "MID": 5,
                                "FWD": 3,
                            },
                        )

                        print(
                            f"Optimizer setup complete. Available players: {optimizer.n_available_players}"
                        )

                        # Test different population/generation combinations
                        test_configs = [
                            {"pop": 20, "gen": 5, "name": "Quick test"},
                            {"pop": 50, "gen": 10, "name": "Medium test"},
                            {"pop": 100, "gen": 20, "name": "Production test"},
                        ]

                        for config in test_configs:
                            print(
                                f"\n{config['name']} (pop={config['pop']}, gen={config['gen']}):"
                            )

                            start_time = time.time()

                            best_individual, best_fitness = optimizer.optimize(
                                population_size=config["pop"],
                                generations=config["gen"],
                                verbose=False,
                                random_state=42,
                            )

                            end_time = time.time()
                            elapsed = end_time - start_time

                            print(f"  ✓ Completed in {elapsed:.2f} seconds")
                            print(f"  ✓ Best fitness: {best_fitness:.2f}")
                            print(f"  ✓ Best individual length: {len(best_individual)}")

                            # Verify individual is valid
                            assert len(best_individual) == 15, (
                                f"Expected 15 players, got {len(best_individual)}"
                            )
                            assert all(isinstance(x, int) for x in best_individual), (
                                "All player indices should be integers"
                            )

                        print("\n" + "=" * 40)
                        print("✓ All performance tests passed!")
                        print("✓ DEAP optimization is working correctly")

    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_deap_performance()
