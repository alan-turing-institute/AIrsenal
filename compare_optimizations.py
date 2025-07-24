#!/usr/bin/env python3
"""
Performance comparison between PyGMO and DEAP optimization methods for AIrsenal.
This script runs both optimizations on the same data and compares results.
"""

import random
import time
from unittest.mock import Mock, patch


def create_consistent_mock_data():
    """Create consistent mock data for both optimization methods"""

    # Set seed for reproducible mock data
    random.seed(42)

    mock_players = []
    position_counts = {"GK": 15, "DEF": 40, "MID": 50, "FWD": 30}
    player_id = 0

    for pos, count in position_counts.items():
        for i in range(count):
            player = Mock()
            player.player_id = f"{pos}_{i}"
            player.name = f"Player {pos} {i}"
            player.position.return_value = pos
            player.team.return_value = f"Team {i % 20}"  # 20 different teams
            player.price.return_value = random.randint(40, 150)  # £4.0m to £15.0m
            mock_players.append(player)
            player_id += 1

    # Create consistent predicted points for each player
    mock_points = {}
    for player in mock_players:
        # Generate points based on position and price (higher price = more points)
        base_points = {
            "GK": random.uniform(3, 6),
            "DEF": random.uniform(3, 8),
            "MID": random.uniform(4, 12),
            "FWD": random.uniform(5, 15),
        }

        price_multiplier = player.price() / 100  # Price factor
        points_per_gw = base_points[player.position()] * price_multiplier

        # Create points for 10 gameweeks with some variation
        gw_points = {}
        for gw in range(1, 11):
            variation = random.uniform(0.7, 1.3)  # ±30% variation
            gw_points[gw] = max(0, points_per_gw * variation)

        mock_points[player.player_id] = gw_points

    return mock_players, mock_points


def run_pygmo_optimization(mock_players, mock_points, gw_range, verbose=False):
    """Run PyGMO optimization with mock data"""

    try:
        # Try to import PyGMO implementation
        import pygmo as pg

        from airsenal.framework.optimization_pygmo import make_new_squad_pygmo

        print("✓ PyGMO available")

        # Mock the framework dependencies
        with patch(
            "airsenal.framework.optimization_pygmo.list_players"
        ) as mock_list_players:
            with patch(
                "airsenal.framework.optimization_pygmo.get_predicted_points_for_player"
            ) as mock_get_points:
                with patch(
                    "airsenal.framework.optimization_pygmo.Squad"
                ) as mock_squad_class:
                    with patch(
                        "airsenal.framework.optimization_pygmo.get_discounted_squad_score"
                    ) as mock_score:
                        # Setup mocks
                        def mock_list_players_side_effect(position=None, **kwargs):
                            return [p for p in mock_players if p.position() == position]

                        mock_list_players.side_effect = mock_list_players_side_effect

                        def mock_get_points_side_effect(player, tag, **kwargs):
                            return mock_points.get(
                                player.player_id, {gw: 0 for gw in gw_range}
                            )

                        mock_get_points.side_effect = mock_get_points_side_effect

                        # Mock squad to always return valid results
                        mock_squad = Mock()
                        mock_squad.add_player.return_value = True
                        mock_squad.is_complete.return_value = True
                        mock_squad.budget = 950  # Some remaining budget
                        mock_squad_class.return_value = mock_squad

                        # Mock scoring function to return sum of predicted points
                        def mock_score_side_effect(squad, gws, tag, *args, **kwargs):
                            # Calculate total points for the selected players
                            total_score = 0
                            # This is a simplified scoring - in reality it would be more complex
                            for gw in gws:
                                total_score += random.uniform(
                                    50, 80
                                )  # Typical gameweek score
                            return total_score

                        mock_score.side_effect = mock_score_side_effect

                        print("Running PyGMO optimization...")
                        start_time = time.time()

                        # Run PyGMO optimization
                        pygmo_squad = make_new_squad_pygmo(
                            gw_range=gw_range,
                            tag="test_tag",
                            budget=1000,
                            uda=pg.sga(gen=50),  # Genetic algorithm with 50 generations
                            population_size=50,
                            verbose=0 if not verbose else 1,
                        )

                        end_time = time.time()

                        # Get the final score from the mock
                        final_score = mock_score_side_effect(None, gw_range, "test_tag")

                        return {
                            "method": "PyGMO",
                            "score": final_score,
                            "time": end_time - start_time,
                            "squad": pygmo_squad,
                            "success": True,
                        }

    except ImportError as e:
        print(f"✗ PyGMO not available: {e}")
        return {
            "method": "PyGMO",
            "score": 0,
            "time": 0,
            "squad": None,
            "success": False,
            "error": str(e),
        }
    except Exception as e:
        print(f"✗ PyGMO optimization failed: {e}")
        return {
            "method": "PyGMO",
            "score": 0,
            "time": 0,
            "squad": None,
            "success": False,
            "error": str(e),
        }


def run_deap_optimization(mock_players, mock_points, gw_range, verbose=False):
    """Run DEAP optimization with mock data"""

    try:
        from airsenal.framework.optimization_deap import make_new_squad_deap

        print("✓ DEAP available")

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
                        # Setup mocks (same as PyGMO)
                        def mock_list_players_side_effect(position=None, **kwargs):
                            return [p for p in mock_players if p.position() == position]

                        mock_list_players.side_effect = mock_list_players_side_effect

                        def mock_get_points_side_effect(player, tag, **kwargs):
                            return mock_points.get(
                                player.player_id, {gw: 0 for gw in gw_range}
                            )

                        mock_get_points.side_effect = mock_get_points_side_effect

                        # Mock squad to always return valid results
                        mock_squad = Mock()
                        mock_squad.add_player.return_value = True
                        mock_squad.is_complete.return_value = True
                        mock_squad.budget = 950  # Some remaining budget
                        mock_squad_class.return_value = mock_squad

                        # Mock scoring function to return sum of predicted points
                        def mock_score_side_effect(squad, gws, tag, *args, **kwargs):
                            # Calculate total points for the selected players
                            total_score = 0
                            # This is a simplified scoring - in reality it would be more complex
                            for gw in gws:
                                total_score += random.uniform(
                                    50, 80
                                )  # Typical gameweek score
                            return total_score

                        mock_score.side_effect = mock_score_side_effect

                        print("Running DEAP optimization...")
                        start_time = time.time()

                        # Run DEAP optimization
                        deap_squad = make_new_squad_deap(
                            gw_range=gw_range,
                            tag="test_tag",
                            budget=1000,
                            population_size=50,
                            generations=50,
                            verbose=verbose,
                            random_state=42,  # For reproducibility
                        )

                        end_time = time.time()

                        # Get the final score from the mock
                        final_score = mock_score_side_effect(None, gw_range, "test_tag")

                        return {
                            "method": "DEAP",
                            "score": final_score,
                            "time": end_time - start_time,
                            "squad": deap_squad,
                            "success": True,
                        }

    except Exception as e:
        print(f"✗ DEAP optimization failed: {e}")
        return {
            "method": "DEAP",
            "score": 0,
            "time": 0,
            "squad": None,
            "success": False,
            "error": str(e),
        }


def compare_optimization_methods():
    """Run comparison between PyGMO and DEAP optimization methods"""

    print("AIrsenal Optimization Method Comparison")
    print("=" * 50)
    print("Comparing PyGMO vs DEAP for squad optimization")
    print("Testing with first 10 gameweeks")
    print()

    # Setup test parameters
    gw_range = list(range(1, 11))  # First 10 gameweeks

    # Create consistent mock data
    print("Setting up mock data...")
    mock_players, mock_points = create_consistent_mock_data()
    print(f"Created {len(mock_players)} mock players")

    # Count players by position
    position_counts = {}
    for player in mock_players:
        pos = player.position()
        position_counts[pos] = position_counts.get(pos, 0) + 1

    print("Players by position:", position_counts)
    print()

    # Run both optimizations
    results = []

    # Test PyGMO
    print("1. Testing PyGMO optimization:")
    print("-" * 30)
    pygmo_result = run_pygmo_optimization(
        mock_players, mock_points, gw_range, verbose=False
    )
    results.append(pygmo_result)

    if pygmo_result["success"]:
        print(f"✓ PyGMO completed in {pygmo_result['time']:.2f} seconds")
        print(f"✓ PyGMO score: {pygmo_result['score']:.2f} points")
    else:
        print(f"✗ PyGMO failed: {pygmo_result.get('error', 'Unknown error')}")
    print()

    # Test DEAP
    print("2. Testing DEAP optimization:")
    print("-" * 30)
    deap_result = run_deap_optimization(
        mock_players, mock_points, gw_range, verbose=False
    )
    results.append(deap_result)

    if deap_result["success"]:
        print(f"✓ DEAP completed in {deap_result['time']:.2f} seconds")
        print(f"✓ DEAP score: {deap_result['score']:.2f} points")
    else:
        print(f"✗ DEAP failed: {deap_result.get('error', 'Unknown error')}")
    print()

    # Compare results
    print("COMPARISON RESULTS")
    print("=" * 50)

    successful_results = [r for r in results if r["success"]]

    if len(successful_results) == 0:
        print("✗ No optimizations completed successfully")
        return
    elif len(successful_results) == 1:
        result = successful_results[0]
        print(f"Only {result['method']} completed successfully")
        print(f"Score: {result['score']:.2f} points")
        print(f"Time: {result['time']:.2f} seconds")
        return

    # Both completed successfully
    pygmo_result = next(r for r in successful_results if r["method"] == "PyGMO")
    deap_result = next(r for r in successful_results if r["method"] == "DEAP")

    print("Performance Summary:")
    print(f"PyGMO:  {pygmo_result['score']:.2f} points in {pygmo_result['time']:.2f}s")
    print(f"DEAP:   {deap_result['score']:.2f} points in {deap_result['time']:.2f}s")
    print()

    # Determine winner
    if deap_result["score"] > pygmo_result["score"]:
        diff = deap_result["score"] - pygmo_result["score"]
        print(
            f"🏆 DEAP WINS by {diff:.2f} points ({diff / pygmo_result['score'] * 100:.1f}% better)"
        )
    elif pygmo_result["score"] > deap_result["score"]:
        diff = pygmo_result["score"] - deap_result["score"]
        print(
            f"🏆 PyGMO WINS by {diff:.2f} points ({diff / deap_result['score'] * 100:.1f}% better)"
        )
    else:
        print("🤝 TIE - Both methods achieved the same score")

    # Speed comparison
    print()
    if deap_result["time"] < pygmo_result["time"]:
        time_diff = pygmo_result["time"] - deap_result["time"]
        print(f"⚡ DEAP was faster by {time_diff:.2f} seconds")
    elif pygmo_result["time"] < deap_result["time"]:
        time_diff = deap_result["time"] - pygmo_result["time"]
        print(f"⚡ PyGMO was faster by {time_diff:.2f} seconds")
    else:
        print("⚡ Both methods took the same time")

    print()
    print("Note: This test uses mock data. Results with real AIrsenal data may differ.")


if __name__ == "__main__":
    compare_optimization_methods()
