#!/usr/bin/env python3
"""
Extended performance comparison with multiple runs to get statistical significance.
"""

import random
import statistics
import time
from unittest.mock import Mock

from compare_optimizations import (
    create_consistent_mock_data,
    run_deap_optimization,
    run_pygmo_optimization,
)


def run_multiple_comparisons(num_runs=5, gw_range=None):
    """Run multiple optimization comparisons to get statistical significance"""

    if gw_range is None:
        gw_range = list(range(1, 11))  # First 10 gameweeks

    print(f"Running {num_runs} optimization comparisons")
    print("=" * 50)

    pygmo_scores = []
    pygmo_times = []
    deap_scores = []
    deap_times = []

    # Create base mock data once
    base_players, base_points = create_consistent_mock_data()

    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        print("-" * 20)

        # Create slightly varied data for each run
        random.seed(42 + run)  # Different seed for each run
        mock_players, mock_points = create_consistent_mock_data()

        # Run PyGMO
        pygmo_result = run_pygmo_optimization(mock_players, mock_points, gw_range)
        if pygmo_result["success"]:
            pygmo_scores.append(pygmo_result["score"])
            pygmo_times.append(pygmo_result["time"])
            print(
                f"PyGMO: {pygmo_result['score']:.1f} pts in {pygmo_result['time']:.2f}s"
            )
        else:
            print(f"PyGMO failed: {pygmo_result.get('error', 'Unknown error')}")

        # Run DEAP
        deap_result = run_deap_optimization(mock_players, mock_points, gw_range)
        if deap_result["success"]:
            deap_scores.append(deap_result["score"])
            deap_times.append(deap_result["time"])
            print(
                f"DEAP:  {deap_result['score']:.1f} pts in {deap_result['time']:.2f}s"
            )
        else:
            print(f"DEAP failed: {deap_result.get('error', 'Unknown error')}")

    # Analyze results
    print("\n" + "=" * 50)
    print("STATISTICAL ANALYSIS")
    print("=" * 50)

    if pygmo_scores and deap_scores:
        print(f"\nPyGMO Results ({len(pygmo_scores)} successful runs):")
        print(
            f"  Average Score: {statistics.mean(pygmo_scores):.2f} ± {statistics.stdev(pygmo_scores):.2f}"
        )
        print(f"  Best Score:    {max(pygmo_scores):.2f}")
        print(f"  Worst Score:   {min(pygmo_scores):.2f}")
        print(
            f"  Average Time:  {statistics.mean(pygmo_times):.2f}s ± {statistics.stdev(pygmo_times):.2f}"
        )

        print(f"\nDEAP Results ({len(deap_scores)} successful runs):")
        print(
            f"  Average Score: {statistics.mean(deap_scores):.2f} ± {statistics.stdev(deap_scores):.2f}"
        )
        print(f"  Best Score:    {max(deap_scores):.2f}")
        print(f"  Worst Score:   {min(deap_scores):.2f}")
        print(
            f"  Average Time:  {statistics.mean(deap_times):.2f}s ± {statistics.stdev(deap_times):.2f}"
        )

        # Compare averages
        pygmo_avg = statistics.mean(pygmo_scores)
        deap_avg = statistics.mean(deap_scores)

        print("\nCOMPARISON:")
        if deap_avg > pygmo_avg:
            diff = deap_avg - pygmo_avg
            pct = (diff / pygmo_avg) * 100
            print(f"🏆 DEAP WINS on average by {diff:.2f} points ({pct:.1f}% better)")
        elif pygmo_avg > deap_avg:
            diff = pygmo_avg - deap_avg
            pct = (diff / deap_avg) * 100
            print(f"🏆 PyGMO WINS on average by {diff:.2f} points ({pct:.1f}% better)")
        else:
            print("🤝 TIE on average")

        # Speed comparison
        pygmo_avg_time = statistics.mean(pygmo_times)
        deap_avg_time = statistics.mean(deap_times)

        if deap_avg_time < pygmo_avg_time:
            time_diff = pygmo_avg_time - deap_avg_time
            print(f"⚡ DEAP is faster on average by {time_diff:.2f} seconds")
        elif pygmo_avg_time < deap_avg_time:
            time_diff = deap_avg_time - pygmo_avg_time
            print(f"⚡ PyGMO is faster on average by {time_diff:.2f} seconds")

        # Consistency analysis
        pygmo_cv = statistics.stdev(pygmo_scores) / statistics.mean(pygmo_scores)
        deap_cv = statistics.stdev(deap_scores) / statistics.mean(deap_scores)

        print("\nCONSISTENCY (lower coefficient of variation is better):")
        print(f"PyGMO CV: {pygmo_cv:.3f}")
        print(f"DEAP CV:  {deap_cv:.3f}")

        if deap_cv < pygmo_cv:
            print("📈 DEAP is more consistent")
        elif pygmo_cv < deap_cv:
            print("📈 PyGMO is more consistent")
        else:
            print("📈 Both methods are equally consistent")

    else:
        print("❌ Insufficient data for statistical analysis")


def test_different_parameters():
    """Test both methods with different parameter configurations"""

    print("\nTesting different optimization parameters")
    print("=" * 50)

    # Test configurations
    configs = [
        {"name": "Quick", "pop": 30, "gen": 20},
        {"name": "Medium", "pop": 50, "gen": 50},
        {"name": "Intensive", "pop": 100, "gen": 100},
    ]

    gw_range = list(range(1, 6))  # First 5 gameweeks for faster testing
    mock_players, mock_points = create_consistent_mock_data()

    for config in configs:
        print(
            f"\n{config['name']} Configuration (pop={config['pop']}, gen={config['gen']}):"
        )
        print("-" * 40)

        # Test DEAP with this configuration
        try:
            from unittest.mock import patch

            from airsenal.framework.optimization_deap import SquadOptDEAP

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
                                return [
                                    p for p in mock_players if p.position() == position
                                ]

                            mock_list_players.side_effect = (
                                mock_list_players_side_effect
                            )

                            def mock_get_points_side_effect(player, tag, **kwargs):
                                return {gw: random.uniform(3, 8) for gw in gw_range}

                            mock_get_points.side_effect = mock_get_points_side_effect

                            # Mock squad
                            mock_squad = Mock()
                            mock_squad.add_player.return_value = True
                            mock_squad.is_complete.return_value = True
                            mock_squad_class.return_value = mock_squad

                            # Mock scoring
                            mock_score.return_value = random.uniform(200, 400)

                            # Run DEAP optimization
                            optimizer = SquadOptDEAP(
                                gw_range=gw_range,
                                tag="test_tag",
                                budget=1000,
                            )

                            start_time = time.time()
                            best_individual, best_fitness = optimizer.optimize(
                                population_size=config["pop"],
                                generations=config["gen"],
                                verbose=False,
                                random_state=42,
                            )
                            end_time = time.time()

                            print(
                                f"DEAP: {best_fitness:.2f} pts in {end_time - start_time:.2f}s"
                            )

        except Exception as e:
            print(f"DEAP failed: {e}")


if __name__ == "__main__":
    # Run multiple comparisons
    run_multiple_comparisons(num_runs=3)

    # Test different parameters
    test_different_parameters()
