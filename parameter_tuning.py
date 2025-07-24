#!/usr/bin/env python
"""
Parameter tuning experiment for DEAP optimization in AIrsenal
Tests different parameter combinations to find optimal settings
"""

import random
import sys
import time
from typing import Dict
from unittest.mock import MagicMock, Mock

import numpy as np

# Create mock modules
mock_airsenal = MagicMock()
mock_airsenal.framework.optimization_utils.DEFAULT_SUB_WEIGHTS = {
    "GK": 0.01,
    "Outfield": (0.4, 0.1, 0.02),
}
mock_airsenal.framework.optimization_utils.get_discounted_squad_score = (
    lambda *args, **kwargs: random.uniform(200, 700)
)
mock_airsenal.framework.squad.TOTAL_PER_POSITION = {
    "GK": 2,
    "DEF": 5,
    "MID": 5,
    "FWD": 3,
}
mock_airsenal.framework.squad.Squad = Mock
mock_airsenal.framework.utils.CURRENT_SEASON = "2024-25"
mock_airsenal.framework.utils.get_predicted_points_for_player = (
    lambda *args, **kwargs: {1: 5.0, 2: 4.5, 3: 6.2}
)
mock_airsenal.framework.utils.list_players = lambda *args, **kwargs: [
    Mock() for _ in range(50)
]

sys.modules["airsenal"] = mock_airsenal
sys.modules["airsenal.framework"] = mock_airsenal.framework
sys.modules["airsenal.framework.optimization_utils"] = (
    mock_airsenal.framework.optimization_utils
)
sys.modules["airsenal.framework.squad"] = mock_airsenal.framework.squad
sys.modules["airsenal.framework.utils"] = mock_airsenal.framework.utils

# Now import our DEAP optimization
from airsenal.framework.optimization_deap import SquadOptDEAP


def test_parameter_combination(
    population_size: int,
    generations: int,
    crossover_prob: float,
    mutation_prob: float,
    tournament_size: int = 3,
    n_runs: int = 3,
) -> Dict:
    """Test a specific parameter combination."""
    scores = []
    times = []

    for run in range(n_runs):
        try:
            start_time = time.time()

            # Create optimizer with custom tournament size
            opt = SquadOptDEAP(
                gw_range=[1, 2, 3, 4, 5],
                tag="test",
                budget=1000,
            )

            # Modify tournament size if different from default
            if tournament_size != 3:
                from deap import tools

                opt.toolbox.register(
                    "select", tools.selTournament, tournsize=tournament_size
                )

            best_individual, best_fitness = opt.optimize(
                population_size=population_size,
                generations=generations,
                crossover_prob=crossover_prob,
                mutation_prob=mutation_prob,
                verbose=False,
                random_state=42 + run,  # Different seed per run
            )

            end_time = time.time()

            scores.append(best_fitness)
            times.append(end_time - start_time)

        except Exception as e:
            print(f"Error in run {run}: {e}")
            scores.append(0)
            times.append(float("inf"))

    return {
        "population_size": population_size,
        "generations": generations,
        "crossover_prob": crossover_prob,
        "mutation_prob": mutation_prob,
        "tournament_size": tournament_size,
        "avg_score": np.mean(scores),
        "std_score": np.std(scores),
        "avg_time": np.mean(times),
        "best_score": max(scores),
        "scores": scores,
        "times": times,
    }


def run_parameter_tuning_experiment():
    """Run comprehensive parameter tuning experiment."""
    print("🔬 DEAP Parameter Tuning Experiment")
    print("=" * 50)

    # Define parameter ranges to test
    parameter_sets = [
        # Baseline (current default)
        (50, 50, 0.7, 0.3, 3),
        # Population size variations
        (30, 50, 0.7, 0.3, 3),  # Smaller population
        (100, 50, 0.7, 0.3, 3),  # Larger population
        (150, 50, 0.7, 0.3, 3),  # Much larger population
        # Generation variations
        (50, 30, 0.7, 0.3, 3),  # Fewer generations
        (50, 100, 0.7, 0.3, 3),  # More generations
        (50, 200, 0.7, 0.3, 3),  # Many more generations
        # Crossover probability variations
        (50, 50, 0.5, 0.3, 3),  # Lower crossover
        (50, 50, 0.8, 0.3, 3),  # Higher crossover
        (50, 50, 0.9, 0.3, 3),  # Very high crossover
        # Mutation probability variations
        (50, 50, 0.7, 0.1, 3),  # Lower mutation
        (50, 50, 0.7, 0.5, 3),  # Higher mutation
        (50, 50, 0.7, 0.7, 3),  # Very high mutation
        # Tournament size variations
        (50, 50, 0.7, 0.3, 2),  # Smaller tournament
        (50, 50, 0.7, 0.3, 5),  # Larger tournament
        (50, 50, 0.7, 0.3, 7),  # Much larger tournament
        # Combined optimizations
        (100, 100, 0.8, 0.2, 5),  # Larger, more selective
        (80, 80, 0.6, 0.4, 4),  # Balanced approach
        (120, 60, 0.9, 0.1, 6),  # High crossover, low mutation
    ]

    results = []

    for i, (pop_size, gens, cx_prob, mut_prob, tourn_size) in enumerate(parameter_sets):
        print(
            f"\nTest {i + 1}/{len(parameter_sets)}: pop={pop_size}, gen={gens}, cx={cx_prob}, mut={mut_prob}, tourn={tourn_size}"
        )

        result = test_parameter_combination(
            population_size=pop_size,
            generations=gens,
            crossover_prob=cx_prob,
            mutation_prob=mut_prob,
            tournament_size=tourn_size,
            n_runs=3,
        )

        results.append(result)

        print(f"  Score: {result['avg_score']:.2f} ± {result['std_score']:.2f}")
        print(f"  Time:  {result['avg_time']:.2f}s")
        print(f"  Best:  {result['best_score']:.2f}")

    # Analyze results
    print("\n" + "=" * 50)
    print("📊 PARAMETER TUNING RESULTS")
    print("=" * 50)

    # Sort by average score (descending)
    results.sort(key=lambda x: x["avg_score"], reverse=True)

    print("\nTop 5 Parameter Combinations (by average score):")
    print("-" * 50)
    for i, result in enumerate(results[:5]):
        print(f"{i + 1}. Score: {result['avg_score']:.2f} ± {result['std_score']:.2f}")
        print(f"   Time:  {result['avg_time']:.2f}s")
        print(
            f"   Params: pop={result['population_size']}, gen={result['generations']}"
        )
        print(
            f"           cx={result['crossover_prob']}, mut={result['mutation_prob']}, tourn={result['tournament_size']}"
        )
        print()

    # Find fastest configurations
    results_by_time = sorted(results, key=lambda x: x["avg_time"])
    print("Fastest 3 Configurations:")
    print("-" * 30)
    for i, result in enumerate(results_by_time[:3]):
        print(
            f"{i + 1}. Time: {result['avg_time']:.2f}s, Score: {result['avg_score']:.2f}"
        )
        print(
            f"   Params: pop={result['population_size']}, gen={result['generations']}"
        )
        print()

    # Efficiency analysis (score per second)
    for result in results:
        result["efficiency"] = result["avg_score"] / result["avg_time"]

    results_by_efficiency = sorted(results, key=lambda x: x["efficiency"], reverse=True)
    print("Most Efficient Configurations (score/second):")
    print("-" * 45)
    for i, result in enumerate(results_by_efficiency[:3]):
        print(f"{i + 1}. Efficiency: {result['efficiency']:.2f} pts/sec")
        print(f"   Score: {result['avg_score']:.2f}, Time: {result['avg_time']:.2f}s")
        print(
            f"   Params: pop={result['population_size']}, gen={result['generations']}"
        )
        print()

    # Best overall recommendation
    best_overall = results[0]  # Highest scoring
    most_efficient = results_by_efficiency[0]

    print("🏆 RECOMMENDATIONS:")
    print("-" * 20)
    print(
        f"Best Performance: pop={best_overall['population_size']}, gen={best_overall['generations']}"
    )
    print(
        f"                  cx={best_overall['crossover_prob']}, mut={best_overall['mutation_prob']}"
    )
    print(f"                  tourn={best_overall['tournament_size']}")
    print(
        f"                  → {best_overall['avg_score']:.2f} pts in {best_overall['avg_time']:.2f}s"
    )
    print()
    print(
        f"Most Efficient:   pop={most_efficient['population_size']}, gen={most_efficient['generations']}"
    )
    print(
        f"                  cx={most_efficient['crossover_prob']}, mut={most_efficient['mutation_prob']}"
    )
    print(f"                  tourn={most_efficient['tournament_size']}")
    print(f"                  → {most_efficient['efficiency']:.2f} pts/sec")


if __name__ == "__main__":
    run_parameter_tuning_experiment()
