#!/usr/bin/env python
"""
Quick test of improved DEAP parameters vs current defaults
"""

import sys

sys.path.append("/Users/jroberts/repos/AIrsenal")

import random
import time
from unittest.mock import Mock


# Mock the airsenal modules for testing
def mock_get_discounted_squad_score(*args, **kwargs):
    # Simulate realistic squad scores with some randomness
    base_score = random.uniform(550, 650)
    # Add some noise
    return base_score + random.gauss(0, 10)


def mock_list_players(*args, **kwargs):
    """Mock player list with realistic attributes"""
    players = []
    for i in range(60):  # 60 players per position
        player = Mock()
        player.player_id = f"player_{i}"
        player.name = f"Player {i}"
        player.team = lambda *args: f"Team {i % 20}"
        player.price = lambda *args: random.randint(40, 150)
        player.position = lambda *args: kwargs.get("position", "MID")
        players.append(player)
    return players


def mock_get_predicted_points(*args, **kwargs):
    """Mock predicted points"""
    return {1: 5.0, 2: 4.5, 3: 6.2, 4: 5.8, 5: 4.9}


# Apply mocks
import airsenal.framework.optimization_utils as opt_utils
import airsenal.framework.squad as squad_module
import airsenal.framework.utils as utils

opt_utils.get_discounted_squad_score = mock_get_discounted_squad_score
opt_utils.DEFAULT_SUB_WEIGHTS = {"GK": 0.01, "Outfield": (0.4, 0.1, 0.02)}
utils.list_players = mock_list_players
utils.get_predicted_points_for_player = mock_get_predicted_points
utils.CURRENT_SEASON = "2024-25"
squad_module.TOTAL_PER_POSITION = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}


# Mock Squad class
class MockSquad:
    def __init__(self, budget=1000, season="2024-25"):
        self.budget = budget
        self.season = season
        self.players = []

    def add_player(self, player_or_id, gameweek=None):
        self.players.append(player_or_id)
        return True

    def is_complete(self):
        return len(self.players) >= 15


squad_module.Squad = MockSquad

# Now import our optimization class
from airsenal.framework.optimization_deap import SquadOptDEAP


def test_parameter_set(name, params, n_runs=3):
    """Test a parameter set and return results."""
    print(f"\n🧪 Testing {name}:")
    print(f"   Params: {params}")

    scores = []
    times = []

    for run in range(n_runs):
        try:
            start_time = time.time()

            # Create optimizer
            opt = SquadOptDEAP(
                gw_range=[1, 2, 3, 4, 5],
                tag="test",
                budget=1000,
            )

            # Update tournament size if specified
            if "tournament_size" in params:
                from deap import tools

                opt.toolbox.register(
                    "select", tools.selTournament, tournsize=params["tournament_size"]
                )

            # Run optimization
            best_individual, best_fitness = opt.optimize(
                population_size=params.get("population_size", 50),
                generations=params.get("generations", 50),
                crossover_prob=params.get("crossover_prob", 0.7),
                mutation_prob=params.get("mutation_prob", 0.3),
                verbose=False,
                random_state=42 + run,
            )

            end_time = time.time()

            scores.append(best_fitness)
            times.append(end_time - start_time)

        except Exception as e:
            print(f"   Error in run {run}: {e}")
            scores.append(0)
            times.append(float("inf"))

    avg_score = sum(scores) / len(scores)
    avg_time = sum(times) / len(times)
    best_score = max(scores)

    print(
        f"   Results: {avg_score:.1f} pts (±{max(scores) - min(scores):.1f}) in {avg_time:.2f}s"
    )

    return {
        "name": name,
        "avg_score": avg_score,
        "best_score": best_score,
        "avg_time": avg_time,
        "scores": scores,
        "params": params,
    }


def run_parameter_comparison():
    """Compare different parameter sets."""
    print("🔬 DEAP Parameter Optimization Test")
    print("=" * 50)

    # Define parameter sets to test
    parameter_sets = [
        (
            "Current Default",
            {
                "population_size": 50,
                "generations": 50,
                "crossover_prob": 0.7,
                "mutation_prob": 0.3,
                "tournament_size": 3,
            },
        ),
        (
            "Quick Wins",
            {
                "population_size": 50,
                "generations": 50,
                "crossover_prob": 0.8,  # Increased
                "mutation_prob": 0.2,  # Decreased
                "tournament_size": 5,  # Increased
            },
        ),
        (
            "Fast & Efficient",
            {
                "population_size": 30,  # Smaller
                "generations": 50,
                "crossover_prob": 0.8,
                "mutation_prob": 0.2,
                "tournament_size": 2,
            },
        ),
        (
            "Balanced",
            {
                "population_size": 80,  # Larger
                "generations": 80,  # More generations
                "crossover_prob": 0.75,
                "mutation_prob": 0.25,
                "tournament_size": 4,
            },
        ),
        (
            "High Exploration",
            {
                "population_size": 50,
                "generations": 50,
                "crossover_prob": 0.6,  # Lower crossover
                "mutation_prob": 0.4,  # Higher mutation
                "tournament_size": 2,  # Less selection pressure
            },
        ),
    ]

    # Test each parameter set
    results = []
    for name, params in parameter_sets:
        result = test_parameter_set(name, params)
        results.append(result)

    # Analyze results
    print("\n" + "=" * 50)
    print("📊 RESULTS SUMMARY")
    print("=" * 50)

    # Sort by average score
    results.sort(key=lambda x: x["avg_score"], reverse=True)

    print("\nRanking by Average Score:")
    print("-" * 25)
    for i, result in enumerate(results):
        print(f"{i + 1}. {result['name']}: {result['avg_score']:.1f} pts")
        print(f"   Time: {result['avg_time']:.2f}s")
        print(f"   Best: {result['best_score']:.1f} pts")
        print()

    # Calculate improvements
    baseline = next(r for r in results if r["name"] == "Current Default")
    best_result = results[0]

    if best_result["name"] != "Current Default":
        improvement = (
            (best_result["avg_score"] - baseline["avg_score"]) / baseline["avg_score"]
        ) * 100
        time_change = (
            (best_result["avg_time"] - baseline["avg_time"]) / baseline["avg_time"]
        ) * 100

        print("🏆 BEST CONFIGURATION:")
        print("-" * 25)
        print(f"Winner: {best_result['name']}")
        print(f"Improvement: +{improvement:.1f}% score")
        print(f"Time change: {time_change:+.1f}%")
        print(f"Parameters: {best_result['params']}")

    # Find most efficient
    for result in results:
        result["efficiency"] = result["avg_score"] / result["avg_time"]

    most_efficient = max(results, key=lambda x: x["efficiency"])
    print(f"\n⚡ MOST EFFICIENT: {most_efficient['name']}")
    print(f"   Efficiency: {most_efficient['efficiency']:.1f} pts/sec")


if __name__ == "__main__":
    run_parameter_comparison()
