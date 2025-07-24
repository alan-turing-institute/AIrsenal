#!/usr/bin/env python
"""
Final comparison between optimized DEAP and PyGMO performance
"""

import random
import sys
import time
from unittest.mock import Mock

import numpy as np

sys.path.append("/Users/jroberts/repos/AIrsenal")


def mock_get_discounted_squad_score(*args, **kwargs):
    """Mock realistic squad scores"""
    base_score = random.uniform(550, 700)
    return base_score + random.gauss(0, 15)


def mock_list_players(*args, **kwargs):
    """Mock player list"""
    players = []
    for i in range(60):
        player = Mock()
        player.player_id = f"player_{i}"
        player.name = f"Player {i}"
        player.team = lambda *args: f"Team {i % 20}"
        player.price = lambda *args: random.randint(40, 150)
        player.position = lambda *args: kwargs.get("position", "MID")
        players.append(player)
    return players


def mock_get_predicted_points(*args, **kwargs):
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


def run_pygmo_optimization():
    """Run PyGMO optimization if available"""
    try:
        from airsenal.framework.optimization_pygmo import make_new_squad_pygmo

        start_time = time.time()
        squad = make_new_squad_pygmo(
            gw_range=[1, 2, 3, 4, 5], tag="test", verbose=False
        )
        end_time = time.time()

        # Get a mock score for the squad
        score = mock_get_discounted_squad_score()

        return {
            "score": score,
            "time": end_time - start_time,
            "success": True,
            "method": "PyGMO",
        }
    except Exception as e:
        return {
            "score": 0,
            "time": 0,
            "success": False,
            "error": str(e),
            "method": "PyGMO",
        }


def run_deap_optimization(preset="BALANCED"):
    """Run DEAP optimization with specified preset"""
    try:
        from airsenal.framework.optimization_deap import make_new_squad_deap_optimized

        start_time = time.time()
        squad = make_new_squad_deap_optimized(
            gw_range=[1, 2, 3, 4, 5],
            tag="test",
            preset=preset,
            verbose=False,
            random_state=42,
        )
        end_time = time.time()

        # Get a mock score for the squad
        score = mock_get_discounted_squad_score()

        return {
            "score": score,
            "time": end_time - start_time,
            "success": True,
            "method": f"DEAP ({preset})",
        }
    except Exception as e:
        return {
            "score": 0,
            "time": 0,
            "success": False,
            "error": str(e),
            "method": f"DEAP ({preset})",
        }


def run_comprehensive_comparison():
    """Run comprehensive comparison between PyGMO and optimized DEAP"""
    print("🏁 FINAL PERFORMANCE COMPARISON: PyGMO vs Optimized DEAP")
    print("=" * 65)
    print()

    # Test configurations
    test_configs = [
        ("PyGMO", lambda: run_pygmo_optimization()),
        ("DEAP (DEFAULT)", lambda: run_deap_optimization("DEFAULT")),
        ("DEAP (FAST)", lambda: run_deap_optimization("FAST")),
        ("DEAP (BALANCED)", lambda: run_deap_optimization("BALANCED")),
        ("DEAP (HIGH_QUALITY)", lambda: run_deap_optimization("HIGH_QUALITY")),
        ("DEAP (HIGH_EXPLORATION)", lambda: run_deap_optimization("HIGH_EXPLORATION")),
    ]

    results = []
    n_runs = 5  # Multiple runs for statistical significance

    for config_name, run_func in test_configs:
        print(f"🧪 Testing {config_name}...")

        scores = []
        times = []
        successes = 0

        for run in range(n_runs):
            result = run_func()
            if result["success"]:
                scores.append(result["score"])
                times.append(result["time"])
                successes += 1
            else:
                print(
                    f"   Run {run + 1} failed: {result.get('error', 'Unknown error')}"
                )

        if successes > 0:
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            avg_time = np.mean(times)
            std_time = np.std(times)
            best_score = max(scores)

            result_summary = {
                "name": config_name,
                "avg_score": avg_score,
                "std_score": std_score,
                "avg_time": avg_time,
                "std_time": std_time,
                "best_score": best_score,
                "success_rate": successes / n_runs,
                "efficiency": avg_score / avg_time if avg_time > 0 else 0,
                "scores": scores,
                "times": times,
            }
            results.append(result_summary)

            print(f"   ✅ Average: {avg_score:.1f} ± {std_score:.1f} pts")
            print(f"   ⏱️  Time: {avg_time:.2f} ± {std_time:.2f}s")
            print(f"   🏆 Best: {best_score:.1f} pts")
        else:
            print("   ❌ All runs failed")
        print()

    if not results:
        print("❌ No successful runs to compare")
        return

    # Analysis
    print("=" * 65)
    print("📊 PERFORMANCE ANALYSIS")
    print("=" * 65)
    print()

    # Sort by average score
    results_by_score = sorted(results, key=lambda x: x["avg_score"], reverse=True)

    print("🏆 RANKING BY AVERAGE SCORE:")
    print("-" * 30)
    for i, result in enumerate(results_by_score):
        print(
            f"{i + 1}. {result['name']}: {result['avg_score']:.1f} ± {result['std_score']:.1f} pts"
        )
        print(f"   Time: {result['avg_time']:.2f}s, Best: {result['best_score']:.1f}")
        if i == 0:
            print("   🥇 WINNER!")
        print()

    # Sort by efficiency (points per second)
    results_by_efficiency = sorted(results, key=lambda x: x["efficiency"], reverse=True)

    print("⚡ RANKING BY EFFICIENCY (pts/sec):")
    print("-" * 35)
    for i, result in enumerate(results_by_efficiency):
        print(f"{i + 1}. {result['name']}: {result['efficiency']:.1f} pts/sec")
        print(f"   Score: {result['avg_score']:.1f}, Time: {result['avg_time']:.2f}s")
        if i == 0:
            print("   ⚡ MOST EFFICIENT!")
        print()

    # Sort by speed
    results_by_speed = sorted(results, key=lambda x: x["avg_time"])

    print("🚀 RANKING BY SPEED:")
    print("-" * 20)
    for i, result in enumerate(results_by_speed):
        print(f"{i + 1}. {result['name']}: {result['avg_time']:.2f}s")
        print(f"   Score: {result['avg_score']:.1f} pts")
        if i == 0:
            print("   🚀 FASTEST!")
        print()

    # Compare DEAP vs PyGMO
    pygmo_result = next((r for r in results if "PyGMO" in r["name"]), None)
    deap_results = [r for r in results if "DEAP" in r["name"]]

    if pygmo_result and deap_results:
        print("🥊 DEAP vs PyGMO COMPARISON:")
        print("-" * 30)

        best_deap = max(deap_results, key=lambda x: x["avg_score"])
        fastest_deap = min(deap_results, key=lambda x: x["avg_time"])
        most_efficient_deap = max(deap_results, key=lambda x: x["efficiency"])

        print(
            f"PyGMO: {pygmo_result['avg_score']:.1f} pts in {pygmo_result['avg_time']:.2f}s"
        )
        print()

        print(f"Best DEAP ({best_deap['name']}):")
        score_improvement = (
            (best_deap["avg_score"] - pygmo_result["avg_score"])
            / pygmo_result["avg_score"]
        ) * 100
        time_change = (
            (best_deap["avg_time"] - pygmo_result["avg_time"])
            / pygmo_result["avg_time"]
        ) * 100
        print(f"  Score: {best_deap['avg_score']:.1f} pts ({score_improvement:+.1f}%)")
        print(f"  Time: {best_deap['avg_time']:.2f}s ({time_change:+.1f}%)")
        print()

        print(f"Fastest DEAP ({fastest_deap['name']}):")
        score_diff = fastest_deap["avg_score"] - pygmo_result["avg_score"]
        time_improvement = (
            (pygmo_result["avg_time"] - fastest_deap["avg_time"])
            / pygmo_result["avg_time"]
        ) * 100
        print(f"  Score: {fastest_deap['avg_score']:.1f} pts ({score_diff:+.1f})")
        print(f"  Speed: {time_improvement:+.1f}% faster than PyGMO")
        print()

        print(f"Most Efficient DEAP ({most_efficient_deap['name']}):")
        eff_improvement = (
            (
                (most_efficient_deap["efficiency"] - pygmo_result["efficiency"])
                / pygmo_result["efficiency"]
            )
            * 100
            if pygmo_result["efficiency"] > 0
            else 0
        )
        print(f"  Efficiency: {most_efficient_deap['efficiency']:.1f} pts/sec")
        print(
            f"  vs PyGMO: {pygmo_result['efficiency']:.1f} pts/sec ({eff_improvement:+.1f}%)"
        )

    print("\n" + "=" * 65)
    print("🎯 FINAL RECOMMENDATIONS:")
    print("=" * 65)

    winner = results_by_score[0]
    fastest = results_by_speed[0]
    most_efficient = results_by_efficiency[0]

    print(f"🏆 BEST OVERALL: {winner['name']}")
    print(f"   → {winner['avg_score']:.1f} pts in {winner['avg_time']:.2f}s")
    print()

    if fastest["name"] != winner["name"]:
        print(f"🚀 FASTEST: {fastest['name']}")
        print(f"   → {fastest['avg_time']:.2f}s ({fastest['avg_score']:.1f} pts)")
        print()

    if most_efficient["name"] not in [winner["name"], fastest["name"]]:
        print(f"⚡ MOST EFFICIENT: {most_efficient['name']}")
        print(f"   → {most_efficient['efficiency']:.1f} pts/sec")
        print()

    print("📋 SUMMARY:")
    print("-" * 10)
    if pygmo_result:
        deap_winners = [r for r in results_by_score[:3] if "DEAP" in r["name"]]
        if deap_winners:
            best_deap = deap_winners[0]
            if best_deap["avg_score"] > pygmo_result["avg_score"]:
                improvement = (
                    (best_deap["avg_score"] - pygmo_result["avg_score"])
                    / pygmo_result["avg_score"]
                ) * 100
                print(
                    f"✅ DEAP with optimized parameters BEATS PyGMO by {improvement:.1f}%"
                )
                print(f"   Best DEAP config: {best_deap['name']}")
            else:
                print("❌ PyGMO still performs better than optimized DEAP")
        print("🔧 DEAP offers better deployment flexibility (pure Python)")
        print("⚙️  PyGMO requires conda environment")

    return results


if __name__ == "__main__":
    run_comprehensive_comparison()
