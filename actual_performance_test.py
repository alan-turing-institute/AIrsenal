#!/usr/bin/env python
"""
Real performance comparison between DEAP and PyGMO with actual optimization runs
This script runs actual optimizations for ~1 minute each and compares real results
"""

import random
import sys
import time

import numpy as np

sys.path.append("/Users/jroberts/repos/AIrsenal")


def create_mock_environment():
    """Create mock environment for testing without full database"""

    # Mock player data - realistic FPL-style points
    mock_players = []

    # Goalkeepers (2 players)
    for i in range(2):
        mock_players.append(
            {
                "id": i,
                "name": f"GK_{i}",
                "team": f"Team_{i % 10}",
                "position": "GK",
                "price": random.randint(40, 65),  # £4.0-6.5m
                "predicted_points": random.uniform(80, 150),  # Season points
                "minutes_played": random.randint(2500, 3420),  # Match minutes
            }
        )

    # Defenders (5 players)
    for i in range(2, 7):
        mock_players.append(
            {
                "id": i,
                "name": f"DEF_{i}",
                "team": f"Team_{i % 10}",
                "position": "DEF",
                "price": random.randint(35, 80),  # £3.5-8.0m
                "predicted_points": random.uniform(60, 180),  # Season points
                "minutes_played": random.randint(1800, 3420),
            }
        )

    # Midfielders (5 players)
    for i in range(7, 12):
        mock_players.append(
            {
                "id": i,
                "name": f"MID_{i}",
                "team": f"Team_{i % 10}",
                "position": "MID",
                "price": random.randint(45, 130),  # £4.5-13.0m
                "predicted_points": random.uniform(50, 250),  # Season points
                "minutes_played": random.randint(1500, 3420),
            }
        )

    # Forwards (3 players)
    for i in range(12, 15):
        mock_players.append(
            {
                "id": i,
                "name": f"FWD_{i}",
                "team": f"Team_{i % 10}",
                "position": "FWD",
                "price": random.randint(45, 150),  # £4.5-15.0m
                "predicted_points": random.uniform(40, 280),  # Season points
                "minutes_played": random.randint(1200, 3420),
            }
        )

    return mock_players


def create_larger_player_pool():
    """Create a larger player pool for more realistic optimization"""

    mock_players = []

    # Create more realistic player pool sizes
    positions = [
        ("GK", 20, (40, 70), (60, 160)),  # 20 goalkeepers
        ("DEF", 60, (35, 85), (40, 200)),  # 60 defenders
        ("MID", 80, (40, 140), (30, 280)),  # 80 midfielders
        ("FWD", 40, (45, 160), (20, 300)),  # 40 forwards
    ]

    player_id = 0
    for pos, count, price_range, points_range in positions:
        for i in range(count):
            mock_players.append(
                {
                    "id": player_id,
                    "name": f"{pos}_{player_id}",
                    "team": f"Team_{player_id % 20}",  # 20 teams
                    "position": pos,
                    "price": random.randint(*price_range),
                    "predicted_points": random.uniform(*points_range),
                    "minutes_played": random.randint(500, 3420),
                }
            )
            player_id += 1

    return mock_players


def setup_deap_environment():
    """Set up DEAP optimization environment"""
    try:
        from airsenal.framework.optimization_deap import (
            SquadOptDEAP,
            make_new_squad_deap_optimized,
        )

        return True, "DEAP optimization available"
    except ImportError as e:
        return False, f"DEAP not available: {e}"


def setup_pygmo_environment():
    """Set up PyGMO optimization environment"""
    try:
        import pygmo as pg

        from airsenal.framework.optimization_pygmo import make_new_squad

        return True, "PyGMO optimization available"
    except ImportError as e:
        return False, f"PyGMO not available: {e}"


def run_actual_deap_test(players, target_time_seconds=60):
    """Run actual DEAP optimization for specified time"""

    print(f"🧬 Running DEAP optimization (target: {target_time_seconds}s)...")

    try:
        from airsenal.framework.optimization_deap import (
            get_preset_parameters,
            make_new_squad_deap_optimized,
        )

        # Mock get_player_from_id function
        def mock_get_player(player_id, season=None, dbsession=None):
            player_data = next((p for p in players if p["id"] == player_id), None)
            if not player_data:
                return None

            # Create a mock player object
            mock_player = Mock()
            mock_player.player_id = player_data["id"]
            mock_player.name = player_data["name"]
            mock_player.team = player_data["team"]
            mock_player.position = player_data["position"]
            mock_player.purchase_price = player_data["price"]
            mock_player.predicted_points = player_data["predicted_points"]
            return mock_player

        # Patch the function
        import airsenal.framework.optimization_deap

        airsenal.framework.optimization_deap.get_player_from_id = mock_get_player

        # Test different parameter sets to find ~60 second runtime
        test_configs = [
            ("BALANCED", get_preset_parameters("BALANCED")),
            ("HIGH_EXPLORATION", get_preset_parameters("HIGH_EXPLORATION")),
            ("INTENSIVE", get_preset_parameters("INTENSIVE")),
            (
                "CUSTOM_60S",
                {
                    "population_size": 150,
                    "generations": 250,
                    "crossover_prob": 0.65,
                    "mutation_prob": 0.35,
                    "tournament_size": 4,
                },
            ),
        ]

        best_config = None
        best_time_diff = float("inf")

        for config_name, params in test_configs:
            print(f"  Testing {config_name} configuration...")

            start_time = time.time()

            # Run optimization
            squad_ids = make_new_squad_deap_optimized(
                season="2024-25",
                preset=config_name if config_name != "CUSTOM_60S" else None,
                custom_params=params if config_name == "CUSTOM_60S" else None,
                dbsession=None,  # Use mock
            )

            end_time = time.time()
            actual_time = end_time - start_time
            time_diff = abs(actual_time - target_time_seconds)

            # Calculate score
            total_points = sum(
                next(p["predicted_points"] for p in players if p["id"] == pid)
                for pid in squad_ids
            )

            print(f"    Time: {actual_time:.1f}s, Score: {total_points:.1f} pts")

            if time_diff < best_time_diff:
                best_time_diff = time_diff
                best_config = (config_name, params, actual_time, total_points)

            # Don't test more if we're close enough
            if time_diff < 10:  # Within 10 seconds is good enough
                break

        if best_config:
            name, params, actual_time, score = best_config
            print(f"  ✅ Best config: {name} ({actual_time:.1f}s, {score:.1f} pts)")
            return {
                "success": True,
                "config": name,
                "parameters": params,
                "time": actual_time,
                "score": score,
                "squad_ids": squad_ids,
            }
        else:
            return {"success": False, "error": "No suitable configuration found"}

    except Exception as e:
        print(f"  ❌ DEAP test failed: {e}")
        return {"success": False, "error": str(e)}


def run_actual_pygmo_test(players, target_time_seconds=60):
    """Run actual PyGMO optimization for specified time"""

    print(f"🚀 Running PyGMO optimization (target: {target_time_seconds}s)...")

    try:
        from airsenal.framework.optimization_pygmo import make_new_squad

        # Mock get_player_from_id function
        def mock_get_player(player_id, season=None, dbsession=None):
            player_data = next((p for p in players if p["id"] == player_id), None)
            if not player_data:
                return None

            # Create a mock player object
            mock_player = Mock()
            mock_player.player_id = player_data["id"]
            mock_player.name = player_data["name"]
            mock_player.team = player_data["team"]
            mock_player.position = player_data["position"]
            mock_player.purchase_price = player_data["price"]
            mock_player.predicted_points = player_data["predicted_points"]
            return mock_player

        # Patch the function
        import airsenal.framework.optimization_pygmo

        airsenal.framework.optimization_pygmo.get_player_from_id = mock_get_player

        # Test different generation counts to achieve target time
        test_generations = [5000, 10000, 20000, 30000, 50000]

        best_config = None
        best_time_diff = float("inf")

        for generations in test_generations:
            print(f"  Testing {generations} generations...")

            start_time = time.time()

            # Run optimization
            squad_ids = make_new_squad(
                "2024-25",
                tag="test",
                season="2024-25",
                num_generations=generations,
                dbsession=None,  # Use mock
            )

            end_time = time.time()
            actual_time = end_time - start_time
            time_diff = abs(actual_time - target_time_seconds)

            # Calculate score
            total_points = sum(
                next(p["predicted_points"] for p in players if p["id"] == pid)
                for pid in squad_ids
            )

            print(f"    Time: {actual_time:.1f}s, Score: {total_points:.1f} pts")

            if time_diff < best_time_diff:
                best_time_diff = time_diff
                best_config = (generations, actual_time, total_points)

            # Don't test more if we're close enough
            if time_diff < 10:  # Within 10 seconds is good enough
                break

            # Stop if we're taking too long
            if actual_time > target_time_seconds * 1.5:
                break

        if best_config:
            generations, actual_time, score = best_config
            print(
                f"  ✅ Best config: {generations} generations ({actual_time:.1f}s, {score:.1f} pts)"
            )
            return {
                "success": True,
                "generations": generations,
                "time": actual_time,
                "score": score,
                "squad_ids": squad_ids,
            }
        else:
            return {"success": False, "error": "No suitable configuration found"}

    except Exception as e:
        print(f"  ❌ PyGMO test failed: {e}")
        return {"success": False, "error": str(e)}


def run_multiple_tests(players, num_runs=3, target_time=60):
    """Run multiple tests to get average performance"""

    print("\n🔬 RUNNING COMPREHENSIVE PERFORMANCE TEST")
    print("=" * 55)
    print(f"Target time per run: {target_time} seconds")
    print(f"Number of runs: {num_runs}")
    print(f"Player pool size: {len(players)}")
    print()

    # Check availability
    deap_available, deap_msg = setup_deap_environment()
    pygmo_available, pygmo_msg = setup_pygmo_environment()

    print("📋 ENVIRONMENT CHECK:")
    print(f"  DEAP:  {'✅' if deap_available else '❌'} {deap_msg}")
    print(f"  PyGMO: {'✅' if pygmo_available else '❌'} {pygmo_msg}")
    print()

    results = {"deap": [], "pygmo": []}

    # Run DEAP tests
    if deap_available:
        print("🧬 DEAP PERFORMANCE TESTS:")
        print("-" * 30)

        for run in range(num_runs):
            print(f"\nRun {run + 1}/{num_runs}:")
            result = run_actual_deap_test(players, target_time)
            if result["success"]:
                results["deap"].append(result)
            else:
                print(f"  ❌ Run {run + 1} failed: {result['error']}")

    # Run PyGMO tests
    if pygmo_available:
        print("\n🚀 PYGMO PERFORMANCE TESTS:")
        print("-" * 30)

        for run in range(num_runs):
            print(f"\nRun {run + 1}/{num_runs}:")
            result = run_actual_pygmo_test(players, target_time)
            if result["success"]:
                results["pygmo"].append(result)
            else:
                print(f"  ❌ Run {run + 1} failed: {result['error']}")

    return results


def analyze_results(results):
    """Analyze and display the test results"""

    print("\n📊 PERFORMANCE ANALYSIS")
    print("=" * 25)
    print()

    for optimizer, data in results.items():
        if not data:
            print(f"{optimizer.upper()}: No successful runs")
            continue

        scores = [r["score"] for r in data]
        times = [r["time"] for r in data]

        avg_score = np.mean(scores)
        std_score = np.std(scores)
        avg_time = np.mean(times)
        std_time = np.std(times)

        print(f"{optimizer.upper()} RESULTS ({len(data)} runs):")
        print(f"  Average Score: {avg_score:.2f} ± {std_score:.2f} points")
        print(f"  Average Time:  {avg_time:.2f} ± {std_time:.2f} seconds")
        print(f"  Score Range:   {min(scores):.1f} - {max(scores):.1f} points")
        print(f"  Time Range:    {min(times):.1f} - {max(times):.1f} seconds")

        if optimizer == "deap" and data:
            print(f"  Best Config:   {data[0]['config']}")
        elif optimizer == "pygmo" and data:
            print(f"  Best Config:   {data[0]['generations']} generations")
        print()

    # Head-to-head comparison
    if results["deap"] and results["pygmo"]:
        deap_avg = np.mean([r["score"] for r in results["deap"]])
        pygmo_avg = np.mean([r["score"] for r in results["pygmo"]])
        deap_time = np.mean([r["time"] for r in results["deap"]])
        pygmo_time = np.mean([r["time"] for r in results["pygmo"]])

        score_diff = deap_avg - pygmo_avg
        time_diff = deap_time - pygmo_time

        print("🥊 HEAD-TO-HEAD COMPARISON:")
        print("-" * 30)
        print(f"Score Difference: DEAP {score_diff:+.2f} points vs PyGMO")
        print(f"Time Difference:  DEAP {time_diff:+.2f} seconds vs PyGMO")
        print()

        if abs(score_diff) < 5:
            print("🤝 RESULT: Performance is very similar!")
        elif score_diff > 0:
            print("🏆 WINNER: DEAP has better optimization performance")
        else:
            print("🏆 WINNER: PyGMO has better optimization performance")

        print(
            f"Performance gap: {abs(score_diff):.1f} points ({abs(score_diff) / max(deap_avg, pygmo_avg) * 100:.1f}%)"
        )
        print()

    # Recommendations
    print("✅ RECOMMENDATIONS:")
    print("-" * 20)

    if results["deap"] and results["pygmo"]:
        deap_avg = np.mean([r["score"] for r in results["deap"]])
        pygmo_avg = np.mean([r["score"] for r in results["pygmo"]])

        if abs(deap_avg - pygmo_avg) < 5:
            print("Both optimizers perform similarly for 1-minute runs.")
            print("Choose based on deployment requirements:")
            print("  • DEAP: Pure Python, pip installable")
            print("  • PyGMO: Requires conda, potentially faster")
        elif deap_avg > pygmo_avg:
            print("DEAP shows better performance for extended runs.")
            print("Recommended for maximum optimization quality.")
        else:
            print("PyGMO shows better performance for this timeframe.")
            print("Recommended for faster optimization cycles.")
    elif results["deap"]:
        print("Only DEAP was tested successfully.")
        print("DEAP is working and provides pure Python deployment.")
    elif results["pygmo"]:
        print("Only PyGMO was tested successfully.")
        print("PyGMO is working but requires conda environment.")
    else:
        print("No successful optimization runs completed.")
        print("Check environment setup and dependencies.")


def main():
    """Main test execution"""

    print("🚀 ACTUAL OPTIMIZATION PERFORMANCE TEST")
    print("=" * 45)
    print("Testing real optimization performance with ~1 minute runs")
    print()

    # Create realistic player pool
    print("Creating realistic player pool...")
    players = create_larger_player_pool()
    print(f"Created {len(players)} players across 20 teams")
    print(f"  • {len([p for p in players if p['position'] == 'GK'])} Goalkeepers")
    print(f"  • {len([p for p in players if p['position'] == 'DEF'])} Defenders")
    print(f"  • {len([p for p in players if p['position'] == 'MID'])} Midfielders")
    print(f"  • {len([p for p in players if p['position'] == 'FWD'])} Forwards")
    print()

    # Run the tests
    results = run_multiple_tests(players, num_runs=3, target_time=60)

    # Analyze results
    analyze_results(results)

    print("\n🎯 TEST COMPLETE")
    print("Results are based on actual optimization runs, not projections.")


if __name__ == "__main__":
    main()
