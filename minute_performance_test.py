#!/usr/bin/env python
"""
1-Minute Performance Test: DEAP vs PyGMO with actual runs
"""

import sys
import time

sys.path.append("/Users/jroberts/repos/AIrsenal")


def create_mock_players():
    """Create mock players for testing"""
    import random

    random.seed(42)  # For reproducible results

    players = []

    # Create realistic player pool
    positions = [
        ("GK", 15, (40, 65), (80, 150)),  # 15 goalkeepers
        ("DEF", 40, (35, 80), (60, 180)),  # 40 defenders
        ("MID", 50, (45, 130), (50, 250)),  # 50 midfielders
        ("FWD", 30, (45, 150), (40, 280)),  # 30 forwards
    ]

    player_id = 0
    for pos, count, price_range, points_range in positions:
        for i in range(count):
            players.append(
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

    return players


def test_deap_60_seconds():
    """Test DEAP optimization targeting 60 seconds"""

    print("🧬 DEAP 60-Second Performance Test")
    print("-" * 35)

    try:
        from airsenal.framework.optimization_deap import SquadOptDEAP

        # Create mock players
        players = create_mock_players()
        print(f"Created {len(players)} mock players")

        # Test different configurations to find ~60 second runtime
        configs = [
            ("Quick", {"population_size": 100, "generations": 100}),
            ("Medium", {"population_size": 150, "generations": 150}),
            ("Long", {"population_size": 200, "generations": 200}),
            ("Extended", {"population_size": 250, "generations": 250}),
            ("Maximum", {"population_size": 300, "generations": 300}),
        ]

        results = []

        for name, params in configs:
            print(
                f"\nTesting {name} config (pop={params['population_size']}, gen={params['generations']})..."
            )

            try:
                # Setup optimizer with mock data
                optimizer = SquadOptDEAP(
                    season="2024-25",
                    players_per_position={"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
                    budget=1000,
                    **params,
                )

                # Override with mock players
                optimizer.available_players = players

                start_time = time.time()
                best_squad, best_score = optimizer.optimize()
                end_time = time.time()

                actual_time = end_time - start_time

                results.append(
                    {
                        "name": name,
                        "params": params,
                        "time": actual_time,
                        "score": best_score,
                        "squad_size": len(best_squad),
                    }
                )

                print(f"  ✓ Completed in {actual_time:.2f}s")
                print(f"  ✓ Best score: {best_score:.2f}")
                print(f"  ✓ Squad size: {len(best_squad)}")

                # If we're getting close to 60 seconds, we can stop here
                if 45 <= actual_time <= 75:
                    print("  🎯 Good match for 60-second target!")
                    break

            except Exception as e:
                print(f"  ❌ Failed: {e}")

        return results

    except ImportError as e:
        print(f"❌ DEAP not available: {e}")
        return []


def run_actual_60_second_test():
    """Run the 60-second performance test"""

    print("⏱️  1-MINUTE OPTIMIZATION PERFORMANCE TEST")
    print("=" * 45)
    print("Testing actual optimization performance with ~60 second runs")
    print()

    # Test DEAP
    deap_results = test_deap_60_seconds()

    print()
    print("📊 RESULTS ANALYSIS")
    print("=" * 20)

    if deap_results:
        print("\nDEAP Performance Results:")
        for result in deap_results:
            efficiency = result["score"] / result["time"]  # Points per second
            print(
                f"  {result['name']:<10}: {result['score']:6.1f} pts in {result['time']:5.1f}s ({efficiency:.1f} pts/s)"
            )

        # Find best for 60-second target
        sixty_sec_candidates = [r for r in deap_results if 45 <= r["time"] <= 75]
        if sixty_sec_candidates:
            best_60s = max(sixty_sec_candidates, key=lambda x: x["score"])
            print("\n🎯 Best for ~60s target:")
            print(
                f"   {best_60s['name']}: {best_60s['score']:.1f} pts in {best_60s['time']:.1f}s"
            )
            print(
                f"   Parameters: pop={best_60s['params']['population_size']}, gen={best_60s['params']['generations']}"
            )

        # Overall best
        best_overall = max(deap_results, key=lambda x: x["score"])
        print("\n🏆 Best overall performance:")
        print(
            f"   {best_overall['name']}: {best_overall['score']:.1f} pts in {best_overall['time']:.1f}s"
        )

    else:
        print("❌ No DEAP results available")

    print()
    print("📝 NOTES:")
    print("- These are actual optimization runs with real DEAP code")
    print("- Mock player data used for reproducible testing")
    print("- Performance scales roughly with population_size × generations")
    print("- For production use, real player database would be used")

    # Return results for further analysis
    return deap_results


if __name__ == "__main__":
    results = run_actual_60_second_test()
