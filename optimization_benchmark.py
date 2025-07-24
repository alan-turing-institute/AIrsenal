#!/usr/bin/env python
"""
Comprehensive optimization benchmark script for AIrsenal squad optimization.

This script runs multiple optimization configurations with both pygmo and DEAP
algorithms to find the best hyperparameters for squad optimization over 10 gameweeks.
Results are saved to a CSV file for analysis.

Focus: Long-running, high-quality optimizations for maximum performance.
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

# Add current directory to Python path for imports
sys.path.insert(0, ".")

from airsenal.framework.optimization_squad import make_new_squad
from airsenal.framework.optimization_utils import (
    DEFAULT_SUB_WEIGHTS,
    check_tag_valid,
    get_discounted_squad_score,
)
from airsenal.framework.season import CURRENT_SEASON
from airsenal.framework.utils import (
    get_latest_prediction_tag,
)

try:
    from airsenal.framework.optimization_deap import make_new_squad_deap

    DEAP_AVAILABLE = True
except ModuleNotFoundError:
    make_new_squad_deap = None
    DEAP_AVAILABLE = False
    print("Warning: DEAP not available. Skipping DEAP optimizations.")

try:
    import pygmo as pg

    PYGMO_AVAILABLE = True
except ModuleNotFoundError:
    pg = None
    PYGMO_AVAILABLE = False
    print("Warning: pygmo not available. Skipping pygmo optimizations.")


# High-performance optimization configurations
OPTIMIZATION_CONFIGS = [
    # DEAP configurations - focused on high performance
    {
        "algorithm": "deap",
        "name": "DEAP_HIGH_QUALITY",
        "population_size": 150,
        "generations": 200,
        "crossover_prob": 0.8,
        "mutation_prob": 0.2,
        "description": "High quality DEAP optimization",
    },
    {
        "algorithm": "deap",
        "name": "DEAP_MAXIMUM_PERFORMANCE",
        "population_size": 200,
        "generations": 400,
        "crossover_prob": 0.65,
        "mutation_prob": 0.35,
        "description": "Maximum performance DEAP (very long)",
    },
    {
        "algorithm": "deap",
        "name": "DEAP_ULTRA_LONG",
        "population_size": 300,
        "generations": 600,
        "crossover_prob": 0.7,
        "mutation_prob": 0.3,
        "description": "Ultra-long DEAP optimization",
    },
    {
        "algorithm": "deap",
        "name": "DEAP_EXPLORATION_FOCUSED",
        "population_size": 400,
        "generations": 300,
        "crossover_prob": 0.5,
        "mutation_prob": 0.5,
        "description": "Maximum exploration DEAP",
    },
    {
        "algorithm": "deap",
        "name": "DEAP_BALANCED_LONG",
        "population_size": 120,
        "generations": 250,
        "crossover_prob": 0.75,
        "mutation_prob": 0.25,
        "description": "Balanced long-running DEAP",
    },
    # Pygmo configurations - focused on high performance
    {
        "algorithm": "pygmo",
        "name": "PYGMO_HIGH_QUALITY",
        "population_size": 150,
        "generations": 200,
        "description": "High quality pygmo optimization",
    },
    {
        "algorithm": "pygmo",
        "name": "PYGMO_MAXIMUM_PERFORMANCE",
        "population_size": 200,
        "generations": 400,
        "description": "Maximum performance pygmo (very long)",
    },
    {
        "algorithm": "pygmo",
        "name": "PYGMO_ULTRA_LONG",
        "population_size": 300,
        "generations": 600,
        "description": "Ultra-long pygmo optimization",
    },
    {
        "algorithm": "pygmo",
        "name": "PYGMO_EXPLORATION_FOCUSED",
        "population_size": 400,
        "generations": 300,
        "description": "Maximum exploration pygmo",
    },
    {
        "algorithm": "pygmo",
        "name": "PYGMO_BALANCED_LONG",
        "population_size": 120,
        "generations": 250,
        "description": "Balanced long-running pygmo",
    },
    # Additional experimental configurations
    {
        "algorithm": "deap",
        "name": "DEAP_EXTREME_EXPLORATION",
        "population_size": 500,
        "generations": 200,
        "crossover_prob": 0.4,
        "mutation_prob": 0.6,
        "description": "Extreme exploration with large population",
    },
    {
        "algorithm": "pygmo",
        "name": "PYGMO_EXTREME_EXPLORATION",
        "population_size": 500,
        "generations": 200,
        "description": "Extreme exploration with large population",
    },
]


def run_optimization(
    config: Dict[str, Any],
    gw_range: List[int],
    tag: str,
    season: str,
    budget: int = 1000,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a single optimization configuration and return results.

    Returns:
        Dictionary containing optimization results including score, timing, etc.
    """
    print(f"\n=== Running {config['name']} ===")
    print(f"Description: {config['description']}")
    print(f"Parameters: {config}")

    start_time = time.time()

    try:
        if config["algorithm"] == "deap":
            if not DEAP_AVAILABLE or make_new_squad_deap is None:
                return {
                    "config_name": config["name"],
                    "algorithm": config["algorithm"],
                    "status": "FAILED",
                    "error": "DEAP not available",
                    "score": None,
                    "runtime_seconds": None,
                }

            best_squad = make_new_squad_deap(
                gw_range=gw_range,
                tag=tag,
                budget=budget,
                season=season,
                population_size=config["population_size"],
                generations=config["generations"],
                crossover_prob=config.get("crossover_prob", 0.7),
                mutation_prob=config.get("mutation_prob", 0.3),
                verbose=verbose,
                remove_zero=True,
                sub_weights=DEFAULT_SUB_WEIGHTS,
            )

        elif config["algorithm"] == "pygmo":
            if not PYGMO_AVAILABLE or pg is None:
                return {
                    "config_name": config["name"],
                    "algorithm": config["algorithm"],
                    "status": "FAILED",
                    "error": "pygmo not available",
                    "score": None,
                    "runtime_seconds": None,
                }

            # Use the correct pygmo algorithm name
            uda = pg.sga(gen=config["generations"])
            best_squad = make_new_squad(
                gw_range=gw_range,
                tag=tag,
                budget=budget,
                season=season,
                algorithm="genetic",
                uda=uda,
                population_size=config["population_size"],
                verbose=verbose,
                remove_zero=True,
                sub_weights=DEFAULT_SUB_WEIGHTS,
            )

        else:
            return {
                "config_name": config["name"],
                "algorithm": config["algorithm"],
                "status": "FAILED",
                "error": f"Unknown algorithm: {config['algorithm']}",
                "score": None,
                "runtime_seconds": None,
            }

        end_time = time.time()
        runtime = end_time - start_time

        if best_squad is None:
            return {
                "config_name": config["name"],
                "algorithm": config["algorithm"],
                "status": "FAILED",
                "error": "Squad optimization returned None",
                "score": None,
                "runtime_seconds": runtime,
            }

        # Calculate final score
        optimized_score = get_discounted_squad_score(
            best_squad,
            gw_range,
            tag,
            gw_range[0],
            sub_weights=DEFAULT_SUB_WEIGHTS,
        )

        next_points = best_squad.get_expected_points(gw_range[0], tag)

        print("✓ Optimization completed successfully!")
        print(
            f"  Total score (GW {min(gw_range)}-{max(gw_range)}): {optimized_score:.2f}"
        )
        print(f"  Expected points for GW {gw_range[0]}: {next_points:.2f}")
        print(f"  Runtime: {runtime:.1f} seconds ({runtime / 60:.1f} minutes)")

        return {
            "config_name": config["name"],
            "algorithm": config["algorithm"],
            "status": "SUCCESS",
            "error": None,
            "score": optimized_score,
            "gw_start_points": next_points,
            "runtime_seconds": runtime,
            "runtime_minutes": runtime / 60,
            "population_size": config["population_size"],
            "generations": config["generations"],
            "crossover_prob": config.get("crossover_prob", "N/A"),
            "mutation_prob": config.get("mutation_prob", "N/A"),
            "description": config["description"],
            "budget": budget,
            "gameweeks": f"{min(gw_range)}-{max(gw_range)}",
            "squad_cost": best_squad.budget,
            "money_left": budget - best_squad.budget,
        }

    except Exception as e:
        end_time = time.time()
        runtime = end_time - start_time

        print(f"✗ Optimization failed: {str(e)}")
        print(f"  Runtime before failure: {runtime:.1f} seconds")

        return {
            "config_name": config["name"],
            "algorithm": config["algorithm"],
            "status": "FAILED",
            "error": str(e),
            "score": None,
            "runtime_seconds": runtime,
        }


def save_results_to_csv(results: List[Dict[str, Any]], filename: str):
    """Save optimization results to CSV file."""
    if not results:
        print("No results to save.")
        return

    print(f"\nSaving results to {filename}...")

    # Get all possible field names from all results
    fieldnames = set()
    for result in results:
        fieldnames.update(result.keys())
    fieldnames = sorted(list(fieldnames))

    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"✓ Results saved to {filename}")


def print_summary(results: List[Dict[str, Any]]):
    """Print a summary of optimization results."""
    successful_results = [r for r in results if r["status"] == "SUCCESS"]
    failed_results = [r for r in results if r["status"] == "FAILED"]

    print(f"\n{'=' * 60}")
    print("OPTIMIZATION BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total runs: {len(results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")

    if successful_results:
        print("\nTOP PERFORMING CONFIGURATIONS:")
        print("-" * 60)
        # Sort by score (descending)
        sorted_results = sorted(
            successful_results, key=lambda x: x["score"], reverse=True
        )

        for i, result in enumerate(sorted_results[:10], 1):  # Top 10
            runtime_str = (
                f"{result['runtime_minutes']:.1f}min"
                if result.get("runtime_minutes")
                else "N/A"
            )
            print(
                f"{i:2d}. {result['config_name']:25s} | "
                f"Score: {result['score']:7.2f} | "
                f"Runtime: {runtime_str:8s} | "
                f"Algorithm: {result['algorithm']}"
            )

        print("\nBEST OVERALL:")
        best = sorted_results[0]
        print(f"  Config: {best['config_name']}")
        print(f"  Algorithm: {best['algorithm']}")
        print(f"  Score: {best['score']:.2f} points")
        print(f"  Runtime: {best.get('runtime_minutes', 0):.1f} minutes")
        print(f"  Description: {best['description']}")

    if failed_results:
        print("\nFAILED RUNS:")
        print("-" * 60)
        for result in failed_results:
            print(f"  {result['config_name']:25s} | Error: {result['error']}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive squad optimization benchmark for AIrsenal"
    )
    parser.add_argument(
        "--season", help="Season in format e.g. 2324", default=CURRENT_SEASON
    )
    parser.add_argument(
        "--gameweek_start", help="Starting gameweek", type=int, default=1
    )
    parser.add_argument(
        "--num_gameweeks",
        help="Number of gameweeks to optimize for",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--budget", help="Budget in 0.1 millions", type=int, default=1000
    )
    parser.add_argument(
        "--output",
        help="Output CSV filename",
        default=f"optimization_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )
    parser.add_argument(
        "--configs",
        help="Comma-separated list of config names to run (default: all)",
        default="all",
    )
    parser.add_argument(
        "--dry-run",
        help="Print configurations without running optimizations",
        action="store_true",
    )
    parser.add_argument(
        "--verbose", help="Verbose optimization output", action="store_true"
    )

    args = parser.parse_args()

    # Setup
    season = args.season
    gameweek_start = args.gameweek_start
    gw_range = list(range(gameweek_start, gameweek_start + args.num_gameweeks))
    budget = args.budget

    print("AIrsenal Optimization Benchmark")
    print(f"Season: {season}")
    print(f"Gameweeks: {min(gw_range)} to {max(gw_range)} ({len(gw_range)} gameweeks)")
    print(f"Budget: £{budget / 10:.1f}m")
    print(f"Output file: {args.output}")

    # Filter configurations if specified
    if args.configs != "all":
        config_names = [name.strip() for name in args.configs.split(",")]
        configs_to_run = [c for c in OPTIMIZATION_CONFIGS if c["name"] in config_names]
        if not configs_to_run:
            print(f"Error: No matching configurations found for: {config_names}")
            print(
                f"Available configurations: {[c['name'] for c in OPTIMIZATION_CONFIGS]}"
            )
            sys.exit(1)
    else:
        configs_to_run = OPTIMIZATION_CONFIGS

    print(f"\nConfigurations to run: {len(configs_to_run)}")
    for config in configs_to_run:
        runtime_est = (
            config["population_size"] * config["generations"]
        ) / 1000  # Rough estimate in minutes
        print(
            f"  - {config['name']:25s} ({config['algorithm']:5s}) | "
            f"Pop: {config['population_size']:3d} | "
            f"Gen: {config['generations']:3d} | "
            f"Est: ~{runtime_est:.0f}min"
        )

    total_est_time = sum(
        (c["population_size"] * c["generations"]) / 1000 for c in configs_to_run
    )
    print(
        f"\nEstimated total runtime: ~{total_est_time:.0f} minutes ({total_est_time / 60:.1f} hours)"
    )

    if args.dry_run:
        print("\nDry run mode - exiting without running optimizations.")
        return

    # Check database
    try:
        tag = get_latest_prediction_tag(season)
        if not check_tag_valid(tag, gw_range, season=season):
            print(
                "ERROR: Database does not contain predictions for the specified gameweeks."
            )
            print("Please run 'airsenal_run_prediction' first.")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to validate predictions database: {e}")
        sys.exit(1)

    print(f"\n✓ Database validated for gameweeks {min(gw_range)}-{max(gw_range)}")
    print(f"Using prediction tag: {tag}")

    # Run optimizations
    print("\nStarting optimization benchmark...")
    print(f"Time started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = []
    for i, config in enumerate(configs_to_run, 1):
        print(f"\n[{i}/{len(configs_to_run)}] Running {config['name']}...")

        result = run_optimization(
            config=config,
            gw_range=gw_range,
            tag=tag,
            season=season,
            budget=budget,
            verbose=args.verbose,
        )

        result["run_order"] = i
        result["timestamp"] = datetime.now().isoformat()
        results.append(result)

        # Save intermediate results
        if i % 3 == 0 or i == len(configs_to_run):  # Save every 3 runs and at the end
            save_results_to_csv(results, args.output)

    print(f"\nTime completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Final save and summary
    save_results_to_csv(results, args.output)
    print_summary(results)

    print(f"\n✓ Benchmark completed! Results saved to {args.output}")


if __name__ == "__main__":
    main()
