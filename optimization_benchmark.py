#!/usr/bin/env python
"""
DEAP optimization benchmark script for AIrsenal squad optimization.

This script runs multiple DEAP optimization configurations to find the best
hyperparameters for squad optimization over 10 gameweeks, including the new
crossover_indpb, mutation_indpb, and tournament_size parameters.
Results are saved to a CSV file for analysis.

Focus: Comprehensive DEAP parameter exploration for maximum performance.
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

# Add current directory to Python path for imports
sys.path.insert(0, ".")

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
    print("Error: DEAP not available. Please install with 'pip install deap'")
    sys.exit(1)


# DEAP optimization configurations - comprehensive parameter exploration
OPTIMIZATION_CONFIGS = [
    # Baseline configurations
    {
        "algorithm": "deap",
        "name": "DEAP_BASELINE",
        "population_size": 100,
        "generations": 100,
        "crossover_prob": 0.7,
        "mutation_prob": 0.3,
        "crossover_indpb": 0.5,
        "mutation_indpb": 0.1,
        "tournament_size": 3,
        "description": "Baseline DEAP configuration",
    },
    # High quality configurations
    {
        "algorithm": "deap",
        "name": "DEAP_HIGH_QUALITY",
        "population_size": 150,
        "generations": 200,
        "crossover_prob": 0.8,
        "mutation_prob": 0.2,
        "crossover_indpb": 0.5,
        "mutation_indpb": 0.1,
        "tournament_size": 5,
        "description": "High quality DEAP optimization",
    },
    # Exploration vs exploitation variations
    {
        "algorithm": "deap",
        "name": "DEAP_HIGH_EXPLORATION",
        "population_size": 120,
        "generations": 150,
        "crossover_prob": 0.6,
        "mutation_prob": 0.4,
        "crossover_indpb": 0.7,
        "mutation_indpb": 0.2,
        "tournament_size": 2,
        "description": "High exploration - more mutation, larger crossover_indpb",
    },
    {
        "algorithm": "deap",
        "name": "DEAP_HIGH_EXPLOITATION",
        "population_size": 120,
        "generations": 150,
        "crossover_prob": 0.9,
        "mutation_prob": 0.1,
        "crossover_indpb": 0.3,
        "mutation_indpb": 0.05,
        "tournament_size": 7,
        "description": "High exploitation - more crossover, larger tournament",
    },
    # Tournament size variations
    {
        "algorithm": "deap",
        "name": "DEAP_SMALL_TOURNAMENT",
        "population_size": 100,
        "generations": 120,
        "crossover_prob": 0.75,
        "mutation_prob": 0.25,
        "crossover_indpb": 0.5,
        "mutation_indpb": 0.1,
        "tournament_size": 2,
        "description": "Small tournament size for diversity",
    },
    {
        "algorithm": "deap",
        "name": "DEAP_LARGE_TOURNAMENT",
        "population_size": 100,
        "generations": 120,
        "crossover_prob": 0.75,
        "mutation_prob": 0.25,
        "crossover_indpb": 0.5,
        "mutation_indpb": 0.1,
        "tournament_size": 8,
        "description": "Large tournament size for selection pressure",
    },
    # Crossover indpb variations
    {
        "algorithm": "deap",
        "name": "DEAP_LOW_CROSSOVER_INDPB",
        "population_size": 100,
        "generations": 120,
        "crossover_prob": 0.8,
        "mutation_prob": 0.2,
        "crossover_indpb": 0.2,
        "mutation_indpb": 0.1,
        "tournament_size": 3,
        "description": "Low crossover indpb - conservative crossover",
    },
    {
        "algorithm": "deap",
        "name": "DEAP_HIGH_CROSSOVER_INDPB",
        "population_size": 100,
        "generations": 120,
        "crossover_prob": 0.8,
        "mutation_prob": 0.2,
        "crossover_indpb": 0.8,
        "mutation_indpb": 0.1,
        "tournament_size": 3,
        "description": "High crossover indpb - aggressive crossover",
    },
    # Mutation indpb variations
    {
        "algorithm": "deap",
        "name": "DEAP_LOW_MUTATION_INDPB",
        "population_size": 100,
        "generations": 120,
        "crossover_prob": 0.7,
        "mutation_prob": 0.3,
        "crossover_indpb": 0.5,
        "mutation_indpb": 0.05,
        "tournament_size": 3,
        "description": "Low mutation indpb - conservative mutation",
    },
    {
        "algorithm": "deap",
        "name": "DEAP_HIGH_MUTATION_INDPB",
        "population_size": 100,
        "generations": 120,
        "crossover_prob": 0.7,
        "mutation_prob": 0.3,
        "crossover_indpb": 0.5,
        "mutation_indpb": 0.3,
        "tournament_size": 3,
        "description": "High mutation indpb - aggressive mutation",
    },
    # Long-running configurations
    {
        "algorithm": "deap",
        "name": "DEAP_ULTRA_LONG",
        "population_size": 200,
        "generations": 300,
        "crossover_prob": 0.75,
        "mutation_prob": 0.25,
        "crossover_indpb": 0.6,
        "mutation_indpb": 0.15,
        "tournament_size": 4,
        "description": "Ultra-long DEAP optimization",
    },
    # Balanced configurations with different parameter combinations
    {
        "algorithm": "deap",
        "name": "DEAP_BALANCED_A",
        "population_size": 120,
        "generations": 150,
        "crossover_prob": 0.7,
        "mutation_prob": 0.3,
        "crossover_indpb": 0.4,
        "mutation_indpb": 0.12,
        "tournament_size": 4,
        "description": "Balanced configuration variant A",
    },
    {
        "algorithm": "deap",
        "name": "DEAP_BALANCED_B",
        "population_size": 120,
        "generations": 150,
        "crossover_prob": 0.8,
        "mutation_prob": 0.2,
        "crossover_indpb": 0.6,
        "mutation_indpb": 0.08,
        "tournament_size": 5,
        "description": "Balanced configuration variant B",
    },
    # Extreme configurations
    {
        "algorithm": "deap",
        "name": "DEAP_EXTREME_DIVERSITY",
        "population_size": 200,
        "generations": 150,
        "crossover_prob": 0.5,
        "mutation_prob": 0.5,
        "crossover_indpb": 0.9,
        "mutation_indpb": 0.4,
        "tournament_size": 2,
        "description": "Extreme diversity - maximum exploration",
    },
    {
        "algorithm": "deap",
        "name": "DEAP_EXTREME_CONVERGENCE",
        "population_size": 80,
        "generations": 200,
        "crossover_prob": 0.95,
        "mutation_prob": 0.05,
        "crossover_indpb": 0.2,
        "mutation_indpb": 0.02,
        "tournament_size": 10,
        "description": "Extreme convergence - maximum exploitation",
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
    Run a single DEAP optimization configuration and return results.

    Returns:
        Dictionary containing optimization results including score, timing, etc.
    """
    print(f"\n=== Running {config['name']} ===")
    print(f"Description: {config['description']}")
    print(f"Parameters: {config}")

    start_time = time.time()

    try:
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
            crossover_indpb=config.get("crossover_indpb", 0.5),
            mutation_indpb=config.get("mutation_indpb", 0.1),
            tournament_size=config.get("tournament_size", 3),
            verbose=verbose,
            remove_zero=True,
            sub_weights=DEFAULT_SUB_WEIGHTS,
        )

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
            "crossover_indpb": config.get("crossover_indpb", "N/A"),
            "mutation_indpb": config.get("mutation_indpb", "N/A"),
            "tournament_size": config.get("tournament_size", "N/A"),
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
    print("DEAP OPTIMIZATION BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total runs: {len(results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")

    if successful_results:
        print("\nTOP PERFORMING DEAP CONFIGURATIONS:")
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
                f"CX_indpb: {result.get('crossover_indpb', 'N/A'):4} | "
                f"Mut_indpb: {result.get('mutation_indpb', 'N/A'):5} | "
                f"Tour: {result.get('tournament_size', 'N/A'):2}"
            )

        print("\nBEST OVERALL DEAP CONFIGURATION:")
        best = sorted_results[0]
        print(f"  Config: {best['config_name']}")
        print(f"  Score: {best['score']:.2f} points")
        print(f"  Runtime: {best.get('runtime_minutes', 0):.1f} minutes")
        print(f"  Population Size: {best.get('population_size', 'N/A')}")
        print(f"  Generations: {best.get('generations', 'N/A')}")
        print(f"  Crossover Prob: {best.get('crossover_prob', 'N/A')}")
        print(f"  Mutation Prob: {best.get('mutation_prob', 'N/A')}")
        print(f"  Crossover Indpb: {best.get('crossover_indpb', 'N/A')}")
        print(f"  Mutation Indpb: {best.get('mutation_indpb', 'N/A')}")
        print(f"  Tournament Size: {best.get('tournament_size', 'N/A')}")
        print(f"  Description: {best['description']}")

    if failed_results:
        print("\nFAILED RUNS:")
        print("-" * 60)
        for result in failed_results:
            print(f"  {result['config_name']:25s} | Error: {result['error']}")


def main():
    parser = argparse.ArgumentParser(
        description="DEAP squad optimization benchmark for AIrsenal - comprehensive parameter exploration"
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

    print("AIrsenal DEAP Optimization Benchmark")
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

    print(f"\nDEAP configurations to run: {len(configs_to_run)}")
    for config in configs_to_run:
        runtime_est = (
            config["population_size"] * config["generations"]
        ) / 1000  # Rough estimate in minutes
        print(
            f"  - {config['name']:25s} | "
            f"Pop: {config['population_size']:3d} | "
            f"Gen: {config['generations']:3d} | "
            f"CX_indpb: {config['crossover_indpb']:4.2f} | "
            f"Mut_indpb: {config['mutation_indpb']:5.2f} | "
            f"Tour: {config['tournament_size']:2d} | "
            f"Est: ~{runtime_est:.0f}min"
        )

    total_est_time = sum(
        (c["population_size"] * c["generations"]) / 1000 for c in configs_to_run
    )
    print(
        f"\nEstimated total runtime: ~{total_est_time:.0f} minutes ({total_est_time / 60:.1f} hours)"
    )

    if args.dry_run:
        print("\nDry run mode - exiting without running DEAP optimizations.")
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
    print("\nStarting DEAP optimization benchmark...")
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

    print(f"\n✓ DEAP benchmark completed! Results saved to {args.output}")


if __name__ == "__main__":
    main()
