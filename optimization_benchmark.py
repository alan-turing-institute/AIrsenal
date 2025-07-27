#!/usr/bin/env python
"""
Comparative optimization benchmark script for AIrsenal squad optimization.

This script runs DEAP optimizations with varying population sizes
and generations (100-500 in steps of 100), repeating each configuration 10 times
to compute statistical performance comparisons.
Results are saved to CSV and plotted for comparison.

Focus: DEAP performance scaling analysis.
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

# Add current directory to Python path for imports
sys.path.insert(0, ".")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from airsenal.framework.optimization_deap import make_new_squad_deap
from airsenal.framework.optimization_utils import (
    DEFAULT_SUB_WEIGHTS,
    check_tag_valid,
    get_discounted_squad_score,
)
from airsenal.framework.season import CURRENT_SEASON
from airsenal.framework.utils import (
    get_latest_prediction_tag,
)


# Generate optimization configurations
def generate_configurations():
    """Generate configurations for population/generation size comparison."""
    configs = []
    sizes = [100, 200, 300, 400, 500]  # Population and generation sizes
    n_repeats = 10  # Number of repetitions for each size

    for size in sizes:
        # DEAP configurations (always available now)
        for repeat in range(n_repeats):
            configs.append(
                {
                    "algorithm": "deap",
                    "name": f"DEAP_SIZE_{size}_REP_{repeat + 1}",
                    "population_size": size,
                    "generations": size,
                    "crossover_prob": 0.7,  # Default values
                    "mutation_prob": 0.3,
                    "crossover_indpb": 0.5,
                    "mutation_indpb": 0.1,
                    "tournament_size": 3,
                    "size_category": size,
                    "repeat": repeat + 1,
                    "description": f"DEAP optimization with size {size}, repeat {repeat + 1}",
                }
            )

    return configs


OPTIMIZATION_CONFIGS = generate_configurations()


def run_optimization(
    config: Dict[str, Any],
    gw_range: List[int],
    tag: str,
    season: str,
    budget: int = 1000,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a single optimization configuration (DEAP) and return results.

    Returns:
        Dictionary containing optimization results including score, timing, etc.
    """
    print(f"\n=== Running {config['name']} ===")
    print(f"Algorithm: {config['algorithm']}")
    print(f"Size: {config['size_category']}, Repeat: {config['repeat']}")

    start_time = time.time()

    try:
        if config["algorithm"] == "deap":
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
        print(f"  Score: {optimized_score:.2f}, Runtime: {runtime:.1f}s")

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
            "size_category": config["size_category"],
            "repeat": config["repeat"],
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
            "size_category": config.get("size_category"),
            "repeat": config.get("repeat"),
        }


def compute_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute statistics for each algorithm and size combination."""
    stats_data = {}

    # Filter successful results
    successful_results = [r for r in results if r["status"] == "SUCCESS"]

    # Group by algorithm and size
    for result in successful_results:
        algorithm = result["algorithm"]
        size = result["size_category"]
        score = result["score"]
        runtime = result["runtime_seconds"]

        key = (algorithm, size)
        if key not in stats_data:
            stats_data[key] = {"scores": [], "runtimes": []}

        stats_data[key]["scores"].append(score)
        stats_data[key]["runtimes"].append(runtime)

    # Compute statistics for each group
    statistics = {}
    for (algorithm, size), data in stats_data.items():
        scores = np.array(data["scores"])
        runtimes = np.array(data["runtimes"])

        # Compute confidence intervals (95%)
        score_mean = np.mean(scores)
        score_std = np.std(scores, ddof=1)
        score_ci = stats.t.interval(
            0.95,
            len(scores) - 1,
            loc=score_mean,
            scale=score_std / np.sqrt(len(scores)),
        )

        runtime_mean = np.mean(runtimes)
        runtime_std = np.std(runtimes, ddof=1)
        runtime_ci = stats.t.interval(
            0.95,
            len(runtimes) - 1,
            loc=runtime_mean,
            scale=runtime_std / np.sqrt(len(runtimes)),
        )

        statistics[(algorithm, size)] = {
            "n_samples": len(scores),
            "score_mean": score_mean,
            "score_std": score_std,
            "score_ci_lower": score_ci[0],
            "score_ci_upper": score_ci[1],
            "runtime_mean": runtime_mean,
            "runtime_std": runtime_std,
            "runtime_ci_lower": runtime_ci[0],
            "runtime_ci_upper": runtime_ci[1],
        }

    return statistics


def create_performance_plot(statistics: Dict[str, Any], output_filename: str):
    """Create performance comparison plots."""
    # Extract data for plotting
    deap_sizes = []
    deap_scores = []
    deap_score_errors = []
    deap_runtimes = []
    deap_runtime_errors = []

    for (algorithm, size), stat_dict in statistics.items():
        if algorithm == "deap":
            deap_sizes.append(size)
            deap_scores.append(stat_dict["score_mean"])
            deap_score_errors.append(
                [
                    stat_dict["score_mean"] - stat_dict["score_ci_lower"],
                    stat_dict["score_ci_upper"] - stat_dict["score_mean"],
                ]
            )
            deap_runtimes.append(stat_dict["runtime_mean"])
            deap_runtime_errors.append(
                [
                    stat_dict["runtime_mean"] - stat_dict["runtime_ci_lower"],
                    stat_dict["runtime_ci_upper"] - stat_dict["runtime_mean"],
                ]
            )

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Score comparison plot
    if deap_sizes:
        deap_score_errors_t = np.array(deap_score_errors).T
        ax1.errorbar(
            deap_sizes,
            deap_scores,
            yerr=deap_score_errors_t,
            label="DEAP",
            marker="o",
            capsize=5,
            capthick=2,
        )

    ax1.set_xlabel("Population Size / Generations")
    ax1.set_ylabel("Squad Score (Points)")
    ax1.set_title("Optimization Performance vs. Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Runtime comparison plot
    if deap_sizes:
        deap_runtime_errors_t = np.array(deap_runtime_errors).T
        ax2.errorbar(
            deap_sizes,
            deap_runtimes,
            yerr=deap_runtime_errors_t,
            label="DEAP",
            marker="o",
            capsize=5,
            capthick=2,
        )

    ax2.set_xlabel("Population Size / Generations")
    ax2.set_ylabel("Runtime (seconds)")
    ax2.set_title("Runtime vs. Size")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Performance plot saved to {output_filename}")


def save_statistics_csv(statistics: Dict[str, Any], filename: str):
    """Save statistics summary to CSV."""
    stats_rows = []
    for (algorithm, size), stat_dict in statistics.items():
        stats_rows.append(
            {
                "algorithm": algorithm,
                "size": size,
                "n_samples": stat_dict["n_samples"],
                "score_mean": stat_dict["score_mean"],
                "score_std": stat_dict["score_std"],
                "score_ci_lower": stat_dict["score_ci_lower"],
                "score_ci_upper": stat_dict["score_ci_upper"],
                "runtime_mean": stat_dict["runtime_mean"],
                "runtime_std": stat_dict["runtime_std"],
                "runtime_ci_lower": stat_dict["runtime_ci_lower"],
                "runtime_ci_upper": stat_dict["runtime_ci_upper"],
            }
        )

    stats_filename = filename.replace(".csv", "_statistics.csv")
    fieldnames = [
        "algorithm",
        "size",
        "n_samples",
        "score_mean",
        "score_std",
        "score_ci_lower",
        "score_ci_upper",
        "runtime_mean",
        "runtime_std",
        "runtime_ci_lower",
        "runtime_ci_upper",
    ]

    with open(stats_filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_rows)

    print(f"✓ Statistics saved to {stats_filename}")


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
    """Print a basic summary of optimization results (kept for compatibility)."""
    successful_results = [r for r in results if r["status"] == "SUCCESS"]
    failed_results = [r for r in results if r["status"] == "FAILED"]

    print("\nBASIC SUMMARY:")
    print(f"Total runs: {len(results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")

    if failed_results:
        print("\nFAILED RUNS:")
        for result in failed_results[:5]:  # Show first 5 failures
            print(f"  {result['config_name']:25s} | Error: {result['error']}")
        if len(failed_results) > 5:
            print(f"  ... and {len(failed_results) - 5} more failures")


def main():
    parser = argparse.ArgumentParser(
        description="Comparative DEAP vs PyGMO optimization benchmark - statistical performance analysis"
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

    print("AIrsenal DEAP vs PyGMO Optimization Benchmark")
    print(f"Season: {season}")
    print(f"Gameweeks: {min(gw_range)} to {max(gw_range)} ({len(gw_range)} gameweeks)")
    print(f"Budget: £{budget / 10:.1f}m")
    print(f"Output file: {args.output}")

    # Check what algorithms are available
    available_algorithms = ["DEAP"]  # DEAP is now required

    print(f"Available algorithms: {', '.join(available_algorithms)}")

    if not available_algorithms:
        print("ERROR: DEAP is not available. Install DEAP.")
        sys.exit(1)

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

    # Group by size for summary
    size_summary = {}
    for config in configs_to_run:
        size = config["size_category"]
        alg = config["algorithm"]
        if size not in size_summary:
            size_summary[size] = {"deap": 0, "pygmo": 0}
        size_summary[size][alg] += 1

    for size in sorted(size_summary.keys()):
        deap_count = size_summary[size]["deap"]
        pygmo_count = size_summary[size]["pygmo"]
        total_est = (size * size * (deap_count + pygmo_count)) / 1000  # Rough estimate
        print(
            f"  Size {size}: {deap_count} DEAP + {pygmo_count} PyGMO runs (~{total_est:.0f}min estimated)"
        )

    total_configs = len(configs_to_run)
    total_est_time = (
        sum(c["population_size"] * c["generations"] for c in configs_to_run) / 1000
    )
    print(
        f"\nTotal: {total_configs} runs, estimated {total_est_time:.0f} minutes ({total_est_time / 60:.1f} hours)"
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
    print("\nStarting DEAP vs PyGMO benchmark...")
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
        if i % 5 == 0 or i == len(configs_to_run):  # Save every 5 runs and at the end
            save_results_to_csv(results, args.output)

    print(f"\nTime completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Final save and statistical analysis
    save_results_to_csv(results, args.output)

    # Compute statistics and create plots
    print("\nComputing statistics and creating plots...")
    statistics = compute_statistics(results)

    # Save statistics
    save_statistics_csv(statistics, args.output)

    # Create performance plot
    plot_filename = args.output.replace(".csv", "_performance_plot.png")
    try:
        create_performance_plot(statistics, plot_filename)
    except Exception as e:
        print(f"Warning: Could not create plot: {e}")

    # Print statistical summary
    print_statistical_summary(statistics)

    print(f"\n✓ Benchmark completed! Results saved to {args.output}")


def print_statistical_summary(statistics: Dict[str, Any]):
    """Print a summary of statistical results."""
    print(f"\n{'=' * 70}")
    print("STATISTICAL PERFORMANCE SUMMARY")
    print(f"{'=' * 70}")

    # Group by algorithm
    deap_results = {
        size: stat_data
        for (alg, size), stat_data in statistics.items()
        if alg == "deap"
    }
    pygmo_results = {
        size: stat_data
        for (alg, size), stat_data in statistics.items()
        if alg == "pygmo"
    }

    print(
        f"{'Size':<6} {'Algorithm':<8} {'Score (Mean±CI)':<20} {'Runtime (Mean±CI)':<20} {'N':<3}"
    )
    print("-" * 70)

    sizes = sorted(set(size for (_, size) in statistics.keys()))

    for size in sizes:
        # DEAP results
        if size in deap_results:
            data = deap_results[size]
            score_str = f"{data['score_mean']:.1f}±{(data['score_ci_upper'] - data['score_ci_lower']) / 2:.1f}"
            runtime_str = f"{data['runtime_mean']:.0f}±{(data['runtime_ci_upper'] - data['runtime_ci_lower']) / 2:.0f}s"
            print(
                f"{size:<6} {'DEAP':<8} {score_str:<20} {runtime_str:<20} {data['n_samples']:<3}"
            )

        # PyGMO results
        if size in pygmo_results:
            data = pygmo_results[size]
            score_str = f"{data['score_mean']:.1f}±{(data['score_ci_upper'] - data['score_ci_lower']) / 2:.1f}"
            runtime_str = f"{data['runtime_mean']:.0f}±{(data['runtime_ci_upper'] - data['runtime_ci_lower']) / 2:.0f}s"
            print(
                f"{size:<6} {'PyGMO':<8} {score_str:<20} {runtime_str:<20} {data['n_samples']:<3}"
            )

        if size in deap_results or size in pygmo_results:
            print("-" * 70)


if __name__ == "__main__":
    main()
