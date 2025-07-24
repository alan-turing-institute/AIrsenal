#!/usr/bin/env python3
"""
Example script showing how to use the new DEAP-based optimization
compared to the original PyGMO implementation.
"""

from airsenal.framework.optimization_deap import make_new_squad_deap
from airsenal.framework.utils import CURRENT_SEASON


def main():
    # Example usage of the DEAP optimization
    print("Running DEAP-based squad optimization...")

    # Define gameweeks to optimize for (e.g., first 5 gameweeks)
    gw_range = list(range(1, 6))

    # Tag for prediction model (you'll need to adjust this based on your setup)
    tag = "default"

    try:
        # Run optimization with DEAP
        optimized_squad = make_new_squad_deap(
            gw_range=gw_range,
            tag=tag,
            budget=1000,  # £100m budget (in units of 0.1m)
            season=CURRENT_SEASON,
            population_size=50,  # Smaller for quick testing
            generations=20,  # Fewer generations for quick testing
            verbose=True,
            random_state=42,  # For reproducible results
        )

        print("\nOptimization completed successfully!")
        print(f"Squad budget remaining: £{optimized_squad.budget / 10:.1f}m")

    except Exception as e:
        print(f"Error during optimization: {e}")
        print("\nMake sure to install DEAP first:")
        print("pip install deap")
        print("or with poetry:")
        print("poetry install --extras optimization")


if __name__ == "__main__":
    main()
