#!/usr/bin/env python3
"""
Comparison between PyGMO and DEAP implementations for AIrsenal squad optimization.
This script demonstrates the differences and shows how to use both.
"""


def show_pygmo_vs_deap_comparison():
    """Show the key differences between PyGMO and DEAP implementations"""

    print("PyGMO vs DEAP Comparison for AIrsenal Squad Optimization")
    print("=" * 60)

    print("\n1. INSTALLATION:")
    print("-" * 30)
    print("PyGMO:")
    print("  - Requires conda: conda install pygmo")
    print("  - Cannot be installed with pip alone")
    print("  - Complex build dependencies")

    print("\nDEAP:")
    print("  - Simple pip install: pip install deap")
    print("  - Pure Python package")
    print("  - No special build requirements")

    print("\n2. API COMPARISON:")
    print("-" * 30)
    print("PyGMO approach:")
    print("""
    from airsenal.framework.optimization_pygmo import make_new_squad_pygmo
    import pygmo as pg
    
    squad = make_new_squad_pygmo(
        gw_range=[1, 2, 3, 4, 5],
        tag="my_prediction_tag",
        budget=1000,
        uda=pg.sga(gen=100),  # PyGMO-specific algorithm
        population_size=100,
        verbose=1,
    )
    """)

    print("DEAP approach:")
    print("""
    from airsenal.framework.optimization_deap import make_new_squad_deap
    
    squad = make_new_squad_deap(
        gw_range=[1, 2, 3, 4, 5],
        tag="my_prediction_tag",
        budget=1000,
        population_size=100,
        generations=100,      # More explicit control
        crossover_prob=0.7,   # Genetic algorithm parameters
        mutation_prob=0.3,
        verbose=True,
        random_state=42,      # Reproducibility
    )
    """)

    print("\n3. KEY ADVANTAGES:")
    print("-" * 30)
    print("DEAP advantages:")
    print("  ✓ Easier installation (pip only)")
    print("  ✓ Better reproducibility (random_state parameter)")
    print("  ✓ More explicit parameter control")
    print("  ✓ Position-aware genetic operators")
    print("  ✓ Better debugging (pure Python)")
    print("  ✓ Extensive documentation")

    print("\nPyGMO advantages:")
    print("  ✓ More sophisticated algorithms available")
    print("  ✓ Potentially faster convergence")
    print("  ✓ Multi-objective optimization support")

    print("\n4. GENETIC ALGORITHM FEATURES:")
    print("-" * 30)
    print("DEAP implementation includes:")
    print("  • Custom crossover respecting position constraints")
    print("  • Position-aware mutation")
    print("  • Tournament selection")
    print("  • Fitness tracking and statistics")
    print("  • Hall of fame for best solutions")

    print("\n5. PERFORMANCE TUNING:")
    print("-" * 30)
    print("For quick testing:")
    print("  population_size=50, generations=20")
    print("\nFor production use:")
    print("  population_size=100-200, generations=100-200")
    print("\nFor best results:")
    print("  population_size=300+, generations=300+")


def demonstrate_deap_usage():
    """Show practical usage examples"""
    print("\n" + "=" * 60)
    print("PRACTICAL DEAP USAGE EXAMPLES")
    print("=" * 60)

    print("\n1. BASIC USAGE:")
    print("-" * 30)
    print("""
# Basic squad optimization
from airsenal.framework.optimization_deap import make_new_squad_deap

squad = make_new_squad_deap(
    gw_range=[1, 2, 3, 4, 5],
    tag="my_prediction_tag",
    verbose=True
)
    """)

    print("\n2. QUICK TESTING:")
    print("-" * 30)
    print("""
# Fast optimization for testing (smaller population/generations)
squad = make_new_squad_deap(
    gw_range=[1, 2, 3],
    tag="test_tag",
    population_size=30,
    generations=10,
    verbose=True,
    random_state=42
)
    """)

    print("\n3. PRODUCTION OPTIMIZATION:")
    print("-" * 30)
    print("""
# High-quality optimization (larger population/generations)
squad = make_new_squad_deap(
    gw_range=[1, 2, 3, 4, 5, 6, 7, 8],
    tag="prediction_tag",
    population_size=200,
    generations=200,
    crossover_prob=0.8,
    mutation_prob=0.2,
    verbose=True,
    random_state=42
)
    """)

    print("\n4. PARTIAL SQUAD OPTIMIZATION:")
    print("-" * 30)
    print("""
# Optimize only part of the squad (e.g., 12 out of 15 players)
squad = make_new_squad_deap(
    gw_range=[1, 2, 3, 4, 5],
    tag="prediction_tag",
    players_per_position={"GK": 1, "DEF": 4, "MID": 4, "FWD": 3},
    dummy_sub_cost=45,  # Price for dummy players
    verbose=True
)
    """)

    print("\n5. ADVANCED USAGE:")
    print("-" * 30)
    print("""
# Direct access to optimization class for custom control
from airsenal.framework.optimization_deap import SquadOptDEAP

optimizer = SquadOptDEAP(
    gw_range=[1, 2, 3, 4, 5],
    tag="prediction_tag",
    budget=1000
)

# Run optimization with custom parameters
best_individual, best_fitness = optimizer.optimize(
    population_size=150,
    generations=150,
    crossover_prob=0.75,
    mutation_prob=0.25,
    verbose=True,
    random_state=42
)

print(f"Best fitness: {best_fitness}")
print(f"Player indices: {best_individual}")
    """)


def main():
    """Main function to run the comparison"""
    show_pygmo_vs_deap_comparison()
    demonstrate_deap_usage()

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("The DEAP implementation provides:")
    print("✓ Easier installation and setup")
    print("✓ Better reproducibility and debugging")
    print("✓ More explicit control over optimization parameters")
    print("✓ Position-aware genetic operators for fantasy football")
    print("✓ Comprehensive documentation and examples")
    print("\nRecommended for most AIrsenal users!")


if __name__ == "__main__":
    main()
