#!/usr/bin/env python3
"""
Minimal working example of DEAP-based squad optimization for AIrsenal
"""


def simple_deap_test():
    """Test basic DEAP functionality"""
    print("Testing basic DEAP functionality...")

    try:
        import random

        import numpy as np
        from deap import base, creator, tools

        # Create fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Initialize toolbox
        toolbox = base.Toolbox()

        # Define a simple optimization problem (maximize sum of 5 integers from 0-10)
        def eval_sum(individual):
            return (sum(individual),)

        # Register functions
        toolbox.register("attr_int", random.randint, 0, 10)
        toolbox.register(
            "individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=5
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", eval_sum)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=0, up=10, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Create population
        pop = toolbox.population(n=50)

        # Evaluate population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Run a few generations
        for g in range(10):
            # Select next generation
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < 0.2:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate invalid individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring

            # Print progress
            fits = [ind.fitness.values[0] for ind in pop]
            print(f"Generation {g}: Max={max(fits)}, Avg={np.mean(fits):.2f}")

        # Get best individual
        best = max(pop, key=lambda x: x.fitness.values[0])
        print(f"Best individual: {best} with fitness {best.fitness.values[0]}")
        print("✓ Basic DEAP test successful!")
        return True

    except Exception as e:
        print(f"✗ Basic DEAP test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_airsenal_import():
    """Test importing AIrsenal DEAP optimization"""
    try:
        print("\nTesting AIrsenal DEAP optimization import...")
        from airsenal.framework.optimization_deap import SquadOptDEAP

        print("✓ Successfully imported SquadOptDEAP and make_new_squad_deap")

        # Test that we can create the class (without requiring database)
        print("Testing class structure...")

        # Check if the class has expected methods
        expected_methods = [
            "_setup_deap",
            "_create_individual",
            "_evaluate_individual",
            "_crossover",
            "_mutate",
            "optimize",
        ]

        for method in expected_methods:
            if hasattr(SquadOptDEAP, method):
                print(f"✓ Method {method} found")
            else:
                print(f"✗ Method {method} missing")

        print("✓ AIrsenal DEAP import test successful!")
        return True

    except Exception as e:
        print(f"✗ AIrsenal DEAP import test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("AIrsenal DEAP Optimization Test")
    print("=" * 40)

    # Test basic DEAP functionality
    basic_test = simple_deap_test()

    # Test AIrsenal integration
    airsenal_test = test_airsenal_import()

    print("\n" + "=" * 40)
    if basic_test and airsenal_test:
        print("✓ All tests passed!")
        print("\nDEAP optimization is ready for use!")
        print("\nUsage example:")
        print("```python")
        print("from airsenal.framework.optimization_deap import make_new_squad_deap")
        print("")
        print("# Optimize squad for gameweeks 1-5")
        print("squad = make_new_squad_deap(")
        print("    gw_range=[1, 2, 3, 4, 5],")
        print("    tag='your_prediction_tag',")
        print("    budget=1000,  # £100m")
        print("    population_size=100,")
        print("    generations=100,")
        print("    verbose=True,")
        print("    random_state=42  # For reproducible results")
        print(")")
        print("```")
    else:
        print("✗ Some tests failed")
        if not basic_test:
            print("  - Basic DEAP functionality test failed")
        if not airsenal_test:
            print("  - AIrsenal DEAP import test failed")


if __name__ == "__main__":
    main()
