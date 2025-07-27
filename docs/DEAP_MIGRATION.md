# Optimization with DEAP

This document explains how to use the DEAP-based optimization, which has replaced the previous PyGMO implementation.

## Why DEAP?

The DEAP (Distributed Evolutionary Algorithms in Python) package offers several advantages:

1. **Pure Python installation**: Can be installed with `pip install deap` without requiring conda
2. **Better documentation**: More extensive documentation and examples
3. **Flexible genetic algorithm implementation**: Easy to customize operators and constraints
4. **Active development**: Well-maintained and regularly updated
5. **Better debugging**: Python-native implementation makes debugging easier

## Installation

### With pip:
```bash
pip install deap
```

### With poetry (recommended for this project):
```bash
poetry install --extras optimization
```

## API Usage

The DEAP optimization provides a clean interface for genetic algorithm-based optimization:

### DEAP API:
```python
from airsenal.framework.optimization_deap import make_new_squad_deap

squad = make_new_squad_deap(
    gw_range=[1, 2, 3, 4, 5],
    tag="my_prediction_tag",
    budget=1000,
    population_size=100,
    generations=100,      # Number of generations to evolve
    crossover_prob=0.7,   # Genetic algorithm parameters
    mutation_prob=0.3,
    verbose=True,
    random_state=42,      # Reproducibility
)
```

## Key Improvements

### 1. Better Parameter Control
The DEAP implementation provides more explicit control over genetic algorithm parameters:

- `population_size`: Number of individuals in each generation
- `generations`: Number of generations to evolve
- `crossover_prob`: Probability of crossover between parents
- `mutation_prob`: Probability of mutation for each individual
- `random_state`: Seed for reproducible results

### 2. Position-Aware Genetic Operators
The genetic operators (crossover and mutation) are designed to respect fantasy football position constraints:

- **Crossover**: Combines parent solutions while maintaining the correct number of players per position
- **Mutation**: Replaces players only with others from the same position
- **Initialization**: Creates valid starting solutions that meet all constraints

### 3. Enhanced Error Handling
Better error messages and validation for common issues:

```python
try:
    squad = make_new_squad_deap(gw_range=[1, 2, 3], tag="my_tag")
except ModuleNotFoundError:
    print("Please install DEAP: pip install deap")
except Exception as e:
    print(f"Optimization failed: {e}")
```

### 4. Reproducible Results
Set `random_state` for consistent results across runs:

```python
# This will always produce the same result
squad1 = make_new_squad_deap(gw_range=[1, 2, 3], tag="tag", random_state=42)
squad2 = make_new_squad_deap(gw_range=[1, 2, 3], tag="tag", random_state=42)
# squad1 and squad2 will be identical
```

## Performance Comparison

Both implementations should give similar results, but with different characteristics:

- **PyGMO**: May converge faster due to more sophisticated algorithms
- **DEAP**: More predictable behavior and easier to tune parameters

## Advanced Usage

### Custom Genetic Algorithm Parameters

```python
# Aggressive optimization (higher diversity)
squad = make_new_squad_deap(
    gw_range=[1, 2, 3, 4, 5],
    tag="my_tag",
    population_size=200,    # Larger population
    generations=150,        # More generations
    crossover_prob=0.8,     # Higher crossover rate
    mutation_prob=0.2,      # Lower mutation rate
    verbose=True,
)

# Conservative optimization (more exploitation)
squad = make_new_squad_deap(
    gw_range=[1, 2, 3, 4, 5],
    tag="my_tag",
    population_size=50,     # Smaller population
    generations=200,        # Many generations
    crossover_prob=0.6,     # Lower crossover rate
    mutation_prob=0.4,      # Higher mutation rate
    verbose=True,
)
```

### Direct Access to Optimization Class

For more advanced use cases, you can use the `SquadOptDEAP` class directly:

```python
from airsenal.framework.optimization_deap import SquadOptDEAP

# Create optimizer
optimizer = SquadOptDEAP(
    gw_range=[1, 2, 3, 4, 5],
    tag="my_tag",
    budget=1000,
)

# Run optimization with custom parameters
best_individual, best_fitness = optimizer.optimize(
    population_size=100,
    generations=100,
    crossover_prob=0.7,
    mutation_prob=0.3,
    verbose=True,
    random_state=42,
)

print(f"Best fitness: {best_fitness}")
print(f"Player indices: {best_individual}")
```

## Usage Checklist

- [ ] Install DEAP: `pip install deap` 
- [ ] Use `optimization_deap` module
- [ ] Use `make_new_squad_deap` function  
- [ ] Set parameters: `generations`, `crossover_prob`, `mutation_prob`
- [ ] Add `random_state` for reproducibility
- [ ] Test with small population/generations first
- [ ] Tune parameters for your specific use case

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install DEAP with `pip install deap`
2. **Slow optimization**: Reduce `population_size` and `generations` for testing
3. **Poor results**: Increase `population_size` and `generations`, try different genetic algorithm parameters
4. **Memory issues**: Reduce `population_size`

### Getting Help

If you encounter issues with the DEAP implementation:

1. Check that all required data is available (players, predictions, etc.)
2. Try with smaller parameter values first
3. Set `verbose=True` to see optimization progress
4. Use `random_state` to get reproducible results for debugging
