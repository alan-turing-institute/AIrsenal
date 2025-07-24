## Optimization Benchmark Results Summary

**Test Configuration:**
- Season: 2526
- Gameweeks: 1-3 (3 gameweeks total)
- Budget: £100.0m
- Date: 2025-07-24

**Configurations Tested:**
1. DEAP_HIGH_QUALITY (Pop: 150, Gen: 200, Crossover: 0.8, Mutation: 0.2)
2. PYGMO_HIGH_QUALITY (Pop: 150, Gen: 200)

**Results:**

| Rank | Algorithm | Configuration | Score | Runtime | GW1 Points |
|------|-----------|---------------|-------|---------|------------|
| 1    | PYGMO     | HIGH_QUALITY  | 182.98| 1.0 min | 62.99      |
| 2    | DEAP      | HIGH_QUALITY  | 180.81| 0.8 min | 61.76      |

**Performance Analysis:**
- **Winner**: PYGMO_HIGH_QUALITY with 182.98 points (+2.17 points advantage, +1.2% better)
- **Speed**: DEAP was slightly faster (0.8 min vs 1.0 min)
- **Both algorithms**: Successfully completed without failures
- **Score Range**: 180.81 - 182.98 points (very competitive results)

**Key Insights:**
1. Both algorithms produced high-quality results in the 180+ point range
2. PYGMO had a slight performance edge for this specific configuration
3. DEAP was marginally faster to execute
4. Both algorithms are viable for squad optimization with similar parameter settings
5. The difference is small enough that other factors (stability, parameter tuning) may be more important

**Output Files:**
- Detailed CSV results: `optimization_benchmark_20250724_155721.csv`
- Contains complete run data including timing, parameters, and squad details

**Success Rate:** 100% (2/2 configurations completed successfully)

This demonstrates that the optimization benchmark script is working correctly and can effectively compare different algorithms and hyperparameters for squad optimization in AIrsenal.
