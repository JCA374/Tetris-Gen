# Tetris-Gen Optimization Analysis

This folder contains optimization tools and configuration recommendations for training the Tetris genetic algorithm efficiently on specific hardware.

## System Configuration Files

- `system_config_DESKTOP_8CORE.json` - Configuration for Jonas's 8-core desktop (current machine)
- Future configs for other machines should follow the naming pattern: `system_config_<HOSTNAME>_<CORES>CORE.json`

## Optimization Scripts

### 1. `analyze_worker_efficiency.py`

Analyzes how efficiently parallel workers are utilized based on population size alignment.

**Purpose**: Multiprocessing workers sit idle when `population % workers != 0`. This script identifies optimal population sizes that achieve 100% worker utilization.

**Usage:**
```bash
# Analyze efficiency for different population sizes
python optimization/analyze_worker_efficiency.py --workers 4

# Visualize batching for a specific population
python optimization/analyze_worker_efficiency.py --workers 4 --visualize 30

# Custom range
python optimization/analyze_worker_efficiency.py --workers 4 --min-pop 10 --max-pop 60
```

**Key Insight**: For 4 workers, use populations that are multiples of 4 (e.g., 8, 12, 16, 20, 24, 28, 32, 40, 48) to achieve 100% efficiency.

### 2. `optimize_config.py`

Provides comprehensive configuration recommendations based on hardware throughput and time constraints.

**Purpose**: Helps choose optimal balance between population size, generations, and games per genome.

**Usage:**
```bash
# General recommendations
python optimization/optimize_config.py --games-per-second 0.34

# With time budget
python optimization/optimize_config.py --games-per-second 0.34 --time-budget 8

# Analyze current training
python optimization/optimize_config.py --games-per-second 0.34 --current-gen 46 --analyze-current
```

## Optimization Theory

### The Three-Way Trade-off

Training time = `population × generations × games_per_genome / throughput`

With fixed time budget, you must balance:

1. **Population Size** (exploration breadth)
   - Searches wider solution space
   - Maintains genetic diversity
   - Reduces premature convergence
   - Optimal: 20-50 for 6-dimensional problems

2. **Generations** (exploitation depth)
   - Refines existing solutions
   - Allows convergence to local optima
   - Diminishing returns after stabilization
   - Optimal: 50-200 depending on convergence

3. **Games per Genome** (measurement accuracy)
   - Reduces fitness variance
   - More reliable selection
   - Slower but more accurate
   - Optimal: 3-5 games

### Worker Alignment Efficiency

**Critical Optimization**: Population size should be divisible by worker count.

Example with 4 workers, population=30:
```
Batch 1: [Worker1][Worker2][Worker3][Worker4]  (4 genomes)
Batch 2: [Worker1][Worker2][Worker3][Worker4]  (4 genomes)
...
Batch 7: [Worker1][Worker2][Worker3][Worker4]  (4 genomes)
Batch 8: [Worker1][Worker2][IDLE   ][IDLE   ]  (2 genomes) ← 50% waste!
```

**Efficiency**: 30 genomes / (8 batches × 4 workers) = 30/32 = 93.8%

With population=32 (aligned):
```
All batches: [Worker1][Worker2][Worker3][Worker4]  (4 genomes each)
```

**Efficiency**: 32 genomes / (8 batches × 4 workers) = 32/32 = 100%

**Impact**: On a 100-generation run, the wasted time compounds every generation. Population=30 wastes ~6-7% total runtime compared to population=32.

## Recommended Configurations

### For 8-Core System (4 Workers Optimal)

Based on benchmark results: **0.34 games/second** throughput with 4 workers.

#### Quick Test Run (~2.5 hours)
```bash
python src/train.py --population 20 --generations 50 --games 3 --workers 4
```
- **Population**: 20 (divisible by 4, minimum recommended for GA)
- **Efficiency**: 100%
- **Total evaluations**: 3,000 games
- **Use case**: Quick iteration, testing algorithm changes

#### Recommended Standard Run (~7.3 hours)
```bash
python src/train.py --population 28 --generations 100 --games 3 --workers 4
```
- **Population**: 28 (divisible by 4, excellent diversity)
- **Efficiency**: 100%
- **Total evaluations**: 8,400 games
- **Use case**: Production training, best balance of quality and time

#### Extended Quality Run (~7.8 hours)
```bash
python src/train.py --population 32 --generations 100 --games 3 --workers 4
```
- **Population**: 32 (divisible by 4, high diversity)
- **Efficiency**: 100%
- **Total evaluations**: 9,600 games
- **Use case**: When you want slightly more diversity than 28

#### Maximum Quality Run (~16.3 hours, overnight)
```bash
python src/train.py --population 40 --generations 100 --games 5 --workers 4
```
- **Population**: 40 (divisible by 4, maximum recommended diversity)
- **Games**: 5 (maximum accuracy)
- **Efficiency**: 100%
- **Total evaluations**: 20,000 games
- **Use case**: Final production model, overnight training

### Configurations to AVOID

❌ **Population=5** (current broken config)
- Only 62.5% worker efficiency
- Insufficient genetic diversity for 6D problem
- High fitness variance
- Will converge prematurely

❌ **Population=30**
- 93.8% efficiency (6.2% waste)
- Common mistake (seems like "nice round number")
- Use 28 or 32 instead

❌ **Population=50**
- 96.2% efficiency (3.8% waste)
- Use 48 instead for perfect alignment

## RAM Considerations

### Current System (Sufficient RAM)

Benchmark shows stable performance with 4 workers evaluating genomes in parallel.

### Low RAM Systems

**Problem**: Each worker process duplicates:
- Python interpreter
- Gymnasium environment
- Game state
- Agent code

**Estimated RAM per worker**: ~200-500 MB

**If RAM is limited (<4 GB available):**

1. **Reduce worker count**:
   ```bash
   # Use 2 workers instead of 4
   python src/train.py --population 20 --generations 100 --games 3 --workers 2
   ```
   - Aligns with populations: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24...
   - Recommended: 20, 24, 28, 32, 40

2. **Use sequential evaluation**:
   ```bash
   # No parallelization
   python src/train.py --population 28 --generations 100 --games 3 --workers 1
   ```
   - Zero RAM overhead from multiprocessing
   - Slower but guaranteed to work
   - Good for systems with <2 GB available RAM

3. **Reduce population and increase generations**:
   ```bash
   # Smaller memory footprint, deeper optimization
   python src/train.py --population 16 --generations 150 --games 3 --workers 2
   ```

### RAM Usage Formula

Approximate peak RAM usage:
```
RAM = Base (1 GB) + (Workers × Worker_Size) + (Population × Genome_Size)

Where:
- Base: Python + OS overhead
- Worker_Size: ~300 MB per worker
- Genome_Size: Negligible (~10 KB)

Examples:
- 4 workers, pop 28: ~1 GB + 1.2 GB = 2.2 GB
- 2 workers, pop 20: ~1 GB + 0.6 GB = 1.6 GB
- 1 worker,  pop 28: ~1 GB + 0.3 GB = 1.3 GB
```

## How to Optimize for a New System

### Step 1: Run Benchmark
```bash
python src/benchmark.py
```

This will:
- Test sequential (1 worker) performance
- Test parallel (4 workers) performance
- Test parallel (8 workers or max cores) performance
- Output optimal worker count
- Measure games/second throughput

### Step 2: Create System Config
Copy `system_config_DESKTOP_8CORE.json` and update:
- hostname
- cpu_cores
- recommended_workers (from benchmark)
- games_per_second (from benchmark)
- ram_gb (available RAM)

### Step 3: Analyze Worker Efficiency
```bash
python optimization/analyze_worker_efficiency.py --workers <OPTIMAL_WORKERS>
```

Note which population sizes achieve 100% efficiency.

### Step 4: Choose Configuration
```bash
python optimization/optimize_config.py --games-per-second <THROUGHPUT>
```

Select from recommended configurations based on time budget.

### Step 5: Document Results
Update this README with new system configuration and recommendations.

## Benchmark Results Archive

### DESKTOP_8CORE (Current System)
- **Date**: 2025-11-09
- **CPU**: 8 cores
- **RAM**: Sufficient (>4 GB available)
- **Benchmark Results**:
  - Sequential (1 worker): 0.05 games/second (baseline)
  - Parallel (4 workers): 0.34 games/second (6.96× speedup) ← **OPTIMAL**
  - Parallel (8 workers): 0.19 games/second (3.83× speedup)
- **Optimal Configuration**: 4 workers
- **Recommended Population Sizes**: 20, 24, 28, 32, 36, 40

**Why 4 workers beats 8 workers:**
- Multiprocessing overhead (process creation, IPC)
- Context switching on 8 cores
- Small population sizes mean workers idle more
- Tetris games are I/O bound (rendering state)

## Troubleshooting

### Training is Slower Than Expected

1. **Check actual worker usage**: Use Task Manager (Windows) or `htop` (Linux) to verify all workers are active
2. **Verify population alignment**: Population should be divisible by worker count
3. **Consider game duration**: Later generations have better genomes that survive longer, naturally slowing down training

### Out of Memory Errors

1. **Reduce worker count**: Try `--workers 2` or `--workers 1`
2. **Reduce population**: Smaller populations use less RAM
3. **Close other applications**: Free up system RAM
4. **Check for memory leaks**: Restart training if RAM usage grows over time

### Inconsistent Performance

1. **Background processes**: Close unnecessary applications
2. **Thermal throttling**: Ensure adequate cooling for long runs
3. **Power settings**: Use "High Performance" mode on laptops

## Future Improvements

### Potential Optimizations

1. **Adaptive population sizing**: Start large (exploration), shrink later (exploitation)
2. **Island model**: Multiple subpopulations with periodic migration
3. **Fitness caching**: Skip re-evaluation if genome unchanged
4. **GPU acceleration**: Use GPU for game simulation (requires gym rewrite)

### Other Hardware Configurations

When running on different systems, create new config files:
- High-core servers (16+ cores): May benefit from 8+ workers
- Low-RAM systems: Document optimal worker counts
- Cloud instances: Cost vs performance analysis

## References

- Genetic Algorithm theory: Goldberg, "Genetic Algorithms in Search, Optimization, and Machine Learning"
- Multiprocessing efficiency: Python multiprocessing docs
- Tetris AI: See CLAUDE.md for project-specific details
