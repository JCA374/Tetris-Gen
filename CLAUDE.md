# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tetris-Gen is a genetic algorithm-based Tetris AI that evolves agents through heuristic weight optimization. The system uses a population-based evolutionary approach to discover optimal playing strategies.

## Core Architecture

### Genetic Algorithm Pipeline

The training pipeline follows a standard genetic algorithm flow:

1. **Population Initialization** (`genetic.py`): Creates a population of Genomes, each representing a unique set of heuristic weights
2. **Fitness Evaluation** (`agent.py`): Each genome plays multiple Tetris games to determine fitness (based on lines cleared and score)
3. **Evolution** (`genetic.py`): Generates next generation through:
   - Tournament selection
   - Uniform crossover between parents
   - Mutation with ±20% weight adjustments
   - Elitism (preserves top performers)

### Agent Decision-Making

The agent (`agent.py`) uses a **planning-based reactive strategy**:
- When a new piece spawns, evaluates all possible placements (up to 4 rotations × 11 horizontal positions)
- Simulates each action sequence using `get_state()`/`set_state()` for rollback
- Scores resulting boards using weighted heuristics
- Executes the action sequence that yields the highest score

### Heuristic System

Six board evaluation metrics in `heuristics.py`:
- **complete_lines**: Rows ready to clear (reward)
- **aggregate_height**: Sum of column heights (penalty)
- **holes**: Empty cells with blocks above (penalty)
- **bumpiness**: Surface irregularity (penalty)
- **max_height**: Tallest column (penalty)
- **wells**: Deep valleys in surface (penalty)

The genetic algorithm learns optimal weights for combining these heuristics.

## Development Commands

### System Optimization (Run First on New Hardware)

Before training, benchmark your system to find optimal settings:

```bash
# Run comprehensive benchmark test
python src/benchmark.py
```

This will:
- Test sequential vs parallel performance
- Compare different worker counts
- Estimate training times for various scenarios
- Recommend optimal worker configuration

After benchmarking, analyze worker efficiency:

```bash
# Analyze population alignment for optimal workers
python optimization/analyze_worker_efficiency.py --workers <OPTIMAL_WORKERS>

# Get configuration recommendations
python optimization/optimize_config.py --games-per-second <THROUGHPUT>
```

**CRITICAL**: Population size should be divisible by worker count for 100% efficiency. See `optimization/README.md` for details.

### Training

**Recommended command for current system (DESKTOP-APPKIAQ, 8 cores, 4 workers optimal):**

```bash
python src/train.py --population 28 --generations 100 --games 3 --workers 4
```

**Other training options:**

```bash
# Quick test run
python src/train.py --population 20 --generations 50 --games 3 --workers 4

# Resume from checkpoint
python src/train.py --resume results/run_YYYYMMDD_HHMMSS/checkpoint_gen_25.pkl

# Custom parameters
python src/train.py --generations 100 --population 32 --games 5 --workers 4

# Adjust genetic parameters
python src/train.py --mutation-rate 0.15 --mutation-step 0.3

# Low RAM system (disable parallelization)
python src/train.py --workers 1
```

**Key training parameters:**
- `--population`: Number of genomes (MUST be divisible by workers for efficiency)
- `--generations`: Number of evolution cycles
- `--games`: Games per genome for fitness averaging (3-5 recommended)
- `--workers`: Parallel worker count (0 = auto-detect, recommended: benchmark first)

### Monitoring Training Progress

```bash
# Analyze progress from a checkpoint
python src/analyze_progress.py results/run_YYYYMMDD_HHMMSS/checkpoint_gen_N.pkl
```

This will show:
- Current generation and population size
- Best fitness achieved across all generations
- Generation-by-generation fitness history with improvements
- Progress statistics

### Evaluation

```bash
# Watch a trained agent play with visualization
python src/play.py --genome results/run_YYYYMMDD_HHMMSS/best_genome.pkl

# Play multiple games for statistics
python src/play.py --genome results/run_YYYYMMDD_HHMMSS/best_genome.pkl --games 10

# Run without rendering (faster evaluation)
python src/play.py --genome results/run_YYYYMMDD_HHMMSS/best_genome.pkl --no-render

# Adjust visualization speed
python src/play.py --genome results/run_YYYYMMDD_HHMMSS/best_genome.pkl --delay 0.05
```

### Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Or install in a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## File Structure

```
src/
  ├── genetic.py              - GeneticAlgorithm and Genome classes
  ├── agent.py                - TetrisAgent (planning and execution)
  ├── heuristics.py           - Board evaluation functions
  ├── train.py                - Training loop with checkpointing
  ├── play.py                 - Visualization and evaluation script
  ├── analyze_progress.py     - Checkpoint analysis tool
  └── benchmark.py            - System performance benchmarking

optimization/
  ├── README.md                            - Comprehensive optimization guide
  ├── QUICK_START.md                       - Quick reference for optimal configs
  ├── system_config_DESKTOP-APPKIAQ_8CORE.json  - Current system profile
  ├── analyze_worker_efficiency.py         - Worker alignment analysis
  └── optimize_config.py                   - Configuration optimizer

results/                      - Training outputs (gitignored)
  └── run_YYYYMMDD_HHMMSS/
      ├── best_genome.pkl               - Best performing weights
      ├── final_checkpoint.pkl          - Complete algorithm state
      ├── checkpoint_gen_N.pkl          - Periodic checkpoints (every 5 gens)
      └── progress_gen_N.png            - Training plots
```

## Key Implementation Details

### Parallel Training and Worker Efficiency

**CRITICAL OPTIMIZATION**: Population size must be divisible by worker count to avoid idle workers.

Example with 4 workers:
- Population=28: 100% efficient (28 ÷ 4 = 7 batches, no remainder)
- Population=30: 93.8% efficient (30 ÷ 4 = 7.5 batches, 2 idle worker-slots per generation)

The efficiency loss compounds across all generations. On a 100-generation run, misaligned populations can waste 6-7% of total training time.

**Recommended populations for 4 workers**: 20, 24, 28, 32, 36, 40, 48
**Avoid for 4 workers**: 5, 10, 15, 25, 30, 35, 50

Worker configuration:
- `--workers 0` (default): Auto-detect and use all CPU cores (may not be optimal)
- `--workers N`: Use N parallel workers
- `--workers 1`: Disable parallelization (sequential evaluation, lowest RAM usage)

Always run `python src/benchmark.py` first to determine optimal worker count for your system.

### State Management

The agent relies on `env.unwrapped.get_state()` and `env.unwrapped.set_state()` for action simulation. This allows evaluating hypothetical moves without affecting the actual game state. This is critical for the planning algorithm which tests ~44 placements per piece.

### Fitness Function

```
Fitness = (avg_lines_cleared × 100) + avg_score
```

This heavily prioritizes line clearing over raw score accumulation. The 100× multiplier ensures that clearing even one line is worth more than 100 points of score.

### Checkpoint Format

Pickled checkpoints contain:
- `generation`: Current generation number
- `population`: List of (weights, fitness) tuples
- `best_genome`: (weights, fitness) of all-time best
- `history`: Per-generation statistics for plotting
- `config`: Hyperparameters (mutation_rate, mutation_step, elite_size, etc.)

Checkpoints are saved every 5 generations and at completion. Use `--resume` to continue from any checkpoint.

### Training Outputs

Results are saved to timestamped directories: `results/run_YYYYMMDD_HHMMSS/`
- `best_genome.pkl`: Best performing weights
- `final_checkpoint.pkl`: Complete algorithm state at end
- `checkpoint_gen_N.pkl`: Periodic checkpoints (every 5 generations)
- `progress_gen_N.png`: Training plots (fitness curves and weight evolution)

## Common Gotchas

- The agent assumes `tetris-gymnasium` environment supports `get_state()`/`set_state()` methods
- Fitness evaluation is stochastic due to random piece sequences; use `--games 3` or higher to average multiple runs
- Training generates large checkpoint files; `results/` directory is gitignored
- The agent evaluates moves greedily (one piece at a time); it does not plan multiple pieces ahead
- **Training slows down significantly** as genomes improve and play longer games. Expect 2-3x slowdown from early to late generations. The benchmark uses random genomes that fail quickly, so real training takes much longer than benchmark estimates.
- **Small populations (<20) fail**: Insufficient genetic diversity leads to premature convergence and high fitness variance. Minimum recommended: 20 for quick tests, 28-32 for production.
- **Population misalignment wastes compute**: If population is not divisible by worker count, workers sit idle. This compounds across all generations.

## System-Specific Configurations

See `optimization/system_config_*.json` for hardware-specific recommendations. When moving to a new system:

1. Run `python src/benchmark.py`
2. Create new config file in `optimization/` with hostname
3. Update population recommendations based on optimal worker count
4. Note RAM constraints (use fewer workers if <4GB available)

Current system: DESKTOP-APPKIAQ (8 cores, 4 workers optimal, 0.34 games/sec)
