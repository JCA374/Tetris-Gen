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

### Benchmarking

Before starting a long training run, benchmark your system to find optimal settings:

```bash
# Run comprehensive benchmark test
python src/benchmark.py
```

This will:
- Test sequential vs parallel performance
- Compare different worker counts
- Estimate training times for various scenarios
- Recommend optimal worker configuration

### Training

```bash
# Basic training (50 generations, population of 30, auto-detects CPU cores)
python src/train.py

# Use specific number of parallel workers
python src/train.py --workers 8

# Disable parallel processing (single-threaded)
python src/train.py --workers 1

# Custom parameters
python src/train.py --generations 100 --population 50 --games 5

# Resume from checkpoint
python src/train.py --resume results/run_YYYYMMDD_HHMMSS/checkpoint_gen_25.pkl

# Adjust genetic parameters
python src/train.py --mutation-rate 0.15 --mutation-step 0.3
```

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
  ├── genetic.py       - GeneticAlgorithm and Genome classes
  ├── agent.py         - TetrisAgent (planning and execution)
  ├── heuristics.py    - Board evaluation functions
  ├── train.py         - Training loop with checkpointing
  └── play.py          - Visualization and evaluation script
```

## Key Implementation Details

### Parallel Training

By default, training uses all available CPU cores to evaluate genomes in parallel. Each worker independently plays games for a genome and returns fitness scores. The `--workers` flag controls parallelization:
- `--workers 0` (default): Auto-detect and use all CPU cores
- `--workers N`: Use N parallel workers
- `--workers 1`: Disable parallelization (sequential evaluation)

### State Management

The agent relies on `env.unwrapped.get_state()` and `env.unwrapped.set_state()` for action simulation. This allows evaluating hypothetical moves without affecting the actual game state.

### Fitness Function

Fitness = (avg_lines_cleared × 100) + avg_score

This heavily prioritizes line clearing over raw score accumulation.

### Checkpoint Format

Pickled checkpoints contain:
- `generation`: Current generation number
- `population`: List of (weights, fitness) tuples
- `best_genome`: (weights, fitness) of all-time best
- `history`: Per-generation statistics for plotting
- `config`: Hyperparameters (mutation_rate, etc.)

### Training Outputs

Results are saved to timestamped directories: `results/run_YYYYMMDD_HHMMSS/`
- `best_genome.pkl`: Best performing weights
- `final_checkpoint.pkl`: Complete algorithm state
- `checkpoint_gen_N.pkl`: Periodic checkpoints (every 5 generations)
- `progress_gen_N.png`: Training plots (fitness curves and weight evolution)

## Common Gotchas

- The agent assumes `tetris-gymnasium` environment supports `get_state()`/`set_state()` methods
- Fitness evaluation is stochastic due to random piece sequences; use `--games` to average multiple runs
- Training generates large checkpoint files; results/ directory is gitignored
- The agent evaluates moves greedily (one piece at a time); it does not plan multiple pieces ahead
- **Training slows down significantly** as genomes improve and play longer games. Expect 2-3x slowdown from early to late generations. The benchmark uses random genomes that fail quickly, so real training takes much longer.
