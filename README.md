# Tetris-Gen: Genetic Algorithm Tetris AI

A Tetris AI that learns to play using genetic algorithms. The system evolves a population of agents, each with different heuristic weights, to find optimal playing strategies.

## How It Works

The genetic algorithm optimizes a set of weights for board evaluation heuristics:

- **Rows Cleared**: Reward for completing lines
- **Aggregate Height**: Penalty for tall stacks
- **Holes**: Penalty for empty cells with blocks above
- **Bumpiness**: Penalty for uneven surface
- **Max Height**: Penalty for tallest column
- **Wells**: Penalty for deep valleys

Each agent evaluates all possible placements for each piece and selects the move with the best weighted score.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Train a new population:

```bash
python src/train.py
```

Watch the best agent play:

```bash
python src/play.py --genome models/best_genome.pkl
```

## Project Structure

```
src/
  ├── heuristics.py    # Board evaluation functions
  ├── agent.py         # Tetris playing agent
  ├── genetic.py       # Genetic algorithm implementation
  ├── train.py         # Training script
  └── play.py          # Play/visualize trained agent
```

## Algorithm Parameters

- **Population Size**: 30 genomes
- **Generations**: 50+
- **Selection**: Top 50% survival + elitism
- **Crossover**: Uniform crossover between parents
- **Mutation Rate**: 10% per gene
- **Mutation Step**: ±20% adjustment

## Results

Results and visualizations are saved to the `results/` directory after training.
