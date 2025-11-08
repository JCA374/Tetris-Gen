"""
Training script using the improved genetic algorithm with constraints.

This version includes:
- Biased initialization (starts near optimal range)
- Weight sign constraints (prevents bad patterns)
- Improved fitness function
- Pattern monitoring
"""

import sys
import os
import argparse
import pickle
from datetime import datetime
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import numpy as np
from tetris_gymnasium.envs import Tetris

from genetic_improved import ImprovedGeneticAlgorithm
from heuristics import HEURISTIC_NAMES


def create_env():
    """Create a Tetris gymnasium environment."""
    try:
        env = Tetris()
        return env
    except Exception as e:
        print(f"Error creating environment: {e}")
        sys.exit(1)


def plot_training_progress(history, save_path):
    """Plot and save training progress visualizations."""
    generations = [h['generation'] for h in history]
    best_fitness = [h['best_fitness'] for h in history]
    avg_fitness = [h['avg_fitness'] for h in history]
    worst_fitness = [h['worst_fitness'] for h in history]

    plt.figure(figsize=(12, 6))

    # Fitness over generations
    plt.subplot(1, 2, 1)
    plt.plot(generations, best_fitness, 'g-', label='Best', linewidth=2)
    plt.plot(generations, avg_fitness, 'b-', label='Average', linewidth=2)
    plt.plot(generations, worst_fitness, 'r-', label='Worst', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Over Generations')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Weight evolution
    plt.subplot(1, 2, 2)
    for i, name in enumerate(HEURISTIC_NAMES):
        weights = [h['best_weights'][i] for h in history]
        plt.plot(generations, weights, label=name, linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Weight Value')
    plt.title('Best Genome Weights Over Generations')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training plot saved to {save_path}")


def print_genome_weights(genome):
    """Print the weights of a genome in a readable format."""
    print("\nBest Genome Weights:")
    for name, weight in zip(HEURISTIC_NAMES, genome.weights):
        print(f"  {name:20s}: {weight:8.3f}")


def main():
    parser = argparse.ArgumentParser(description='Train Tetris AI using Improved Genetic Algorithm')
    parser.add_argument('--generations', type=int, default=100,
                        help='Number of generations to train (default: 100)')
    parser.add_argument('--population', type=int, default=50,
                        help='Population size (default: 50)')
    parser.add_argument('--games', type=int, default=5,
                        help='Games per genome for fitness evaluation (default: 5)')
    parser.add_argument('--mutation-rate', type=float, default=0.1,
                        help='Mutation rate (default: 0.1)')
    parser.add_argument('--mutation-step', type=float, default=0.2,
                        help='Mutation step size (default: 0.2)')
    parser.add_argument('--no-constraints', action='store_true',
                        help='Disable weight sign constraints')
    parser.add_argument('--old-fitness', action='store_true',
                        help='Use old fitness function instead of improved')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results (default: results)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output_dir, f'run_improved_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 60)
    print("Tetris Improved Genetic Algorithm Training")
    print("=" * 60)
    print(f"Generations: {args.generations}")
    print(f"Population: {args.population}")
    print(f"Games per genome: {args.games}")
    print(f"Mutation rate: {args.mutation_rate}")
    print(f"Mutation step: {args.mutation_step}")
    print(f"Weight constraints: {'Disabled' if args.no_constraints else 'Enabled'}")
    print(f"Fitness function: {'Original' if args.old_fitness else 'Improved'}")
    print(f"Output directory: {run_dir}")
    print("=" * 60)

    # Initialize improved genetic algorithm
    ga = ImprovedGeneticAlgorithm(
        population_size=args.population,
        mutation_rate=args.mutation_rate,
        mutation_step=args.mutation_step,
        elite_size=2,
        tournament_size=3,
        use_constraints=not args.no_constraints,
        use_improved_fitness=not args.old_fitness
    )

    ga.initialize_population()

    # Test environment creation
    print("\nTesting environment...")
    test_env = create_env()
    print(f"Environment created successfully: {test_env}")
    test_env.close()

    # Training loop
    print("\nStarting training...\n")

    try:
        gen_start_times = []

        for gen in range(args.generations):
            gen_start_time = datetime.now()

            print(f"\n{'='*60}")
            print(f"Generation {gen + 1}/{args.generations}")
            if gen_start_times:
                avg_time = sum((t[1] - t[0]).total_seconds() for t in gen_start_times) / len(gen_start_times)
                remaining_gens = args.generations - (gen + 1)
                eta_seconds = avg_time * remaining_gens
                eta_hours = int(eta_seconds // 3600)
                eta_mins = int((eta_seconds % 3600) // 60)
                print(f"Avg time/gen: {int(avg_time/60)}m {int(avg_time%60)}s | "
                      f"ETA: {eta_hours}h {eta_mins}m")
            print(f"{'='*60}")

            # Evaluate fitness
            print("Evaluating fitness...")
            ga.evaluate_fitness(create_env, games_per_genome=args.games)

            # Evolve population
            ga.evolve_population()

            gen_end_time = datetime.now()
            gen_start_times.append((gen_start_time, gen_end_time))
            elapsed = (gen_end_time - gen_start_time).total_seconds()
            print(f"Generation {gen + 1} completed in {int(elapsed/60)}m {int(elapsed%60)}s")

            # Save checkpoint every 10 generations
            if (gen + 1) % 10 == 0:
                checkpoint_path = os.path.join(run_dir, f'checkpoint_gen_{gen+1}.pkl')
                print(f"Saving checkpoint...")
                # Note: Would need to add save_checkpoint to ImprovedGeneticAlgorithm
                # For now, just save the best genome
                best_genome_path = os.path.join(run_dir, f'best_genome_gen_{gen+1}.pkl')
                with open(best_genome_path, 'wb') as f:
                    pickle.dump({
                        'weights': ga.best_genome.weights,
                        'fitness': ga.best_genome.fitness,
                        'generation': gen + 1
                    }, f)

            # Plot progress every 10 generations
            if (gen + 1) % 10 == 0:
                plot_path = os.path.join(run_dir, f'progress_gen_{gen+1}.png')
                plot_training_progress(ga.history, plot_path)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")

    # Final results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    best_genome = ga.get_best_genome()
    print(f"\nBest fitness achieved: {best_genome.fitness:.2f}")
    print_genome_weights(best_genome)

    # Save final results
    best_genome_path = os.path.join(run_dir, 'best_genome.pkl')
    with open(best_genome_path, 'wb') as f:
        pickle.dump({
            'weights': best_genome.weights,
            'fitness': best_genome.fitness,
            'generation': ga.generation
        }, f)
    print(f"Best genome saved to {best_genome_path}")

    # Final plot
    final_plot = os.path.join(run_dir, 'final_progress.png')
    plot_training_progress(ga.history, final_plot)

    print(f"\nAll results saved to: {run_dir}")

    # Compare to optimal weights
    print("\n" + "=" * 60)
    print("COMPARISON TO OPTIMAL WEIGHTS")
    print("=" * 60)

    optimal = {
        'aggregate_height': -0.510066,
        'complete_lines': 0.760666,
        'holes': -0.35663,
        'bumpiness': -0.184483,
        'max_height': -0.5,
        'wells': -0.2
    }

    print(f"{'Heuristic':<20} {'Evolved':<12} {'Optimal':<12} {'Diff':<10}")
    print("-" * 60)
    for name, weight in zip(HEURISTIC_NAMES, best_genome.weights):
        opt_val = optimal.get(name, 0)
        diff = abs(weight - opt_val)
        print(f"{name:<20} {weight:>+.3f}       {opt_val:>+.3f}       {diff:.3f}")


if __name__ == '__main__':
    main()
