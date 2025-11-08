"""
Training script with comprehensive logging for monitoring and sharing.

Creates detailed logs that can be easily shared for progress review.
"""

import sys
import os
import argparse
import pickle
import logging
from datetime import datetime
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import numpy as np
from tetris_gymnasium.envs import Tetris

from genetic_improved import ImprovedGeneticAlgorithm
from heuristics import HEURISTIC_NAMES


def setup_logging(run_dir):
    """Setup comprehensive logging to both file and console."""
    log_file = os.path.join(run_dir, 'training.log')

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler (detailed)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Console handler (less detailed)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(detailed_formatter)

    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file


def create_env():
    """Create a Tetris gymnasium environment."""
    try:
        env = Tetris()
        return env
    except Exception as e:
        logging.error(f"Error creating environment: {e}")
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
    logging.info(f"Plot saved: {save_path}")


def save_progress_summary(run_dir, ga, generation):
    """Save a human-readable progress summary."""
    summary_file = os.path.join(run_dir, f'summary_gen_{generation}.txt')

    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"TRAINING PROGRESS SUMMARY - Generation {generation}\n")
        f.write("="*70 + "\n\n")

        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Generations: {generation}\n")
        f.write(f"Population Size: {ga.population_size}\n\n")

        f.write("BEST GENOME:\n")
        f.write("-"*70 + "\n")
        best = ga.get_best_genome()
        f.write(f"Fitness: {best.fitness:.2f}\n\n")

        f.write("Weights:\n")
        for name, weight in zip(HEURISTIC_NAMES, best.weights):
            f.write(f"  {name:20s}: {weight:+.4f}\n")

        f.write("\n\nFITNESS PROGRESSION:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Gen':>5} {'Best':>12} {'Average':>12} {'Worst':>12} {'Improvement':>12}\n")

        prev_best = 0
        for h in ga.history[-20:]:  # Last 20 generations
            improvement = h['best_fitness'] - prev_best
            f.write(f"{h['generation']:5d} {h['best_fitness']:12.2f} "
                   f"{h['avg_fitness']:12.2f} {h['worst_fitness']:12.2f} "
                   f"{improvement:+12.2f}\n")
            prev_best = h['best_fitness']

        f.write("\n\nCOMPARISON TO OPTIMAL WEIGHTS:\n")
        f.write("-"*70 + "\n")
        optimal = {
            'aggregate_height': -0.510066,
            'complete_lines': 0.760666,
            'holes': -0.35663,
            'bumpiness': -0.184483,
            'max_height': -0.5,
            'wells': -0.2
        }

        f.write(f"{'Heuristic':20s} {'Evolved':>12} {'Optimal':>12} {'Diff':>12}\n")
        for name, weight in zip(HEURISTIC_NAMES, best.weights):
            opt_val = optimal.get(name, 0)
            diff = abs(weight - opt_val)
            f.write(f"{name:20s} {weight:>+12.4f} {opt_val:>+12.4f} {diff:>12.4f}\n")

    logging.info(f"Summary saved: {summary_file}")
    return summary_file


def main():
    parser = argparse.ArgumentParser(description='Train Tetris AI with Comprehensive Logging')
    parser.add_argument('--generations', type=int, default=100,
                        help='Number of generations (default: 100)')
    parser.add_argument('--population', type=int, default=50,
                        help='Population size (default: 50)')
    parser.add_argument('--games', type=int, default=5,
                        help='Games per genome (default: 5)')
    parser.add_argument('--mutation-rate', type=float, default=0.1,
                        help='Mutation rate (default: 0.1)')
    parser.add_argument('--mutation-step', type=float, default=0.2,
                        help='Mutation step (default: 0.2)')
    parser.add_argument('--no-constraints', action='store_true',
                        help='Disable weight constraints')
    parser.add_argument('--old-fitness', action='store_true',
                        help='Use old fitness function')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory (default: results)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output_dir, f'run_improved_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    # Setup logging
    logger, log_file = setup_logging(run_dir)

    logging.info("="*70)
    logging.info("TETRIS GENETIC ALGORITHM TRAINING - COMPREHENSIVE LOGGING")
    logging.info("="*70)
    logging.info(f"Run directory: {run_dir}")
    logging.info(f"Log file: {log_file}")
    logging.info("")
    logging.info("CONFIGURATION:")
    logging.info(f"  Generations: {args.generations}")
    logging.info(f"  Population: {args.population}")
    logging.info(f"  Games per genome: {args.games}")
    logging.info(f"  Mutation rate: {args.mutation_rate}")
    logging.info(f"  Mutation step: {args.mutation_step}")
    logging.info(f"  Weight constraints: {'Disabled' if args.no_constraints else 'Enabled'}")
    logging.info(f"  Fitness function: {'Original' if args.old_fitness else 'Improved'}")
    logging.info("="*70)

    # Initialize GA
    ga = ImprovedGeneticAlgorithm(
        population_size=args.population,
        mutation_rate=args.mutation_rate,
        mutation_step=args.mutation_step,
        elite_size=2,
        tournament_size=3,
        use_constraints=not args.no_constraints,
        use_improved_fitness=not args.old_fitness
    )

    logging.info("\nInitializing population...")
    ga.initialize_population()

    # Test environment
    logging.info("\nTesting environment creation...")
    test_env = create_env()
    logging.info(f"Environment created successfully")
    test_env.close()

    # Training loop
    logging.info("\n" + "="*70)
    logging.info("STARTING TRAINING")
    logging.info("="*70)

    training_start = datetime.now()
    gen_times = []

    try:
        for gen in range(args.generations):
            gen_start = datetime.now()

            logging.info("\n" + "="*70)
            logging.info(f"GENERATION {gen + 1}/{args.generations}")

            # ETA calculation
            if gen_times:
                avg_time = sum(gen_times) / len(gen_times)
                remaining = args.generations - (gen + 1)
                eta_seconds = avg_time * remaining
                eta_str = f"{int(eta_seconds//3600)}h {int((eta_seconds%3600)//60)}m"
                elapsed_total = (gen_start - training_start).total_seconds()
                elapsed_str = f"{int(elapsed_total//3600)}h {int((elapsed_total%3600)//60)}m"
                logging.info(f"Elapsed: {elapsed_str} | Avg/gen: {int(avg_time/60)}m {int(avg_time%60)}s | ETA: {eta_str}")

            logging.info("="*70)

            # Evaluate
            logging.info("Evaluating fitness...")
            ga.evaluate_fitness(create_env, games_per_genome=args.games)

            # Evolve
            logging.info("Evolving population...")
            ga.evolve_population()

            # Track time
            gen_end = datetime.now()
            gen_duration = (gen_end - gen_start).total_seconds()
            gen_times.append(gen_duration)

            logging.info(f"Generation completed in {int(gen_duration/60)}m {int(gen_duration%60)}s")

            # Save checkpoints
            if (gen + 1) % 10 == 0:
                logging.info("Saving checkpoint...")
                best_genome_path = os.path.join(run_dir, f'best_genome_gen_{gen+1}.pkl')
                with open(best_genome_path, 'wb') as f:
                    pickle.dump({
                        'weights': ga.best_genome.weights,
                        'fitness': ga.best_genome.fitness,
                        'generation': gen + 1
                    }, f)
                logging.info(f"Checkpoint saved: {best_genome_path}")

                # Save summary
                summary_file = save_progress_summary(run_dir, ga, gen + 1)

                # Plot
                plot_path = os.path.join(run_dir, f'progress_gen_{gen+1}.png')
                plot_training_progress(ga.history, plot_path)

    except KeyboardInterrupt:
        logging.warning("\n\nTraining interrupted by user!")
    except Exception as e:
        logging.error(f"\n\nTraining failed with error: {e}", exc_info=True)
        raise

    # Final results
    logging.info("\n" + "="*70)
    logging.info("TRAINING COMPLETE")
    logging.info("="*70)

    total_time = (datetime.now() - training_start).total_seconds()
    logging.info(f"Total training time: {int(total_time//3600)}h {int((total_time%3600)//60)}m")

    best_genome = ga.get_best_genome()
    logging.info(f"\nBest fitness achieved: {best_genome.fitness:.2f}")
    logging.info("\nBest weights:")
    for name, weight in zip(HEURISTIC_NAMES, best_genome.weights):
        logging.info(f"  {name:20s}: {weight:+.4f}")

    # Save final
    best_genome_path = os.path.join(run_dir, 'best_genome_final.pkl')
    with open(best_genome_path, 'wb') as f:
        pickle.dump({
            'weights': best_genome.weights,
            'fitness': best_genome.fitness,
            'generation': ga.generation,
            'total_time': total_time
        }, f)
    logging.info(f"\nFinal genome saved: {best_genome_path}")

    # Final summary and plot
    final_summary = save_progress_summary(run_dir, ga, ga.generation)
    final_plot = os.path.join(run_dir, 'final_progress.png')
    plot_training_progress(ga.history, final_plot)

    logging.info(f"\n" + "="*70)
    logging.info(f"All results saved to: {run_dir}")
    logging.info(f"Share these files for review:")
    logging.info(f"  1. {log_file}")
    logging.info(f"  2. {final_summary}")
    logging.info(f"  3. {final_plot}")
    logging.info("="*70)


if __name__ == '__main__':
    main()
