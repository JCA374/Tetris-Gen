"""
Benchmark script to test training performance before running full training.

This script compares sequential vs parallel training and provides timing analysis.
"""

import time
import sys
from multiprocessing import cpu_count
from tetris_gymnasium.envs import Tetris

from genetic import GeneticAlgorithm


def create_env():
    """Create a Tetris environment."""
    return Tetris()


def run_benchmark(population_size, games_per_genome, n_workers, description):
    """
    Run a benchmark with specified parameters.

    Args:
        population_size: Number of genomes
        games_per_genome: Games to play per genome
        n_workers: Number of parallel workers
        description: Description of this benchmark

    Returns:
        Dictionary with timing results
    """
    print(f"\n{'='*70}")
    print(f"Benchmark: {description}")
    print(f"{'='*70}")
    print(f"Population: {population_size}, Games: {games_per_genome}, Workers: {n_workers}")

    # Initialize GA
    ga = GeneticAlgorithm(
        population_size=population_size,
        n_workers=n_workers
    )
    ga.initialize_population()

    # Time the fitness evaluation
    start_time = time.time()
    ga.evaluate_fitness(create_env, games_per_genome=games_per_genome)
    end_time = time.time()

    elapsed = end_time - start_time
    total_games = population_size * games_per_genome
    games_per_second = total_games / elapsed if elapsed > 0 else 0

    # Get best fitness
    best_genome = ga.get_best_genome()

    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"  Total time: {elapsed:.2f} seconds")
    print(f"  Total games: {total_games}")
    print(f"  Games/second: {games_per_second:.2f}")
    print(f"  Average time per game: {elapsed/total_games:.3f} seconds")
    print(f"  Best fitness: {best_genome.fitness:.2f}")
    print(f"{'='*70}")

    return {
        'description': description,
        'population': population_size,
        'games': games_per_genome,
        'workers': n_workers,
        'elapsed': elapsed,
        'total_games': total_games,
        'games_per_second': games_per_second,
        'best_fitness': best_genome.fitness
    }


def print_comparison(results):
    """
    Print a comparison of benchmark results.

    Args:
        results: List of result dictionaries
    """
    print(f"\n{'='*70}")
    print("BENCHMARK COMPARISON")
    print(f"{'='*70}")
    print(f"{'Description':<30} {'Time (s)':<12} {'Games/s':<12} {'Speedup':<10}")
    print(f"{'-'*70}")

    baseline_time = results[0]['elapsed']

    for r in results:
        speedup = baseline_time / r['elapsed'] if r['elapsed'] > 0 else 0
        print(f"{r['description']:<30} {r['elapsed']:<12.2f} "
              f"{r['games_per_second']:<12.2f} {speedup:<10.2f}x")

    print(f"{'='*70}")


def estimate_training_time(games_per_second, generations, population, games):
    """
    Estimate total training time based on benchmark results.

    Args:
        games_per_second: Games processed per second
        generations: Number of generations to train
        population: Population size
        games: Games per genome

    Returns:
        Estimated time in seconds
    """
    total_games = generations * population * games
    estimated_seconds = total_games / games_per_second if games_per_second > 0 else 0
    return estimated_seconds


def format_time(seconds):
    """Format seconds into human-readable time."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def main():
    print("="*70)
    print("TETRIS GENETIC ALGORITHM - BENCHMARK TEST")
    print("="*70)
    print(f"System CPUs detected: {cpu_count()}")

    # Small test to warm up and validate
    print("\nRunning warm-up test...")
    test_ga = GeneticAlgorithm(population_size=2, n_workers=1)
    test_ga.initialize_population()
    test_ga.evaluate_fitness(create_env, games_per_genome=1)
    print("Warm-up complete!")

    # Benchmark parameters
    benchmark_population = 8
    benchmark_games = 2

    results = []

    # Test 1: Sequential (baseline)
    results.append(run_benchmark(
        population_size=benchmark_population,
        games_per_genome=benchmark_games,
        n_workers=1,
        description="Sequential (1 worker)"
    ))

    # Test 2: Half of CPUs
    half_cpus = max(1, cpu_count() // 2)
    if half_cpus > 1:
        results.append(run_benchmark(
            population_size=benchmark_population,
            games_per_genome=benchmark_games,
            n_workers=half_cpus,
            description=f"Parallel ({half_cpus} workers)"
        ))

    # Test 3: All CPUs
    all_cpus = cpu_count()
    if all_cpus > half_cpus:
        results.append(run_benchmark(
            population_size=benchmark_population,
            games_per_genome=benchmark_games,
            n_workers=all_cpus,
            description=f"Parallel ({all_cpus} workers)"
        ))

    # Print comparison
    print_comparison(results)

    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS FOR FULL TRAINING")
    print(f"{'='*70}")

    # Find best configuration
    best_result = max(results, key=lambda x: x['games_per_second'])
    print(f"\nRecommended configuration: {best_result['workers']} workers")
    print(f"  -> Achieves {best_result['games_per_second']:.2f} games/second")

    # Estimate training times for different scenarios
    print("\nEstimated training times:")
    scenarios = [
        ("Quick test", 10, 20, 3),
        ("Medium run", 30, 30, 3),
        ("Full training", 50, 30, 5),
        ("Extended training", 100, 50, 5)
    ]

    for name, gens, pop, games in scenarios:
        est_time = estimate_training_time(
            best_result['games_per_second'],
            gens, pop, games
        )
        total_games = gens * pop * games
        print(f"  {name:<20} ({gens} gens, pop {pop}, {games} games): "
              f"{format_time(est_time)} ({total_games:,} total games)")

    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print(f"To start training with optimal settings:")
    print(f"  python src/train.py --workers {best_result['workers']} \\")
    print(f"                      --generations 50 \\")
    print(f"                      --population 30 \\")
    print(f"                      --games 3")
    print(f"\nTo start a quick test run:")
    print(f"  python src/train.py --workers {best_result['workers']} \\")
    print(f"                      --generations 10 \\")
    print(f"                      --population 20 \\")
    print(f"                      --games 2")
    print(f"{'='*70}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        sys.exit(0)
