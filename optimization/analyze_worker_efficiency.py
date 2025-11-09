"""
Analyze worker efficiency based on population size alignment.

With parallel evaluation, workers idle when population % workers != 0.
This script calculates optimal population sizes for given worker count.
"""

import argparse


def calculate_efficiency(population, workers):
    """
    Calculate worker efficiency for given population and worker count.

    Args:
        population: Number of genomes
        workers: Number of parallel workers

    Returns:
        Dictionary with efficiency metrics
    """
    # Calculate batches needed
    full_batches = population // workers
    remainder = population % workers
    total_batches = full_batches + (1 if remainder > 0 else 0)

    # Calculate worker utilization
    total_worker_slots = total_batches * workers
    used_worker_slots = population
    wasted_worker_slots = total_worker_slots - used_worker_slots

    efficiency = (used_worker_slots / total_worker_slots) * 100

    return {
        'population': population,
        'workers': workers,
        'full_batches': full_batches,
        'remainder': remainder,
        'total_batches': total_batches,
        'wasted_slots': wasted_worker_slots,
        'efficiency': efficiency
    }


def analyze_population_range(workers, min_pop=10, max_pop=60):
    """
    Analyze efficiency for a range of population sizes.
    """
    print("=" * 80)
    print(f"WORKER EFFICIENCY ANALYSIS ({workers} workers)")
    print("=" * 80)
    print()
    print(f"{'Pop':>5} {'Batches':>8} {'Remainder':>10} {'Wasted':>8} {'Efficiency':>12} {'Rec.':>5}")
    print("-" * 80)

    results = []
    for pop in range(min_pop, max_pop + 1):
        metrics = calculate_efficiency(pop, workers)
        results.append(metrics)

        # Determine if this is a recommended size (multiple of workers)
        is_optimal = (pop % workers == 0)
        rec_mark = "✓" if is_optimal else ""

        print(f"{pop:>5} {metrics['total_batches']:>8} {metrics['remainder']:>10} "
              f"{metrics['wasted_slots']:>8} {metrics['efficiency']:>11.1f}% {rec_mark:>5}")

    return results


def recommend_populations(workers, target_populations=None):
    """
    Recommend optimal population sizes for given worker count.
    """
    print()
    print("=" * 80)
    print("OPTIMAL POPULATION RECOMMENDATIONS")
    print("=" * 80)
    print()

    if target_populations is None:
        # Standard GA population ranges for 6D problem
        target_populations = [20, 30, 40, 50]

    print(f"For {workers} workers, optimal populations are multiples of {workers}:")
    print()

    recommended = []
    for target in target_populations:
        # Find nearest multiples
        lower = (target // workers) * workers
        upper = lower + workers

        # Calculate efficiency difference
        lower_metrics = calculate_efficiency(lower, workers) if lower >= 10 else None
        target_metrics = calculate_efficiency(target, workers)
        upper_metrics = calculate_efficiency(upper, workers)

        print(f"Target population ~{target}:")
        if lower_metrics and lower >= 10:
            print(f"  {lower:>3} (aligned)    - 100% efficient ✓")
            recommended.append(lower)
        print(f"  {target:>3} (requested)  - {target_metrics['efficiency']:.1f}% efficient, "
              f"wastes {target_metrics['wasted_slots']} worker-slots")
        print(f"  {upper:>3} (aligned)    - 100% efficient ✓")
        recommended.append(upper)
        print()

    return recommended


def compare_configurations(workers, games_per_second):
    """
    Compare training time for aligned vs unaligned populations.
    """
    print("=" * 80)
    print("CONFIGURATION COMPARISON")
    print("=" * 80)
    print()

    configs = [
        # (population, generations, games, description)
        (5, 50, 3, "Current (BAD alignment)"),
        (8, 50, 3, "Small aligned"),
        (20, 50, 3, "Good aligned"),
        (30, 50, 3, "Unaligned (common mistake)"),
        (32, 50, 3, "Large aligned"),
        (28, 75, 3, "Medium aligned + deep"),
        (40, 50, 3, "Very large aligned"),
    ]

    print(f"{'Description':<35} {'Pop':>5} {'Gen':>5} {'Eff.':>6} {'Time':>8} {'Waste':>7}")
    print("-" * 80)

    for pop, gen, games, desc in configs:
        metrics = calculate_efficiency(pop, workers)
        total_games = pop * gen * games
        time_hours = (total_games / games_per_second) / 3600

        # Calculate wasted time
        waste_percent = 100 - metrics['efficiency']

        print(f"{desc:<35} {pop:>5} {gen:>5} {metrics['efficiency']:>5.1f}% "
              f"{time_hours:>7.1f}h {waste_percent:>6.1f}%")


def visualize_batching(population, workers):
    """
    Visualize how work is distributed across batches.
    """
    print()
    print("=" * 80)
    print(f"BATCH VISUALIZATION: {population} genomes on {workers} workers")
    print("=" * 80)
    print()

    metrics = calculate_efficiency(population, workers)

    print("Each '█' represents one genome being evaluated")
    print()

    remaining = population
    batch_num = 1

    while remaining > 0:
        genomes_in_batch = min(remaining, workers)

        # Create visualization
        filled = '█' * genomes_in_batch
        empty = '░' * (workers - genomes_in_batch)

        print(f"Batch {batch_num}: [{filled}{empty}]  ({genomes_in_batch}/{workers} workers busy)")

        remaining -= genomes_in_batch
        batch_num += 1

    print()
    print(f"Total batches: {metrics['total_batches']}")
    print(f"Wasted worker-slots: {metrics['wasted_slots']}")
    print(f"Overall efficiency: {metrics['efficiency']:.1f}%")

    if metrics['wasted_slots'] > 0:
        print()
        print(f"⚠️  Suggestion: Use population of {(metrics['total_batches']) * workers} instead")
        print(f"   This would achieve 100% worker utilization")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze worker efficiency for different population sizes"
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    parser.add_argument(
        '--games-per-second',
        type=float,
        default=0.34,
        help='Throughput from benchmark (default: 0.34)'
    )
    parser.add_argument(
        '--visualize',
        type=int,
        help='Visualize batching for specific population size'
    )
    parser.add_argument(
        '--min-pop',
        type=int,
        default=10,
        help='Minimum population to analyze (default: 10)'
    )
    parser.add_argument(
        '--max-pop',
        type=int,
        default=50,
        help='Maximum population to analyze (default: 50)'
    )

    args = parser.parse_args()

    print()

    if args.visualize:
        visualize_batching(args.visualize, args.workers)
    else:
        analyze_population_range(args.workers, args.min_pop, args.max_pop)
        recommend_populations(args.workers)
        compare_configurations(args.workers, args.games_per_second)

    print()


if __name__ == '__main__':
    main()
