"""
Analyze optimal training configuration for given hardware and time constraints.

This script helps determine the best trade-off between:
- Population size (exploration breadth)
- Number of generations (optimization depth)
- Games per genome (measurement accuracy)
"""

import numpy as np
import argparse


def estimate_training_time(population, generations, games, games_per_second):
    """
    Estimate total training time.

    Args:
        population: Population size
        generations: Number of generations
        games: Games per genome
        games_per_second: Throughput from benchmark

    Returns:
        Estimated time in seconds
    """
    total_games = population * generations * games
    return total_games / games_per_second


def genetic_algorithm_theory():
    """
    Print theoretical guidelines for GA configuration.
    """
    print("=" * 70)
    print("GENETIC ALGORITHM THEORY")
    print("=" * 70)
    print()
    print("Population Size Guidelines:")
    print("  - Too small (<10): High variance, poor exploration, early convergence")
    print("  - Too large (>100): Slow convergence, wasted computation")
    print("  - Sweet spot: 20-50 for 6-dimensional problem")
    print()
    print("Generation Guidelines:")
    print("  - Rule of thumb: 50-200 generations for convergence")
    print("  - Diminishing returns after population stabilizes")
    print("  - More generations = deeper optimization (exploitation)")
    print()
    print("Games per Genome:")
    print("  - More games = reduced variance in fitness measurement")
    print("  - Fewer games = faster iteration, but noisier signal")
    print("  - Recommendation: 3-5 games for balance")
    print()
    print("Trade-off:")
    print("  - Fixed budget: population × generations × games = constant")
    print("  - Early stage: Favor larger population (exploration)")
    print("  - Late stage: Favor more generations (exploitation)")
    print("  - With time constraint: Optimize for convergence rate")
    print()


def analyze_configurations(games_per_second, time_budget_hours=None):
    """
    Analyze different configurations.

    Args:
        games_per_second: Throughput from benchmark
        time_budget_hours: Optional time constraint in hours
    """
    print("=" * 70)
    print("CONFIGURATION ANALYSIS")
    print("=" * 70)
    print(f"Benchmark throughput: {games_per_second:.2f} games/second")
    if time_budget_hours:
        print(f"Time budget: {time_budget_hours} hours")
    print()

    # Define configurations to test
    configs = [
        # (population, generations, games, description)
        (5, 100, 3, "Current (very small pop)"),
        (10, 50, 3, "Small population"),
        (20, 50, 3, "Balanced small"),
        (30, 50, 3, "Recommended baseline"),
        (30, 50, 5, "Recommended + accuracy"),
        (50, 30, 3, "Large population"),
        (20, 100, 3, "Deep optimization"),
        (40, 50, 3, "Large balanced"),
        (30, 100, 3, "Extended training"),
        (50, 50, 3, "Large extended"),
    ]

    print(f"{'Config':<30} {'Pop':>5} {'Gen':>5} {'Games':>6} {'Total':>8} {'Time':>10} {'Eff.':>6}")
    print("-" * 70)

    results = []
    for pop, gen, games, desc in configs:
        total_games = pop * gen * games
        time_sec = estimate_training_time(pop, gen, games, games_per_second)
        time_hours = time_sec / 3600

        # Efficiency metric: population * generations (exploration * exploitation)
        efficiency = pop * gen

        results.append({
            'description': desc,
            'population': pop,
            'generations': gen,
            'games': games,
            'total_games': total_games,
            'time_hours': time_hours,
            'efficiency': efficiency,
            'time_sec': time_sec
        })

        if time_budget_hours and time_hours > time_budget_hours:
            status = "TOO LONG"
        else:
            status = f"{efficiency}"

        print(f"{desc:<30} {pop:>5} {gen:>5} {games:>6} {total_games:>8} "
              f"{time_hours:>8.1f}h {status:>6}")

    return results


def recommend_configuration(games_per_second, time_budget_hours=None, current_gen=0):
    """
    Recommend optimal configuration based on constraints.

    Args:
        games_per_second: Throughput from benchmark
        time_budget_hours: Optional time constraint
        current_gen: If resuming, current generation number
    """
    print()
    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print()

    if current_gen > 0:
        print(f"Note: Currently at generation {current_gen}")
        print()

    print("Based on your hardware (8 cores, 4 workers optimal):")
    print(f"  Throughput: {games_per_second:.2f} games/second")
    print()

    if not time_budget_hours:
        print("Without time constraint:")
        print()
        print("  RECOMMENDED: --population 30 --generations 100 --games 3 --workers 4")
        print("  - Population of 30 gives good genetic diversity")
        print("  - 100 generations allows deep optimization")
        print("  - 3 games balances accuracy vs speed")
        print(f"  - Estimated time: {estimate_training_time(30, 100, 3, games_per_second)/3600:.1f} hours")
        print()
        print("  AGGRESSIVE: --population 50 --generations 100 --games 5 --workers 4")
        print("  - Maximum exploration and accuracy")
        print(f"  - Estimated time: {estimate_training_time(50, 100, 5, games_per_second)/3600:.1f} hours")
        print()
        print("  QUICK TEST: --population 20 --generations 30 --games 3 --workers 4")
        print("  - Fast iteration for testing")
        print(f"  - Estimated time: {estimate_training_time(20, 30, 3, games_per_second)/3600:.1f} hours")
    else:
        available_seconds = time_budget_hours * 3600
        available_games = available_seconds * games_per_second

        print(f"With {time_budget_hours}h budget:")
        print(f"  Available game evaluations: ~{int(available_games)}")
        print()

        # Optimize for this budget
        # Target: population * generations * games ≈ available_games

        if time_budget_hours <= 2:
            pop, gen, games = 20, 30, 3
            desc = "Quick run"
        elif time_budget_hours <= 6:
            pop, gen, games = 30, 50, 3
            desc = "Medium run"
        elif time_budget_hours <= 12:
            pop, gen, games = 30, 100, 3
            desc = "Long run"
        else:
            pop, gen, games = 50, 100, 5
            desc = "Extended run"

        actual_time = estimate_training_time(pop, gen, games, games_per_second) / 3600

        print(f"  OPTIMAL ({desc}):")
        print(f"    --population {pop} --generations {gen} --games {games} --workers 4")
        print(f"    Estimated time: {actual_time:.1f}h")
        print(f"    Total evaluations: {pop * gen * games}")


def analyze_current_training(checkpoint_path=None):
    """
    Analyze current training and suggest improvements.
    """
    print()
    print("=" * 70)
    print("CURRENT CONFIGURATION ANALYSIS")
    print("=" * 70)
    print()

    print("Your current run (population=5, generations=50, games=3):")
    print()
    print("  PROBLEMS IDENTIFIED:")
    print("  ❌ Population of 5 is TOO SMALL")
    print("     - Minimal genetic diversity")
    print("     - High chance of premature convergence")
    print("     - Standard recommendation: 20-50 for 6 variables")
    print()
    print("  ❌ High variance in fitness")
    print("     - Your history shows many negative improvements")
    print("     - Caused by small population + only 3 games")
    print("     - Generation 35 breakthrough was likely lucky discovery")
    print()
    print("  ✓ 50 generations is reasonable")
    print("  ✓ 3 games per genome is acceptable")
    print("  ✓ Using 4 workers is optimal")
    print()
    print("  RECOMMENDATION:")
    print("  Start fresh with population=30 for better results")
    print("  OR: Continue current run but don't expect major improvements")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize genetic algorithm configuration for your hardware"
    )
    parser.add_argument(
        '--games-per-second',
        type=float,
        default=0.34,
        help='Throughput from benchmark (default: 0.34 for 4 workers)'
    )
    parser.add_argument(
        '--time-budget',
        type=float,
        help='Time budget in hours (optional)'
    )
    parser.add_argument(
        '--current-gen',
        type=int,
        default=0,
        help='Current generation if resuming (default: 0)'
    )
    parser.add_argument(
        '--analyze-current',
        action='store_true',
        help='Analyze the current training configuration'
    )

    args = parser.parse_args()

    print()
    genetic_algorithm_theory()
    print()

    analyze_configurations(args.games_per_second, args.time_budget)
    print()

    recommend_configuration(args.games_per_second, args.time_budget, args.current_gen)

    if args.analyze_current:
        analyze_current_training()

    print()
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
