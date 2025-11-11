"""
Analyze population size vs generation count trade-off for fixed training time.

Given that generation time scales with population AND with agent skill,
find the optimal population size to maximize total evolution.
"""

def analyze_tradeoff(total_hours=24, workers=4):
    """
    Calculate optimal population size for a fixed training budget.

    Args:
        total_hours: Total training time available
        workers: Number of parallel workers
    """
    print(f"\n{'='*70}")
    print(f"POPULATION SIZE TRADE-OFF ANALYSIS")
    print(f"{'='*70}")
    print(f"Training budget: {total_hours} hours")
    print(f"Workers: {workers}")
    print(f"\n")

    # Candidate populations (divisible by 4 for 100% efficiency)
    populations = [16, 20, 24, 28, 32, 36, 40]

    # Estimated time per generation in LATE stages (minutes)
    # Based on observed data: pop 28 -> 57 min, pop 30 -> 120 min
    # Rough scaling: time scales ~linearly with population
    time_per_gen_base = 57  # minutes for pop 28 in late stage
    base_pop = 28

    print(f"{'Pop':<6} {'Eff':<6} {'Early':<8} {'Late':<8} {'Avg':<8} {'Gens':<6} {'Score':<8}")
    print(f"{'Size':<6} {'%':<6} {'min/gen':<8} {'min/gen':<8} {'min/gen':<8} {'(24h)':<6} {'(Pop×Gen)':<8}")
    print(f"{'-'*70}")

    results = []

    for pop in populations:
        # Worker efficiency
        efficiency = 100.0 if pop % workers == 0 else (pop // workers * workers) / pop * 100

        # Estimated time per generation (scales with population)
        late_time = time_per_gen_base * (pop / base_pop)
        early_time = late_time * 0.25  # Early gens are ~4x faster
        avg_time = (early_time + late_time) / 2  # Rough average over training

        # How many generations fit in the budget?
        total_minutes = total_hours * 60
        generations = int(total_minutes / avg_time)

        # Evolution "score" = population × generations (genetic diversity × evolution time)
        score = pop * generations

        results.append({
            'pop': pop,
            'efficiency': efficiency,
            'early_time': early_time,
            'late_time': late_time,
            'avg_time': avg_time,
            'generations': generations,
            'score': score
        })

        print(f"{pop:<6} {efficiency:>5.1f}% {early_time:>7.1f} {late_time:>7.1f} {avg_time:>7.1f} {generations:>6} {score:>8}")

    # Find optimal
    best = max(results, key=lambda x: x['score'])

    print(f"{'-'*70}")
    print(f"\nOPTIMAL CONFIGURATION:")
    print(f"  Population: {best['pop']}")
    print(f"  Estimated generations in {total_hours}h: {best['generations']}")
    print(f"  Evolution score: {best['score']:,}")
    print(f"\n  Rationale: Maximizes (population × generations) for genetic")
    print(f"             diversity AND evolutionary pressure.")

    # Compare with current recommendation
    current = [r for r in results if r['pop'] == 28][0]
    if best['pop'] != 28:
        improvement = ((best['score'] - current['score']) / current['score']) * 100
        print(f"\n  vs Pop 28: {improvement:+.1f}% more total evolution")

    print(f"\n{'='*70}")

    # Show different time budgets
    print(f"\nRECOMMENDATIONS FOR DIFFERENT TIME BUDGETS:")
    print(f"{'-'*70}")

    for hours in [12, 24, 48, 72]:
        best_for_time = None
        best_score = 0

        for pop in populations:
            if pop % workers != 0:
                continue  # Only consider fully efficient populations

            late_time = time_per_gen_base * (pop / base_pop)
            avg_time = (late_time * 0.25 + late_time) / 2
            gens = int((hours * 60) / avg_time)
            score = pop * gens

            if score > best_score:
                best_score = score
                best_for_time = pop

        gens = int((hours * 60) / (time_per_gen_base * (best_for_time / base_pop) * 0.625))
        print(f"  {hours:2d}h: Population {best_for_time}, ~{gens:3d} generations")

    print(f"{'='*70}\n")

if __name__ == "__main__":
    analyze_tradeoff(total_hours=24, workers=4)
