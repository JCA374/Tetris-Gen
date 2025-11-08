"""Quick script to analyze training progress."""
import pickle
import sys

checkpoint_path = sys.argv[1]

with open(checkpoint_path, 'rb') as f:
    checkpoint = pickle.load(f)

print("="*70)
print("TRAINING PROGRESS ANALYSIS")
print("="*70)
print(f"Current Generation: {checkpoint['generation']}")
print(f"Population Size: {checkpoint['config']['population_size']}")
print()

print("Best Genome Ever:")
if checkpoint['best_genome']:
    weights, fitness = checkpoint['best_genome']
    print(f"  Fitness: {fitness:.2f}")
else:
    print("  Not yet available")

print()
print("Generation History:")
print("-"*70)
print(f"{'Gen':<6} {'Best':<12} {'Average':<12} {'Worst':<12} {'Improvement':<12}")
print("-"*70)

prev_best = None
for i, gen in enumerate(checkpoint['history']):
    improvement = ""
    if prev_best is not None:
        diff = gen['best_fitness'] - prev_best
        if diff > 0:
            improvement = f"+{diff:.1f}"
        elif diff < 0:
            improvement = f"{diff:.1f}"
        else:
            improvement = "="

    print(f"{gen['generation']:<6} {gen['best_fitness']:<12.1f} "
          f"{gen['avg_fitness']:<12.1f} {gen['worst_fitness']:<12.1f} "
          f"{improvement:<12}")
    prev_best = gen['best_fitness']

print("="*70)

# Calculate time estimates
if len(checkpoint['history']) >= 2:
    print("\nProgress Stats:")
    best_fitnesses = [h['best_fitness'] for h in checkpoint['history']]
    print(f"  Best fitness seen: {max(best_fitnesses):.1f}")
    print(f"  Current best: {checkpoint['history'][-1]['best_fitness']:.1f}")
    print(f"  Generations completed: {checkpoint['generation']}")
