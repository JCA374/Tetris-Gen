"""Check if games are hitting the max_steps limit."""
import pickle
import sys

checkpoint_path = sys.argv[1]

with open(checkpoint_path, 'rb') as f:
    checkpoint = pickle.load(f)

print("Checking for max_steps issue...")
print("="*70)

# Get the latest generation's population
population = checkpoint['population']
print(f"Population size: {len(population)}")
print()

# Check fitness values to estimate game lengths
fitnesses = [fitness for weights, fitness in population]
print(f"Fitness range: {min(fitnesses):.1f} - {max(fitnesses):.1f}")
print(f"Average fitness: {sum(fitnesses)/len(fitnesses):.1f}")
print()

# High fitness values suggest long games
# Fitness = lines*100 + score
# A fitness of ~1400 means roughly 14 lines or equivalent score
# In Tetris, clearing 14 lines is achievable in <1000 steps

print("Analysis:")
if max(fitnesses) > 2000:
    print("  ⚠️  Very high fitness detected (>2000)")
    print("     Games might be hitting max_steps limit (1000)")
    print("     Consider increasing max_steps in agent.py")
else:
    print("  ✓ Fitness values look reasonable")
    print("    Games are likely completing naturally")

print()
print("Note: Current max_steps = 1000 in agent.py:186")
print("      Each step = one action (move/rotate/drop)")
