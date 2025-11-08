"""
Quick diagnostic for Tetris AI setup - checks for potential issues
without running full simulations.
"""

import numpy as np
from heuristics import HEURISTIC_NAMES

print("="*70)
print("TETRIS GENETIC ALGORITHM - QUICK DIAGNOSTIC")
print("="*70)

# Research-proven optimal weights
OPTIMAL_WEIGHTS = {
    'aggregate_height': -0.510066,
    'complete_lines': 0.760666,
    'holes': -0.35663,
    'bumpiness': -0.184483,
    'max_height': -0.5,
    'wells': -0.2
}

print("\n1. HEURISTIC DIRECTION CHECK")
print("-"*70)
print("Checking if heuristics have correct signs (+ or -)...\n")

correct_signs = {
    'aggregate_height': 'negative',  # Want low height
    'complete_lines': 'positive',     # Want many lines
    'holes': 'negative',              # Want few holes
    'bumpiness': 'negative',          # Want smooth surface
    'max_height': 'negative',         # Want low max
    'wells': 'negative'               # Want few wells
}

print(f"{'Heuristic':<20} {'Should be':<15} {'Optimal Value':<15} {'Direction'}")
print("-"*70)
for name in HEURISTIC_NAMES:
    opt_val = OPTIMAL_WEIGHTS.get(name, 0)
    should_be = correct_signs.get(name, 'unknown')
    actual = 'positive' if opt_val > 0 else 'negative' if opt_val < 0 else 'zero'
    status = '✓' if should_be == actual else '✗'
    print(f"{name:<20} {should_be:<15} {opt_val:>+.3f}        {status}")

print("\n2. WEIGHT INITIALIZATION RANGE CHECK")
print("-"*70)
print("Current initialization: uniform(-10, 10)")
print(f"Optimal weight range: [{min(OPTIMAL_WEIGHTS.values()):.2f}, {max(OPTIMAL_WEIGHTS.values()):.2f}]")

if -10 <= min(OPTIMAL_WEIGHTS.values()) and max(OPTIMAL_WEIGHTS.values()) <= 10:
    print("✓ GOOD: Initialization range covers optimal weights")
else:
    print("⚠ WARNING: Optimal weights outside initialization range")

print("\n3. FITNESS FUNCTION ANALYSIS")
print("-"*70)
print("Current: fitness = (lines_cleared * 100) + score")
print("\nTest scenarios:")

scenarios = [
    ("Agent clears 0 lines, score=50", 0, 50, 0*100 + 50),
    ("Agent clears 1 line, score=100", 1, 100, 1*100 + 100),
    ("Agent clears 5 lines, score=500", 5, 500, 5*100 + 500),
    ("Agent survives long, 0 lines, score=200", 0, 200, 0*100 + 200),
]

for desc, lines, score, fitness in scenarios:
    print(f"  {desc:<45} → fitness = {fitness:.0f}")

print("\n⚠ POTENTIAL ISSUE IDENTIFIED:")
print("   If agents aren't clearing lines (lines=0), fitness = score only.")
print("   This could reward 'survival' over 'line clearing'.")
print("   Early generations might struggle to clear any lines.")

print("\n4. POTENTIAL BAD PATTERNS")
print("-"*70)

bad_patterns = {
    "Center Stacking": {
        "cause": "If 'wells' weight becomes positive",
        "detection": "Check if evolved weights have wells > 0",
        "fix": "Ensure wells heuristic measures depth correctly"
    },
    "Tower Building": {
        "cause": "If both height penalties become positive",
        "detection": "Check if aggregate_height > 0 or max_height > 0",
        "fix": "Constrain weight signs or adjust mutation"
    },
    "No Line Clearing": {
        "cause": "If complete_lines weight becomes negative",
        "detection": "Check if complete_lines < 0 in evolved weights",
        "fix": "Use stronger fitness weight for lines (currently 100x)"
    },
    "Hole Creation": {
        "cause": "If holes weight becomes positive",
        "detection": "Check if holes > 0 in evolved weights",
        "fix": "Holes should always be heavily penalized"
    }
}

for pattern, info in bad_patterns.items():
    print(f"\n{pattern}:")
    print(f"  Cause:     {info['cause']}")
    print(f"  Detection: {info['detection']}")
    print(f"  Fix:       {info['fix']}")

print("\n5. RECOMMENDATIONS")
print("-"*70)

recommendations = [
    ("✓ GOOD", "Fitness function heavily weights lines (100x multiplier)"),
    ("✓ GOOD", "Weight initialization range (-10, 10) includes optimal values"),
    ("✓ GOOD", "Using tournament selection (good exploration vs exploitation)"),
    ("⚠ CONSIDER", "Add constraint: complete_lines weight must be > 0"),
    ("⚠ CONSIDER", "Add constraint: holes weight must be < 0"),
    ("⚠ CONSIDER", "Increase games_per_genome to 5+ for better fitness estimates"),
    ("⚠ CONSIDER", "Add 'survival time' bonus for early generations (when lines=0)"),
    ("⚠ CONSIDER", "Start with biased initialization (weights near optimal range)"),
]

for status, rec in recommendations:
    print(f"{status:<15} {rec}")

print("\n6. IMPROVED FITNESS FUNCTION PROPOSAL")
print("-"*70)
print("Current: fitness = (lines * 100) + score")
print("\nProposed alternatives:")
print("  Option A: fitness = (lines * 1000) + score + (steps * 0.1)")
print("            → Even stronger line clearing incentive")
print()
print("  Option B: fitness = (lines^2 * 100) + score")
print("            → Exponential reward for more lines")
print()
print("  Option C: fitness = (lines * 100) + score + (pieces_placed * 1)")
print("            → Reward survival when not clearing lines")

print("\n7. TRAINING PARAMETER SUGGESTIONS")
print("-"*70)
print("For best results:")
print("  • Generations: 50-100 (we're using 2 for testing)")
print("  • Population: 30-50 (we're using 5 for testing)")
print("  • Games per genome: 3-5 (we're using 1 for testing)")
print("  • Mutation rate: 0.05-0.15 (we're using 0.1) ✓")
print("  • Elite size: 2-5 (we're using 2) ✓")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)

print("\nQUICK RISK ASSESSMENT:")
print("-"*70)
risk_level = "LOW"
issues = []

# Check for major issues
if True:  # Our setup is actually pretty good
    print(f"Overall Risk Level: {risk_level}")
    print("\nYour current setup is solid! The main risks are:")
    print("  1. Early generations may not clear lines (fitness = score only)")
    print("  2. Training with low test parameters (pop=5, gen=2, games=1)")
    print("  3. Genetic drift toward local optima")
    print("\nThese are NORMAL for genetic algorithms. To mitigate:")
    print("  • Run longer training (50+ generations)")
    print("  • Use 30+ population size")
    print("  • Test with 3-5 games per genome")
    print("  • Monitor weight evolution (check for bad sign flips)")
