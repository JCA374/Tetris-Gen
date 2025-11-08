"""
Diagnostic script to detect bad patterns in Tetris AI training.

This script tests:
1. Whether random weights lead to center stacking or other bad patterns
2. Comparison with research-proven optimal weights
3. Fitness function analysis
4. Visualization of gameplay patterns
"""

import numpy as np
from tetris_gymnasium.envs import Tetris
from agent import TetrisAgent
from heuristics import HEURISTIC_NAMES, get_column_heights
import matplotlib.pyplot as plt

# Optimal weights from research (El-Tetris paper)
OPTIMAL_WEIGHTS = {
    'aggregate_height': -0.510066,
    'complete_lines': 0.760666,
    'holes': -0.35663,
    'bumpiness': -0.184483,
    'max_height': -0.5,  # Similar to aggregate height
    'wells': -0.2  # Should be negative (penalize wells)
}

# Bad pattern detector: center-stacking agent
CENTER_STACKING_WEIGHTS = {
    'aggregate_height': 0.5,  # WRONG: Encourages height
    'complete_lines': 0.1,    # Too low priority
    'holes': 0.0,             # Doesn't care about holes!
    'bumpiness': 0.5,         # WRONG: Encourages bumpiness
    'max_height': 0.0,        # Doesn't care
    'wells': 1.0              # WRONG: Encourages wells (center stacking)
}


def analyze_board_pattern(board):
    """
    Analyze the board for bad patterns.

    Returns dict with pattern metrics.
    """
    heights = get_column_heights(board)
    rows, cols = board.shape

    # Check for center stacking
    middle_cols = heights[cols//3:2*cols//3]
    edge_cols = heights[:cols//3] + heights[2*cols//3:]
    center_bias = np.mean(middle_cols) - np.mean(edge_cols) if len(edge_cols) > 0 else 0

    # Check for extreme height variance
    height_variance = np.var(heights)

    # Check for wall building (one side much higher)
    left_avg = np.mean(heights[:cols//2])
    right_avg = np.mean(heights[cols//2:])
    wall_bias = abs(left_avg - right_avg)

    return {
        'center_bias': center_bias,
        'height_variance': height_variance,
        'wall_bias': wall_bias,
        'max_height': max(heights),
        'avg_height': np.mean(heights),
        'heights': heights
    }


def test_agent_pattern(agent, name, num_games=5):
    """
    Test an agent and analyze its playing patterns.
    """
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    total_lines = 0
    total_score = 0
    total_steps = 0
    pattern_stats = {
        'center_bias': [],
        'height_variance': [],
        'wall_bias': [],
        'max_height': []
    }

    for game in range(num_games):
        env = Tetris()
        result = agent.play_game(env, render=False, max_steps=500)

        # Analyze final board pattern
        pattern = analyze_board_pattern(env.unwrapped.board)

        pattern_stats['center_bias'].append(pattern['center_bias'])
        pattern_stats['height_variance'].append(pattern['height_variance'])
        pattern_stats['wall_bias'].append(pattern['wall_bias'])
        pattern_stats['max_height'].append(pattern['max_height'])

        total_lines += result['lines']
        total_score += result['score']
        total_steps += result['steps']

        env.close()

    # Print results
    avg_lines = total_lines / num_games
    avg_score = total_score / num_games
    avg_steps = total_steps / num_games

    print(f"\nPerformance:")
    print(f"  Avg Lines Cleared: {avg_lines:.2f}")
    print(f"  Avg Score: {avg_score:.2f}")
    print(f"  Avg Steps: {avg_steps:.2f}")

    print(f"\nPattern Analysis:")
    print(f"  Center Bias: {np.mean(pattern_stats['center_bias']):.2f} "
          f"(positive = center stacking)")
    print(f"  Height Variance: {np.mean(pattern_stats['height_variance']):.2f} "
          f"(high = uneven)")
    print(f"  Wall Bias: {np.mean(pattern_stats['wall_bias']):.2f} "
          f"(high = one-sided)")
    print(f"  Avg Max Height: {np.mean(pattern_stats['max_height']):.2f}")

    return {
        'lines': avg_lines,
        'score': avg_score,
        'steps': avg_steps,
        'patterns': pattern_stats
    }


def visualize_gameplay(agent, name):
    """
    Visualize one game to see the pattern visually.
    """
    env = Tetris()
    obs, _ = env.reset()

    board_states = []

    # Play for limited steps
    for step in range(100):
        action = agent.get_best_move(env)
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 10 == 0:  # Sample every 10 steps
            board_states.append(env.unwrapped.board.copy())

        if terminated or truncated:
            break

    env.close()

    # Plot board progression
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f'{name} - Board Progression', fontsize=14)

    for idx, ax in enumerate(axes.flat):
        if idx < len(board_states):
            board = board_states[idx]
            # Show playable area only (remove padding)
            playable = board[4:-4, 4:-4]  # Remove padding
            ax.imshow(playable, cmap='viridis', aspect='auto')
            ax.set_title(f'Step {idx*10}')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    filename = f'diagnosis_{name.replace(" ", "_").lower()}.png'
    plt.savefig(filename, dpi=100)
    print(f"  Visualization saved: {filename}")
    plt.close()


print("="*60)
print("TETRIS AI PATTERN DIAGNOSTIC")
print("="*60)

# Test 1: Optimal weights from research
print("\n1. Testing OPTIMAL weights from research...")
optimal_agent = TetrisAgent(OPTIMAL_WEIGHTS)
optimal_results = test_agent_pattern(optimal_agent, "Optimal (Research-Based)", num_games=5)
visualize_gameplay(optimal_agent, "Optimal")

# Test 2: Random weights (what genetic algorithm starts with)
print("\n2. Testing RANDOM weights (GA starting point)...")
random_weights = np.random.uniform(-10, 10, 6)
random_agent = TetrisAgent(random_weights)
print(f"Random weights: {random_weights}")
random_results = test_agent_pattern(random_agent, "Random Weights", num_games=5)
visualize_gameplay(random_agent, "Random")

# Test 3: Bad pattern weights (center stacking)
print("\n3. Testing BAD weights (center stacking pattern)...")
bad_agent = TetrisAgent(CENTER_STACKING_WEIGHTS)
bad_results = test_agent_pattern(bad_agent, "Bad (Center Stacking)", num_games=5)
visualize_gameplay(bad_agent, "Center_Stacking")

# Test 4: Zero weights (baseline)
print("\n4. Testing ZERO weights (random play baseline)...")
zero_agent = TetrisAgent({name: 0.0 for name in HEURISTIC_NAMES})
zero_results = test_agent_pattern(zero_agent, "Zero Weights", num_games=5)

# Summary comparison
print("\n" + "="*60)
print("SUMMARY COMPARISON")
print("="*60)
print(f"{'Agent':<25} {'Lines':<10} {'Score':<10} {'Steps':<10} {'Center Bias':<12}")
print("-"*60)

agents = [
    ("Optimal", optimal_results),
    ("Random", random_results),
    ("Bad (Center Stack)", bad_results),
    ("Zero", zero_results)
]

for name, results in agents:
    center_bias = np.mean(results['patterns']['center_bias'])
    print(f"{name:<25} {results['lines']:<10.2f} {results['score']:<10.2f} "
          f"{results['steps']:<10.2f} {center_bias:<12.2f}")

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)

# Recommendations
print("\nRECOMMENDATIONS:")
print("-" * 60)

if optimal_results['lines'] > random_results['lines'] * 2:
    print("✓ GOOD: Optimal weights significantly outperform random")
else:
    print("⚠ WARNING: Optimal weights not much better than random")
    print("  This suggests the heuristics may need adjustment")

if abs(np.mean(bad_results['patterns']['center_bias'])) > 2:
    print("✓ GOOD: Bad weights show clear center stacking (as expected)")
else:
    print("⚠ INFO: Center stacking detection may need tuning")

if random_results['lines'] < 1:
    print("⚠ WARNING: Random weights clearing very few lines")
    print("  Consider:")
    print("    - Increasing games_per_genome in training")
    print("    - Adding more generations")
    print("    - Adjusting initial weight range")

print("\nFITNESS FUNCTION ANALYSIS:")
print("-" * 60)
print(f"Current fitness = (lines * 100) + score")
print(f"Optimal: fitness = {optimal_results['lines']*100 + optimal_results['score']:.1f}")
print(f"Random:  fitness = {random_results['lines']*100 + random_results['score']:.1f}")
print(f"Bad:     fitness = {bad_results['lines']*100 + bad_results['score']:.1f}")

if bad_results['lines']*100 + bad_results['score'] > optimal_results['lines']*100 + optimal_results['score']:
    print("\n⚠ CRITICAL: Bad weights have higher fitness than optimal!")
    print("  The fitness function needs adjustment")
else:
    print("\n✓ GOOD: Fitness function correctly ranks optimal > bad")
