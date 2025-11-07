"""
Quick test script to verify the agent works correctly.
"""

import numpy as np
from tetris_gymnasium.envs import Tetris
from agent import TetrisAgent

print("Testing Tetris Agent...")
print("=" * 60)

# Create environment
env = Tetris()

# Create agent with random weights
weights = np.random.uniform(-5, 5, 6)
print(f"Random weights: {weights}")

agent = TetrisAgent(weights)

# Play a short game
print("\nPlaying a test game...")
result = agent.play_game(env, render=False, max_steps=500)

print("\nResults:")
print(f"  Score: {result['score']}")
print(f"  Lines: {result['lines']}")
print(f"  Steps: {result['steps']}")

env.close()

print("\n" + "=" * 60)
print("Test complete! Agent is working.")
