"""
Exploration script to understand the tetris-gymnasium environment.
"""

import numpy as np
from tetris_gymnasium.envs import Tetris

# Create environment
env = Tetris()

print("=" * 60)
print("Tetris Gymnasium Environment Exploration")
print("=" * 60)

# Reset environment
obs, info = env.reset()

print("\n1. Observation Space:")
print(f"   Type: {env.observation_space}")
print(f"   Shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")
print(f"   Sample observation:\n{obs}")

print("\n2. Action Space:")
print(f"   Type: {env.action_space}")
print(f"   Sample action: {env.action_space.sample()}")

print("\n3. Environment Info:")
print(f"   Initial info: {info}")

print("\n4. Unwrapped Environment Attributes:")
unwrapped = env.unwrapped
print(f"   Board shape: {unwrapped.board.shape if hasattr(unwrapped, 'board') else 'N/A'}")
print(f"   Has active_tetromino: {hasattr(unwrapped, 'active_tetromino')}")

# List all attributes
print("\n5. All unwrapped attributes:")
for attr in dir(unwrapped):
    if not attr.startswith('_'):
        print(f"   - {attr}")

print("\n6. Testing a few random actions:")
for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"   Step {i+1}: action={action}, reward={reward}, terminated={terminated}")
    if terminated or truncated:
        print("   Game ended, resetting...")
        obs, info = env.reset()

env.close()

print("\n" + "=" * 60)
print("Exploration complete!")
print("=" * 60)
