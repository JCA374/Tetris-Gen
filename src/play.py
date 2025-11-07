"""
Script to watch a trained Tetris agent play.

Load a saved genome and watch it play Tetris with visualization.
"""

import argparse
import pickle
import time
import sys

import numpy as np
from tetris_gymnasium.envs import Tetris

from agent import TetrisAgent
from heuristics import HEURISTIC_NAMES


def load_genome(filepath):
    """
    Load a genome from a pickle file.

    Args:
        filepath: Path to the genome file

    Returns:
        Weights array
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        if 'weights' in data:
            weights = data['weights']
            print(f"Loaded genome with fitness: {data.get('fitness', 'N/A')}")
            if 'generation' in data:
                print(f"From generation: {data['generation']}")
        elif 'best_genome' in data:
            weights, fitness = data['best_genome']
            print(f"Loaded best genome with fitness: {fitness}")
    else:
        weights = data

    return weights


def print_weights(weights):
    """
    Print weights in a readable format.

    Args:
        weights: Array of weights
    """
    print("\nGenome Weights:")
    for name, weight in zip(HEURISTIC_NAMES, weights):
        print(f"  {name:20s}: {weight:8.3f}")


def main():
    parser = argparse.ArgumentParser(description='Watch a trained Tetris AI play')
    parser.add_argument('--genome', type=str, required=True,
                        help='Path to saved genome file (.pkl)')
    parser.add_argument('--games', type=int, default=1,
                        help='Number of games to play (default: 1)')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='Delay between moves in seconds (default: 0.1)')
    parser.add_argument('--no-render', action='store_true',
                        help='Do not render the game (just show statistics)')

    args = parser.parse_args()

    # Load genome
    print(f"Loading genome from: {args.genome}")
    try:
        weights = load_genome(args.genome)
        print_weights(weights)
    except Exception as e:
        print(f"Error loading genome: {e}")
        sys.exit(1)

    # Create agent
    agent = TetrisAgent(weights)

    # Play games
    print(f"\nPlaying {args.games} game(s)...")
    print("=" * 60)

    total_score = 0
    total_lines = 0
    total_steps = 0

    for game_num in range(args.games):
        print(f"\nGame {game_num + 1}/{args.games}")

        # Create environment
        if args.no_render:
            env = Tetris()
        else:
            env = Tetris(render_mode='human')

        # Play game
        observation, info = env.reset()
        terminated = False
        truncated = False
        score = 0
        steps = 0
        lines = 0

        while not (terminated or truncated):
            if not args.no_render:
                env.render()
                time.sleep(args.delay)

            # Get action from agent
            action = agent.get_best_move(env)

            # Take action
            observation, reward, terminated, truncated, info = env.step(action)

            score += reward
            steps += 1

            if 'lines_cleared' in info:
                lines = info['lines_cleared']

        env.close()

        print(f"  Final Score: {score}")
        print(f"  Lines Cleared: {lines}")
        print(f"  Steps: {steps}")

        total_score += score
        total_lines += lines
        total_steps += steps

    # Summary
    if args.games > 1:
        print("\n" + "=" * 60)
        print("Summary Statistics:")
        print(f"  Average Score: {total_score / args.games:.2f}")
        print(f"  Average Lines: {total_lines / args.games:.2f}")
        print(f"  Average Steps: {total_steps / args.games:.2f}")
        print("=" * 60)


if __name__ == '__main__':
    main()
