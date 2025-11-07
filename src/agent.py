"""
Tetris agent that uses heuristic-based evaluation to play the game.

The agent evaluates all possible placements of the current piece
and selects the one with the best heuristic score.
"""

import numpy as np
import copy
from heuristics import evaluate_board, HEURISTIC_NAMES


class TetrisAgent:
    """
    Agent that plays Tetris using weighted heuristics.

    For simplicity, this agent uses a reactive strategy:
    it evaluates potential actions at each step and picks the best one.
    """

    def __init__(self, weights=None):
        """
        Initialize the agent with a set of heuristic weights.

        Args:
            weights: Dictionary or list of heuristic weights.
                     If list, must match order of HEURISTIC_NAMES.
                     If None, uses zero weights.
        """
        if weights is None:
            self.weights = {name: 0.0 for name in HEURISTIC_NAMES}
        elif isinstance(weights, dict):
            self.weights = weights
        elif isinstance(weights, (list, np.ndarray)):
            self.weights = {name: weight for name, weight in zip(HEURISTIC_NAMES, weights)}
        else:
            raise ValueError("Weights must be dict, list, or None")

        self.last_piece_id = None
        self.planned_actions = []

    def get_best_move(self, env):
        """
        Get the next best action to take.

        Uses a planning approach: when a new piece appears, evaluate all possible
        placements and plan a sequence of actions. Execute those actions one at a time.

        Args:
            env: Tetris gymnasium environment

        Returns:
            Next action to execute
        """
        # Check if we have a new piece (need to replan)
        current_piece = env.unwrapped.active_tetromino

        if current_piece is None:
            return 7  # no-op

        # Get piece ID (we'll use the piece type as identifier)
        piece_id = id(current_piece)  # Use object id as unique identifier

        # If new piece or no plan, make a new plan
        if piece_id != self.last_piece_id or len(self.planned_actions) == 0:
            self.last_piece_id = piece_id
            self.planned_actions = self._plan_best_placement(env)

        # Execute next action from plan
        if len(self.planned_actions) > 0:
            return self.planned_actions.pop(0)
        else:
            return 5  # hard_drop as fallback

    def _plan_best_placement(self, env):
        """
        Plan the best placement for the current piece.

        Returns a sequence of actions to execute.
        """
        # Try different numbers of rotations
        best_actions = []
        best_score = float('-inf')

        # Try 0-3 rotations
        for num_rotations in range(4):
            # For each rotation, try different horizontal positions
            # We'll try moving left multiple times and right multiple times

            for num_left_moves in range(6):  # Try up to 6 left moves
                actions = self._create_action_sequence(num_rotations, -num_left_moves)
                score = self._simulate_and_evaluate(env, actions)

                if score > best_score:
                    best_score = score
                    best_actions = actions.copy()

            for num_right_moves in range(1, 6):  # Try up to 6 right moves
                actions = self._create_action_sequence(num_rotations, num_right_moves)
                score = self._simulate_and_evaluate(env, actions)

                if score > best_score:
                    best_score = score
                    best_actions = actions.copy()

        return best_actions

    def _create_action_sequence(self, rotations, horizontal_move):
        """
        Create a sequence of actions for a given rotation and horizontal movement.

        Args:
            rotations: Number of clockwise rotations (0-3)
            horizontal_move: Number of horizontal moves (negative = left, positive = right)

        Returns:
            List of actions
        """
        actions = []

        # Add rotations (using clockwise rotation: action 3)
        for _ in range(rotations):
            actions.append(3)

        # Add horizontal moves
        if horizontal_move < 0:
            for _ in range(abs(horizontal_move)):
                actions.append(0)  # move left
        else:
            for _ in range(horizontal_move):
                actions.append(1)  # move right

        # Add hard drop
        actions.append(5)

        return actions

    def _simulate_and_evaluate(self, env, actions):
        """
        Simulate a sequence of actions and evaluate the resulting board.

        Args:
            env: Tetris environment
            actions: List of actions to simulate

        Returns:
            Heuristic score for the resulting board
        """
        try:
            # Get current state
            state = env.unwrapped.get_state()

            # Simulate each action
            for action in actions:
                # Check if game would be over
                if env.unwrapped.game_over:
                    env.unwrapped.set_state(state)
                    return float('-inf')

                # Take action (this modifies the environment)
                _, _, terminated, _, _ = env.step(action)

                if terminated:
                    # Restore state and return very bad score
                    env.unwrapped.set_state(state)
                    return float('-inf')

            # Evaluate the resulting board
            # Extract just the playable area (remove padding)
            board = env.unwrapped.board
            score = evaluate_board(board, self.weights)

            # Restore original state
            env.unwrapped.set_state(state)

            return score

        except Exception as e:
            # If simulation fails, restore state and return bad score
            try:
                env.unwrapped.set_state(state)
            except:
                pass
            return float('-inf')

    def play_game(self, env, render=False, max_steps=1000):
        """
        Play a complete game of Tetris.

        Args:
            env: Tetris gymnasium environment
            render: Whether to render the game
            max_steps: Maximum steps before terminating

        Returns:
            Dictionary with game statistics (score, lines, steps)
        """
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        lines_cleared = 0

        # Reset agent state
        self.last_piece_id = None
        self.planned_actions = []

        terminated = False
        truncated = False

        while not (terminated or truncated) and steps < max_steps:
            if render:
                env.render()

            # Get best action from agent
            action = self.get_best_move(env)

            # Take action
            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            # Track lines cleared if available in info
            if 'lines_cleared' in info:
                lines_cleared = info['lines_cleared']

        return {
            'score': total_reward,
            'lines': lines_cleared,
            'steps': steps,
            'info': info
        }

    def get_weights_array(self):
        """
        Get weights as a numpy array in the standard order.

        Returns:
            Numpy array of weights
        """
        return np.array([self.weights[name] for name in HEURISTIC_NAMES])

    def set_weights_array(self, weights_array):
        """
        Set weights from a numpy array.

        Args:
            weights_array: Array of weights in standard order
        """
        for name, weight in zip(HEURISTIC_NAMES, weights_array):
            self.weights[name] = weight
