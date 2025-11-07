"""
Heuristic functions for evaluating Tetris board states.

Each function takes a board state and returns a numerical value
that can be weighted by the genetic algorithm.
"""

import numpy as np


def get_column_heights(board):
    """
    Get the height of each column (distance from bottom to highest block).

    Args:
        board: 2D numpy array representing the game board

    Returns:
        List of heights for each column
    """
    heights = []
    rows, cols = board.shape

    for col in range(cols):
        height = 0
        for row in range(rows):
            if board[row, col] != 0:
                height = rows - row
                break
        heights.append(height)

    return heights


def aggregate_height(board):
    """
    Sum of all column heights. Lower is generally better.

    Args:
        board: 2D numpy array

    Returns:
        Sum of column heights
    """
    heights = get_column_heights(board)
    return sum(heights)


def complete_lines(board):
    """
    Number of complete lines (rows that will be cleared).
    Higher is better.

    Args:
        board: 2D numpy array

    Returns:
        Number of complete rows
    """
    complete = 0
    for row in board:
        if np.all(row != 0):
            complete += 1
    return complete


def holes(board):
    """
    Number of empty cells that have at least one filled cell above them.
    Lower is better (holes are bad).

    Args:
        board: 2D numpy array

    Returns:
        Number of holes
    """
    rows, cols = board.shape
    hole_count = 0

    for col in range(cols):
        block_found = False
        for row in range(rows):
            if board[row, col] != 0:
                block_found = True
            elif block_found and board[row, col] == 0:
                hole_count += 1

    return hole_count


def bumpiness(board):
    """
    Sum of absolute differences in heights between adjacent columns.
    Lower is better (smoother surface).

    Args:
        board: 2D numpy array

    Returns:
        Total bumpiness
    """
    heights = get_column_heights(board)
    total_bumpiness = 0

    for i in range(len(heights) - 1):
        total_bumpiness += abs(heights[i] - heights[i + 1])

    return total_bumpiness


def max_height(board):
    """
    Maximum column height. Lower is generally better.

    Args:
        board: 2D numpy array

    Returns:
        Height of tallest column
    """
    heights = get_column_heights(board)
    return max(heights) if heights else 0


def wells(board):
    """
    Sum of well depths. A well is a column surrounded by higher columns.
    Can be useful for setting up big clears, but generally lower is better.

    Args:
        board: 2D numpy array

    Returns:
        Sum of well depths
    """
    heights = get_column_heights(board)
    total_wells = 0

    for i in range(len(heights)):
        left_height = heights[i - 1] if i > 0 else 0
        right_height = heights[i + 1] if i < len(heights) - 1 else 0

        # Well depth is how much lower this column is than its neighbors
        well_depth = max(0, min(left_height, right_height) - heights[i])
        total_wells += well_depth

    return total_wells


def evaluate_board(board, weights):
    """
    Evaluate a board state using weighted heuristics.

    Args:
        board: 2D numpy array representing the game board
        weights: Dictionary with keys matching heuristic function names

    Returns:
        Weighted score for the board state
    """
    score = 0
    score += weights.get('aggregate_height', 0) * aggregate_height(board)
    score += weights.get('complete_lines', 0) * complete_lines(board)
    score += weights.get('holes', 0) * holes(board)
    score += weights.get('bumpiness', 0) * bumpiness(board)
    score += weights.get('max_height', 0) * max_height(board)
    score += weights.get('wells', 0) * wells(board)

    return score


# Heuristic names for genetic algorithm
HEURISTIC_NAMES = [
    'aggregate_height',
    'complete_lines',
    'holes',
    'bumpiness',
    'max_height',
    'wells'
]
