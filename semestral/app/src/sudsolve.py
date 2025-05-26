"""
Sudoku Solver Module

This module provides functionality to solve Sudoku puzzles using a backtracking algorithm.
The solver takes a 9x9 grid with empty cells represented as zeros and attempts to fill
all cells with valid numbers (1-9) according to standard Sudoku rules.

The module uses numpy arrays for efficient grid manipulation and implements a recursive
backtracking approach to explore possible solutions systematically.

Functions:
    solve(grid): Solves a given Sudoku grid and returns the solution or original grid.

Example:
    >>> import numpy as np
    >>> from sudsolve import solve
    >>> 
    >>> # Create a Sudoku puzzle (0 represents empty cells)
    >>> puzzle = np.array([
    ...     [5, 3, 0, 0, 7, 0, 0, 0, 0],
    ...     [6, 0, 0, 1, 9, 5, 0, 0, 0],
    ...     # ... rest of the puzzle
    ... ])
    >>> 
    >>> solution = solve(puzzle)
    >>> print(solution)
"""
import numpy as np


def solve(grid: np.ndarray) -> np.ndarray:
    """
    Solves the given Sudoku grid using a backtracking algorithm.

    :param grid: A 9x9 numpy array representing the Sudoku grid, where 0 represents an empty cell.
    :return: A solved 9x9 numpy array if a solution exists, otherwise returns the original grid.
    """
    def is_valid(num, row, col):
        # Check if num is not in the current row
        if num in grid[row]:
            return False
        # Check if num is not in the current column
        if num in grid[:, col]:
            return False
        # Check if num is not in the current 3x3 subgrid
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        if num in grid[start_row:start_row + 3, start_col:start_col + 3]:
            return False
        return True

    def solve_sudoku():
        for row in range(9):
            for col in range(9):
                if grid[row, col] == 0:  # Find an empty cell
                    for num in range(1, 10):  # Try numbers from 1 to 9
                        if is_valid(num, row, col):
                            grid[row, col] = num  # Place the number
                            if solve_sudoku():  # Recur to solve the rest of the grid
                                return True
                            grid[row, col] = 0  # Reset on backtrack
                    return False  # No valid number found, trigger backtrack
        return True  # All cells are filled correctly

    if solve_sudoku():
        return grid
    return np.copy(grid)  # Return original grid if no solution exists
