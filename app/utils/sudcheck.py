import numpy as np

def is_valid_sudoku(board):
    """
    Check if a given Sudoku board is valid.
    
    Args:
        board (np.ndarray): 9x9 Sudoku board with values 0-9.
        
    Returns:
        bool: True if the board is valid, False otherwise.
    """
    # Check rows
    for row in board:
        if not is_valid_group(row):
            return False
    
    # Check columns
    for col in board.T:
        if not is_valid_group(col):
            return False
    
    # Check 3x3 subgrids
    for i in range(3):
        for j in range(3):
            subgrid = board[i*3:(i+1)*3, j*3:(j+1)*3].flatten()
            if not is_valid_group(subgrid):
                return False
    
    return True

def is_valid_group(group):
    """
    Check if a group (row, column, or subgrid) is valid.
    
    Args:
        group (np.ndarray): 1D array of Sudoku values.
        
    Returns:
        bool: True if the group is valid, False otherwise.
    """
    seen = set()
    for value in group:
        if value != 0:  # Ignore empty cells
            if value in seen or not (1 <= value <= 9):
                return False
            seen.add(value)
    return True

def is_nonempty_sudoku(board):
    """
    Check if at least one element is in the Sudoku board.
    Args:
        board (np.ndarray): 9x9 Sudoku board with values 0-9.
    Returns:
        bool: True if at least one element is in the board, False otherwise.
    """
    # Check if the board is empty
    if board.size == 0 or np.all(board == 0):
        return False
    return True

def is_solved_sudoku(board):
    """
    Check if a Sudoku board is completely solved.
    
    Args:
        board (np.ndarray): 9x9 Sudoku board with values 0-9.
        
    Returns:
        bool: True if the board is completely solved, False otherwise.
    """
    return np.all(board != 0) and is_valid_sudoku(board)