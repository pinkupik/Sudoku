from app.gui import dinput, dsolved, dempty
import streamlit as st
from app.src import sudsolve as ssolve
from app.utils import sudcheck

def display_tables(sudoku_board):
    cols = st.columns(2, gap="large")
    with cols[0]:
        updated_board = dinput.display_input(sudoku_board)
    with cols[1]:
        if sudcheck.is_nonempty_sudoku(updated_board):
            if sudcheck.is_valid_sudoku(updated_board):
                solved_board = ssolve.solve(updated_board)
                if sudcheck.is_solved_sudoku(solved_board):
                    dsolved.display_solved(solved_board)
                else:
                    st.error("Sudoku is unsolvable. Maybe check your input?")
            else:
                st.error("Invalid Sudoku input! Please check your entries.")
        else:
            dempty.display_empty()