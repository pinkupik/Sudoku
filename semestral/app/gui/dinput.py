import streamlit as st
import numpy as np

def display_input(board):
        st.write("### Sudoku Board")
        board_state = []
        # Use a container to control layout
        with st.container():
            for i in range(9):
                cols = st.columns(9, gap="small")
                row = []
                for j in range(9):
                    cell_value = "" if board[i][j] == 0 else str(board[i][j])
                    value = cols[j].text_input(
                        label="",
                        value=cell_value,
                        max_chars=1,
                        key=f"sudoku_{i}_{j}",
                        disabled=False,
                        label_visibility="collapsed",
                        # Make the input box more square via CSS
                        placeholder="",
                    )
                    # Convert input to int if possible, else 0
                    try:
                        row.append(int(value))
                    except ValueError:
                        row.append(0)
                board_state.append(row)
        return np.array(board_state)