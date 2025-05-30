import streamlit as st

def display_solved(board):
        st.write("### Solved Sudoku Board")
        with st.container():
            for i in range(9):
                cols = st.columns(9, gap="small")
                for j in range(9):
                    cell_value = "" if board[i][j] == 0 else str(board[i][j])
                    cols[j].text_input(
                        label=f"solved_cell_{i}_{j}",
                        value=cell_value,
                        max_chars=1,
                        key=f"sudoku_replica_{i}_{j}",
                        disabled=True,
                        label_visibility="collapsed",
                        placeholder="",
                    )