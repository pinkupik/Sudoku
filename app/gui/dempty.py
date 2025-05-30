import streamlit as st

def display_empty():
        st.write("### Solved Sudoku Board")
        with st.container():
            for i in range(9):
                cols = st.columns(9, gap="small")
                for j in range(9):
                    cols[j].text_input(
                        label=f"empty_cell_{i}_{j}",
                        value="",
                        max_chars=1,
                        key=f"sudoku_empty_{i}_{j}",
                        disabled=True,
                        label_visibility="collapsed",
                        placeholder="",
                    )