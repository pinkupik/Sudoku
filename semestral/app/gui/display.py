import sys
import os
import importlib
from pathlib import Path

# Ensure paddlex writes to a writable directory
os.environ["HOME"] = "/tmp"
os.environ["XDG_CACHE_HOME"] = "/tmp"

# Patch font download path before paddlex loads fonts
download_mod = importlib.import_module("paddlex.utils.download")
fonts_mod = importlib.import_module("paddlex.utils.fonts")

_orig_get_font = fonts_mod.get_font_file_path

def _get_font_file_path_override(font_name: str):
    font_path = Path("/tmp") / font_name
    if not font_path.is_file():
        download_mod.download(
            url=f"https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/fonts/{font_name}",
            save_path=str(font_path)
        )
    return str(font_path)

fonts_mod.get_font_file_path = _get_font_file_path_override

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import numpy as np
from src import sudsolve as ssolve
from gui import dinput
from gui import dsolved
from src import sudscan as sscan

st.set_page_config(layout="wide")  # Use the whole width of the screen

def display():
    st.header("Crazy AI Sudoku Solver++ Ultra Edition")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        with st.expander("Show Uploaded Image"):
            st.image(uploaded_file, caption="Uploaded Image", width=300)
        # Save the uploaded file to a temporary location
        temp_image_path = "temp_uploaded_image.png"
        with open(temp_image_path, "wb+") as f:
            f.write(uploaded_file.getbuffer())
        # Process the image to extract the Sudoku grid
        sudoku_board = sscan.scan_table(temp_image_path)
        # Optionally, remove the temporary file after processing
    else:
        # Example empty Sudoku board
        sudoku_board = np.zeros((9, 9), dtype=int)
    # Custom CSS for wider input boxes
    st.markdown("""
        <style>
        div[data-baseweb="input"] input {
            width: 2.5em !important;
            height: 1.18em !important;
            text-align: center !important;
            font-size: 2em !important;
            padding: 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    cols = st.columns(2, gap="large")
    with cols[0]:
        updated_board = dinput.display_input(sudoku_board)
    with cols[1]:
        dsolved.display_solved(ssolve.solve(updated_board))


if __name__ == "__main__":
    display()
