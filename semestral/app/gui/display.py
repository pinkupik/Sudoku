import sys
import os
import importlib
# 1) Create and point all cache dirs at a local '.cache' folder
CACHE_DIR = os.path.join(os.getcwd(), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HOME"] = CACHE_DIR
os.environ["XDG_CACHE_HOME"] = CACHE_DIR

# 2) Monkey-patch Paddlex's font loader so downloaded fonts go into .cache/
#    (must do this BEFORE any paddlex/utils/fonts import)
download_mod = importlib.import_module("paddlex.utils.download")
fonts_mod   = importlib.import_module("paddlex.utils.fonts")

_orig_get_font = fonts_mod.get_font_file_path
def _get_font_file_path_override(font_name: str):
    # where to save it
    dest = os.path.join(CACHE_DIR, font_name)
    if not os.path.exists(dest):
        # use the same download logic but override save path
        download_mod.download(
            url=f"https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/fonts/{font_name}",
            save_path=dest,
            print_progress=False
        )
    return dest

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
