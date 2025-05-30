import streamlit as st
import numpy as np
import cv2
import os
from app.gui import dtables
from app.src import sudscan as sscan
from app.src.sudoku_detector import SudokuBoardDetector

def display():
    st.header("Crazy AI Sudoku Solver++ Ultra Edition")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    captured_image = None
    if uploaded_file is None:
        # Show camera input only after pressing "Take a photo"
        if "show_camera" not in st.session_state:
            st.session_state["show_camera"] = False

        if st.button("Take a photo", key="take_photo_button"):
            if st.session_state["show_camera"] == False:
                st.session_state["show_camera"] = True
            else:
                st.session_state["show_camera"] = False

        if st.session_state["show_camera"]:
            captured_image = st.camera_input("Or take a photo")
            if captured_image is not None:
                st.session_state["show_camera"] = False  # Hide camera after capture
        else:
            captured_image = None

    # Prefer captured image if available, else uploaded file
    image_file = captured_image if captured_image is not None else uploaded_file
    if image_file is not None:
        # Save the uploaded file to a temporary location
        temp_image_path = "app/utils"
        with open(os.path.join(temp_image_path, 'temp_uploaded_image.png'), "wb+") as f:
            f.write(image_file.getbuffer())
        detector = SudokuBoardDetector(output_size=450, debug=False)
        cropped_image = detector.detect_and_extract(os.path.join(temp_image_path, 'temp_uploaded_image.png'))
        cv2.imwrite(os.path.join(temp_image_path, 'cropped.jpg'), cropped_image['board'])
        with st.expander("Show Uploaded Image"):
            image_cols = st.columns(2, gap="large")
            image_cols[0].image(image_file, caption="Uploaded Image", width=300)
            image_cols[1].image(cropped_image['board'], caption="Detected Sudoku Board", width=300)
        # Process the image to extract the Sudoku grid
        sudoku_board = sscan.scan_table(os.path.join(temp_image_path, 'cropped.jpg'))
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

    dtables.display_tables(sudoku_board)

if __name__ == "__main__":
    display()