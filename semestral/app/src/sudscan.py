"""
Module for scanning and extracting Sudoku grids from images using OCR.

This module provides functionality to process images containing Sudoku puzzles
and convert them into numerical matrices using PaddleOCR for text recognition.
"""
import numpy as np
import cv2


def scan_table(image_path):
    """
    Extract a Sudoku grid from an image using OCR technology.

    This function processes an image containing a Sudoku puzzle and converts it
    into a 9x9 numerical matrix. It uses PaddleOCR to detect and recognize digits
    within the image, then maps them to their corresponding positions in the grid
    based on their spatial location.

    Args:
        image_path (str): Path to the image file containing the Sudoku puzzle.
    Returns:
        numpy.ndarray: A 9x9 integer matrix representing the Sudoku grid, where
                      0 represents empty cells and 1-9 represent the detected digits.
    Note:
        - The function assumes the input image contains a standard 9x9 Sudoku grid
        - Text boxes smaller than 30% of expected cell size are filtered out
        - Multiple digits in a single detection are handled by placing them in
          consecutive columns
        - Detected text outside the 9x9 grid boundaries is ignored
    """

    import paddleocr
    ocr = paddleocr.PaddleOCR(use_textline_orientation=False, lang='en',
                    text_detection_model_dir="app/official_models/PP-OCRv5_mobile_det",
                    text_recognition_model_dir="app/official_models/PP-OCRv5_mobile_rec",
                    doc_orientation_classify_model_dir="app/official_models/PP-LCNet_x1_0_doc_ori",
                    text_line_orientation_model_dir="app/official_models/PP-LCNet_x0_25_textline_ori",
                    doc_unwarping_model_dir="app/official_models/UVDoc")  # Initialize OCR model
    sudoku_matrix = np.zeros((9, 9), dtype=int)  # Initialize empty Sudoku matrix
    image = ocr.predict(image_path)  # Perform OCR on the image
    image_width = cv2.imread(image_path).shape[1]  # Get image width #pylint: disable=E1101 #(cv2.imread does not exist)
    image_height = cv2.imread(image_path).shape[0]  # Get image height #pylint: disable=E1101 #(cv2.imread does not exist)

    # Calculate expected cell dimensions
    expected_cell_width = image_width / 9
    expected_cell_height = image_height / 9
    min_box_size_threshold = 0.3  # Minimum 30% of expected cell size

    for res in image:
        for poly, text in zip(res["rec_polys"], res["rec_texts"]):
            # Calculate bounding box dimensions
            x_coords = [point[0] for point in poly]
            y_coords = [point[1] for point in poly]
            box_width = max(x_coords) - min(x_coords)
            box_height = max(y_coords) - min(y_coords)

            # Check if box is large enough
            if (box_width < expected_cell_width * min_box_size_threshold or
                box_height < expected_cell_height * min_box_size_threshold):
                continue  # Skip if box is too small

            # Get position
            col = int(((poly[1][0] + poly[0][0]) / 2) // expected_cell_width)
            row = int(((poly[2][1] + poly[0][1]) / 2) // expected_cell_height)
            if col < 0 or col >= 9 or row < 0 or row >= 9:
                continue
            # Handle multiple digits in text
            digits = [char for char in text[0] if char.isdigit()]

            for i, digit in enumerate(digits):
                target_col = col + i
                if target_col < 9 and row < 9:  # Ensure within bounds
                    sudoku_matrix[row, target_col] = int(digit)
    return sudoku_matrix
