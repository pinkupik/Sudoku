# import cv2
import pytesseract as tess
# import numpy as np
# from paddleocr import PaddleOCR
# ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory

def detect_digit(cell):
    """
    Detects a digit in a cell image using OCR with pytesseract.
    
    Args:
        cell: Grayscale image of a single sudoku cell
        
    Returns:
        int: Detected digit (0 if empty or unrecognized)
    """
    # Resize the cell to a larger size for better OCR results
    height, width = cell.shape
    cell = cv2.resize(cell, (width * 3, height * 3))
    
    # Apply thresholding to clean the image
    _, thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply morphological opening to remove small noise
    kernel = np.ones((3,3),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Add padding around the digit for better OCR recognition
    padded = cv2.copyMakeBorder(thresh, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0) # Corrected value for single-channel
    
    # Configure tesseract for digit recognition
    custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=123456789'
    result = tess.image_to_string(padded, config=custom_config).strip()
    
    # Try to get a valid digit
    for char in result:
        if char.isdigit() and char != '0':
            return int(char)
    
    return 0

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to get a binary image
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    return thresh

def create(image, thresh):
    # Find contours in the thresholded image
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area, and take the largest one (should be the sudoku grid)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    sudoku_contour = None
    
    # Loop through contours to find the sudoku grid
    for c in cnts:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # If we have a contour with 4 points, we can assume it's the sudoku grid
        if len(approx) == 4:
            sudoku_contour = approx
            break
    
    if sudoku_contour is None:
        return np.zeros((9, 9), dtype=int)
    
    # Apply perspective transform to get a top-down view of the grid
    pts = sudoku_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    # Order points in: top-left, top-right, bottom-right, bottom-left
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right has smallest difference
    rect[3] = pts[np.argmax(diff)]  # Bottom-left has largest difference
    
    # Compute destination points for perspective transform
    (tl, tr, br, bl) = rect
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
    
    # Apply perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    
    # Convert to grayscale if necessary
    if len(warped.shape) == 3:
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Now divide the grid into 9x9 cells
    cell_height = max_height // 9
    cell_width = max_width // 9
    
    # Add padding to cells to avoid cutting off digits
    padding = min(cell_height, cell_width) // 10
    
    sudoku = np.zeros((9, 9), dtype=int)
    
    # Extract each cell and identify digit if present
    for y in range(9):
        for x in range(9):
            # Extract cell with padding (ensuring we stay within bounds)
            y_start = max(0, y * cell_height - padding)
            y_end = min(warped.shape[0], (y + 1) * cell_height + padding)
            x_start = max(0, x * cell_width - padding)
            x_end = min(warped.shape[1], (x + 1) * cell_width + padding)
            
            cell = warped[y_start:y_end, x_start:x_end]
            
            # Preprocess individual cell before digit detection
            # Apply adaptive thresholding for better digit isolation
            cell_processed = cv2.adaptiveThreshold(
                cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2)
            
            # Remove noise
            kernel = np.ones((1, 1), np.uint8)
            cell_processed = cv2.morphologyEx(cell_processed, cv2.MORPH_OPEN, kernel)
            
            # Find center portion of the cell to reduce border interference
            h, w = cell_processed.shape
            margin_h, margin_w = int(h * 0.15), int(w * 0.15)
            center_cell = cell_processed[margin_h:h-margin_h, margin_w:w-margin_w]
            
            # Only process if there's enough content
            if center_cell.size > 0:
                sudoku[y][x] = detect_digit(center_cell)
            else:
                sudoku[y][x] = 0
    
    return sudoku

import cv2
import numpy as np
import pytesseract
# import matplotlib.pyplot as plt # Uncomment for debugging with plots

# Ak Tesseract nie je v PATH, odkomentujte a upravte nasledujúci riadok:
# On Windows, it might be:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 
# On Linux, it's often automatically found if tesseract-ocr is installed.
# On macOS (using Homebrew):
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract' # or /opt/homebrew/bin/tesseract for Apple Silicon

def order_points(pts):
    """
    Orders the four points of a contour (e.g., Sudoku grid) in a consistent
    top-left, top-right, bottom-right, bottom-left order.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def load_and_preprocess_image(image_path):
    """
    Loads an image, converts it to grayscale, applies blur, and adaptive thresholding.
    Returns the original image, the grayscale image, and the thresholded image for grid detection.
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Nebolo možné načítať obrázok: {image_path}")

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
    
    # Thresholding for grid detection (lines should be black on white or vice-versa)
    # THRESH_BINARY_INV can be good if grid lines are darker than background
    thresholded_for_grid = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2 
    )
    return original_image, gray_image, thresholded_for_grid

def find_sudoku_grid_contour(processed_image):
    """
    Finds the largest square-like contour in the processed (thresholded) image,
    assumed to be the Sudoku grid.
    Returns the contour points.
    """
    contours, _ = cv2.findContours(
        processed_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx_poly = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approx_poly) == 4 and cv2.contourArea(approx_poly) > (processed_image.shape[0] * processed_image.shape[1] * 0.1):
            return approx_poly
    return None

def warp_perspective_grid(image_for_warp, contour_points, output_size=450):
    """
    Applies a perspective transform to the Sudoku grid to get a top-down,
    bird's-eye view. Uses the grayscale image for better digit quality after warp.
    Returns the warped grid image.
    """
    if contour_points is None or len(contour_points) != 4:
        print("Chyba: Kontúra mriežky nie je štvoruholník alebo nebola nájdená.")
        return None
    ordered_pts = order_points(contour_points.reshape(4, 2))
    dst_pts = np.array([
        [0, 0], [output_size - 1, 0],
        [output_size - 1, output_size - 1], [0, output_size - 1]
    ], dtype="float32")
    transform_matrix = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
    warped_grid = cv2.warpPerspective(image_for_warp, transform_matrix, (output_size, output_size))
    return warped_grid

def split_grid_into_cells(warped_grid_image, grid_dimensions=(9, 9)):
    """
    Splits the warped Sudoku grid image into individual cell images.
    Returns a list of cell images.
    """
    rows, cols = grid_dimensions
    height, width = warped_grid_image.shape[:2]
    cell_height = height // rows
    cell_width = width // cols
    cell_images = []
    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * cell_height, (r + 1) * cell_height
            x1, x2 = c * cell_width, (c + 1) * cell_width
            cell = warped_grid_image[y1:y2, x1:x2]
            cell_images.append(cell)
    return cell_images

def preprocess_cell_for_ocr(cell_image, digit_output_size=28, cell_border_ratio=0.15):
    """
    Processes an individual cell image to isolate and prepare a potential digit for OCR.
    - Removes a border to get rid of grid lines.
    - Thresholds the cell to isolate the digit (digit as black on white background for Pytesseract).
    - Finds the largest contour (assumed to be the digit).
    - Resizes the digit to a standard size.
    Returns a processed image of the digit, or None if the cell is likely empty or unclear.
    """
    cell_h, cell_w = cell_image.shape

    border_h = int(cell_h * cell_border_ratio)
    border_w = int(cell_w * cell_border_ratio)
    
    if 2 * border_h >= cell_h or 2 * border_w >= cell_w:
        return None 

    cropped_cell = cell_image[border_h : cell_h - border_h, border_w : cell_w - border_w]

    if cropped_cell.size == 0:
        return None

    # Heuristic: If the cell is almost entirely white (empty), skip
    # Pytesseract generally prefers black text on white background.
    # The input `cell_image` is grayscale from the warped grid.
    # If mean is very high (close to 255), it's mostly white.
    # If mean is very low (close to 0), it's mostly black.
    # We are looking for a digit, which should create some contrast.
    if np.mean(cropped_cell) > 245: # Mostly white, likely empty
        return None

    # Threshold to make the digit black on a white background.
    # This is often preferred by Tesseract.
    # Otsu's method is good for automatically finding an optimal threshold.
    # We want the digit to be black (0) and background white (255).
    _, thresholded_cell_digit = cv2.threshold(
        cropped_cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU # Digit becomes white
    )
    # Invert again so digit is black on white for Pytesseract
    thresholded_cell_digit = cv2.bitwise_not(thresholded_cell_digit)


    # Find contours in the thresholded cell (where digit is now black)
    digit_contours, _ = cv2.findContours(
        cv2.bitwise_not(thresholded_cell_digit.copy()), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE # Find white contours on black
    )

    if not digit_contours:
        return None 

    largest_digit_contour = max(digit_contours, key=cv2.contourArea)
    
    min_digit_area_ratio = 0.03 
    if cv2.contourArea(largest_digit_contour) < (cropped_cell.shape[0] * cropped_cell.shape[1] * min_digit_area_ratio):
        return None

    x, y, w, h = cv2.boundingRect(largest_digit_contour)

    # Extract the digit ROI from the (digit black, bg white) thresholded cell
    # The ROI should be based on the thresholded_cell_digit which has black digit on white bg
    digit_roi = thresholded_cell_digit[y : y + h, x : x + w]
    
    # Add some padding around the digit ROI, keeping it black on white
    padding = 10 # pixels
    padded_digit = cv2.copyMakeBorder(digit_roi, padding, padding, padding, padding, 
                                      cv2.BORDER_CONSTANT, value=[255,255,255]) # White padding


    # Resize to a standard output size, maintaining aspect ratio if needed,
    # but for single digits, direct resize is often fine.
    # Pytesseract might not strictly need 28x28, but consistent size can help if further processing was planned.
    # Let's try without resizing to a fixed square first, using the padded version.
    # final_digit_image = cv2.resize(
    #     padded_digit, (digit_output_size, digit_output_size), interpolation=cv2.INTER_AREA
    # )
    final_digit_image = padded_digit # Use the padded ROI directly

    return final_digit_image

def recognize_digit_with_pytesseract(cell_image_for_ocr):
    """
    Recognizes a digit from a preprocessed cell image using Pytesseract.
    Returns the recognized digit (int 1-9) or 0 if not recognized/invalid.
    """
    if cell_image_for_ocr is None:
        return 0

    # Pytesseract configuration:
    # --psm 10: Treat the image as a single character.
    # --oem 3: Default OCR engine mode.
    # -c tessedit_char_whitelist=123456789: Only recognize digits 1-9.
    custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=123456789'
    
    try:
        text = pytesseract.image_to_string(cell_image_for_ocr, config=custom_config)
        # Clean up result (it might have newlines or spaces)
        text = text.strip()
        if text.isdigit():
            digit = int(text)
            if digit == 0:
                digit = 9
            if 1 <= digit <= 9:
                return digit
    except pytesseract.TesseractNotFoundError:
        print("Chyba: Tesseract nie je nainštalovaný alebo nie je v PATH.")
        print("Nainštalujte Tesseract OCR a/alebo nastavte pytesseract.tesseract_cmd")
        # Propagate the error or handle it by returning 0 for all subsequent cells
        raise 
    except Exception as e:
        # print(f"Chyba pri rozpoznávaní Pytesseractom: {e}")
        pass # Silently fail for a single cell, return 0
    return 0


def extract_sudoku_data_from_image(image_path, show_steps=False):
    """
    Main function to orchestrate Sudoku grid detection and digit recognition.
    Returns a 9x9 NumPy array with recognized digits (0 for empty).
    """
    try:
        original_image, gray_image, processed_for_grid_detection = load_and_preprocess_image(image_path)
        if show_steps:
            cv2.imshow("1. Thresholded for Grid Detection", processed_for_grid_detection)
            cv2.waitKey(0)
    except FileNotFoundError as e:
        print(e)
        return None
    except Exception as e:
        print(f"Chyba pri načítaní alebo predspracovaní obrázka: {e}")
        return None

    grid_contour = find_sudoku_grid_contour(processed_for_grid_detection)
    if grid_contour is None:
        print("Chyba: Sudoku mriežka nebola nájdená na obrázku.")
        return None
    
    if show_steps:
        debug_image_grid = original_image.copy()
        cv2.drawContours(debug_image_grid, [grid_contour], -1, (0, 255, 0), 3)
        cv2.imshow("2. Detected Sudoku Grid Contour", debug_image_grid)
        cv2.waitKey(0)

    # Warp using the grayscale image for better digit quality in cells
    warped_sudoku_grid = warp_perspective_grid(gray_image, grid_contour)
    if warped_sudoku_grid is None:
        print("Chyba: Nepodarilo sa transformovať perspektívu mriežky.")
        return None
    
    if show_steps:
        cv2.imshow("3. Warped Sudoku Grid (Grayscale)", warped_sudoku_grid)
        cv2.waitKey(0)

    sudoku_cell_images = split_grid_into_cells(warped_sudoku_grid)
    if len(sudoku_cell_images) != 81:
        print("Chyba: Mriežku sa nepodarilo rozdeliť na 81 buniek.")
        return None

    sudoku_matrix = np.zeros((9, 9), dtype=int)
    print("Spracovávajú sa bunky a rozpoznávajú číslice pomocou Pytesseract...")

    first_tesseract_error = True

    for i, cell_img_gray in enumerate(sudoku_cell_images):
        row, col = i // 9, i % 9
        
        # Preprocess the grayscale cell for OCR
        # `cell_img_gray` is already grayscale from the warped grid
        image_for_ocr = preprocess_cell_for_ocr(cell_img_gray.copy()) 

        recognized_digit = 0
        if image_for_ocr is not None:
            try:
                recognized_digit = recognize_digit_with_pytesseract(image_for_ocr)
            except pytesseract.TesseractNotFoundError:
                if first_tesseract_error: # Print error only once
                   # Message already printed in recognize_digit_with_pytesseract
                   first_tesseract_error = False
                # If Tesseract is not found, we can't proceed with OCR for other cells.
                # You might choose to return None or an empty matrix.
                # For now, we'll fill with 0s and let the loop finish.
                # Or, re-raise the exception to stop execution:
                # raise 
                pass # Continue filling with 0s if Tesseract is not found after first error.
            except Exception as e: # Other Pytesseract errors
                # Silently ignore for this cell, it will be 0
                pass


        sudoku_matrix[row, col] = recognized_digit
            
        if show_steps and image_for_ocr is not None:
            # Display the image that was sent to Pytesseract
            display_title = f"4. Cell ({row},{col}) for OCR. Recognized: {recognized_digit if recognized_digit else 'N/A'}"
            cv2.imshow(display_title, image_for_ocr)
            key = cv2.waitKey(50) # Small delay, press key to advance faster
            if key == 27: # ESC key to stop showing steps
                show_steps = False 
                cv2.destroyWindow(display_title) # Close current window
                # Potentially close all other step windows if needed, or let them be closed at the end
        elif show_steps and image_for_ocr is None: # Cell was deemed empty by preprocessing
             if (i < 5 or i > 75): # Show a few empty cell attempts
                empty_cell_display = np.full((50, 50), 200, dtype=np.uint8) # Gray image
                cv2.putText(empty_cell_display, "Empty", (5,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                cv2.imshow(f"4. Cell ({row},{col}) - Empty", empty_cell_display)
                cv2.waitKey(50)


    if show_steps:
        print("\nZobrazenie krokov dokončené.")
        cv2.destroyAllWindows()
        
    return sudoku_matrix

# --- Example Usage ---
if __name__ == "__main__":
    image_file_path = '/home/tomas/PYT/motustom/semestral/app/tests/images/image2.png' 
    # Test with a clear image first. You might need to adjust preprocessing
    # parameters (blur, thresholding, border_ratio in preprocess_cell_for_ocr)
    # for different image qualities.
    
    print(f"Pokus o spracovanie obrázka: {image_file_path}")
    
    # Set show_steps=True to see intermediate processing images
    # show_steps=False for just the final array
    final_sudoku_array = extract_sudoku_data_from_image(image_file_path, show_steps=False)

    if final_sudoku_array is not None:
        print("\nVýsledné NumPy pole s rozpoznanými číslicami (0 = prázdna/nerozpoznaná bunka):")
        print(final_sudoku_array)
    else:
        print("\nNepodarilo sa spracovať Sudoku z obrázka.")

    # Example with showing steps:
    # print("\nSpúšťanie s vizualizáciou krokov...")
    # final_sudoku_array_debug = extract_sudoku_data_from_image(image_file_path, show_steps=True)
    # if final_sudoku_array_debug is not None:
    #     print("\nVýsledné NumPy pole (s vizualizáciou):")
    #     print(final_sudoku_array_debug)

