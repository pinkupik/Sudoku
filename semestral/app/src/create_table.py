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

def detect_blue_pen_content(image):
    """
    Enhanced blue pen detection using multiple color spaces
    Optimized for handwritten Sudoku with blue pen
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Method 1: Blue channel enhancement
    blue_channel = image[:,:,2].astype(np.float32)
    red_channel = image[:,:,0].astype(np.float32)
    green_channel = image[:,:,1].astype(np.float32)
    
    # Enhance blue relative to other channels
    blue_enhanced = blue_channel - 0.3 * (red_channel + green_channel) / 2
    blue_enhanced = np.clip(blue_enhanced, 0, 255).astype(np.uint8)
    _, blue_thresh = cv2.threshold(blue_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Method 2: HSV blue detection with multiple ranges
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Standard blue range
    lower_blue1 = np.array([100, 50, 50])
    upper_blue1 = np.array([130, 255, 255])
    blue_mask1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
    
    # Darker blue range
    lower_blue2 = np.array([110, 30, 30])
    upper_blue2 = np.array([140, 255, 200])
    blue_mask2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
    
    # Method 3: LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    b_channel = lab[:,:,2]
    _, lab_blue = cv2.threshold(b_channel, 140, 255, cv2.THRESH_BINARY)
    
    # Combine all blue detection methods
    combined_blue = cv2.bitwise_or(blue_thresh, blue_mask1)
    combined_blue = cv2.bitwise_or(combined_blue, blue_mask2)
    combined_blue = cv2.bitwise_or(combined_blue, lab_blue)
    
    # Clean up noise
    kernel = np.ones((2,2), np.uint8)
    combined_blue = cv2.morphologyEx(combined_blue, cv2.MORPH_OPEN, kernel)
    
    return combined_blue

def load_and_preprocess_image(image_path):
    """
    Enhanced image loading and preprocessing with blue pen detection
    Returns the original image, the grayscale image, and improved thresholded image for grid detection.
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Nebolo možné načítať obrázok: {image_path}")

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Enhanced detection combining traditional methods with blue pen detection
    # Traditional adaptive thresholding
    traditional_thresh = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2 
    )
    
    # Blue pen detection
    blue_pen_mask = detect_blue_pen_content(original_image)
    
    # Edge detection for grid lines
    edges = cv2.Canny(blurred_image, 50, 150)
    
    # Combine all detection methods
    combined_mask = cv2.bitwise_or(traditional_thresh, blue_pen_mask)
    combined_mask = cv2.bitwise_or(combined_mask, edges)
    
    # Clean up the combined mask
    kernel_close = np.ones((5,5), np.uint8)
    thresholded_for_grid = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)
    
    kernel_open = np.ones((3,3), np.uint8)
    thresholded_for_grid = cv2.morphologyEx(thresholded_for_grid, cv2.MORPH_OPEN, kernel_open)
    
    return original_image, gray_image, thresholded_for_grid

def find_sudoku_grid_contour(processed_image):
    """
    Enhanced function to find the largest square-like contour in the processed image.
    Uses multiple approaches to find the Sudoku grid with better robustness.
    Returns the contour points.
    """
    contours, _ = cv2.findContours(
        processed_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    
    # Sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Calculate minimum area threshold (more flexible)
    image_area = processed_image.shape[0] * processed_image.shape[1]
    min_area_threshold = image_area * 0.05  # At least 5% of image
    max_area_threshold = image_area * 0.90  # At most 90% of image
    
    # Try to find a good rectangular contour
    for c in contours:
        area = cv2.contourArea(c)
        
        # Skip contours that are too small or too large
        if area < min_area_threshold or area > max_area_threshold:
            continue
            
        perimeter = cv2.arcLength(c, True)
        
        # Try different approximation levels
        for epsilon_factor in [0.01, 0.02, 0.03, 0.05, 0.08]:
            epsilon = epsilon_factor * perimeter
            approx_poly = cv2.approxPolyDP(c, epsilon, True)
            
            if len(approx_poly) == 4:
                # Check if it's reasonably square
                rect = cv2.minAreaRect(c)
                width, height = rect[1]
                if width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    if aspect_ratio < 2.5:  # Allow some tolerance for perspective
                        return approx_poly
    
    # Fallback: use bounding rectangle of largest valid contour
    for c in contours:
        area = cv2.contourArea(c)
        if min_area_threshold <= area <= max_area_threshold:
            x, y, w, h = cv2.boundingRect(c)
            return np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)
    
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
    Enhanced cell preprocessing optimized for blue pen handwriting.
    Processes an individual cell image to isolate and prepare a potential digit for OCR.
    - Removes a border to get rid of grid lines.
    - Uses multiple thresholding methods to handle blue pen better.
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

    # Enhanced emptiness check - more robust for blue pen
    mean_intensity = np.mean(cropped_cell)
    std_intensity = np.std(cropped_cell)
    
    # If very uniform (low standard deviation) and bright, likely empty
    if mean_intensity > 240 and std_intensity < 10:
        return None
    
    # If very uniform and dark, might be noise
    if mean_intensity < 15 and std_intensity < 10:
        return None

    # Multiple thresholding approaches for better blue pen handling
    # Method 1: OTSU thresholding
    _, thresh_otsu = cv2.threshold(cropped_cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Method 2: Adaptive thresholding
    thresh_adaptive = cv2.adaptiveThreshold(cropped_cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
    
    # Method 3: Manual threshold for blue pen (often appears as gray)
    _, thresh_manual = cv2.threshold(cropped_cell, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Combine thresholding methods
    combined_thresh = cv2.bitwise_or(thresh_otsu, thresh_adaptive)
    combined_thresh = cv2.bitwise_or(combined_thresh, thresh_manual)
    
    # Clean up noise
    kernel_noise = np.ones((2,2), np.uint8)
    combined_thresh = cv2.morphologyEx(combined_thresh, cv2.MORPH_OPEN, kernel_noise)
    
    # Fill small holes in digits
    kernel_fill = np.ones((3,3), np.uint8)
    combined_thresh = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel_fill)

    # Find contours in the enhanced thresholded cell
    digit_contours, _ = cv2.findContours(combined_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not digit_contours:
        return None 

    # Filter contours by area and find the best candidate
    valid_contours = []
    min_area = (cropped_cell.shape[0] * cropped_cell.shape[1]) * 0.02  # At least 2% of cell
    max_area = (cropped_cell.shape[0] * cropped_cell.shape[1]) * 0.80  # At most 80% of cell
    
    for contour in digit_contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            # Check aspect ratio to filter out non-digit shapes
            x, y, w, h = cv2.boundingRect(contour)
            if w > 0 and h > 0:
                aspect_ratio = max(w, h) / min(w, h)
                if aspect_ratio < 5.0:  # Reasonable aspect ratio for digits
                    valid_contours.append((contour, area))
    
    if not valid_contours:
        return None
    
    # Choose the largest valid contour
    largest_digit_contour = max(valid_contours, key=lambda x: x[1])[0]
    
    x, y, w, h = cv2.boundingRect(largest_digit_contour)

    # Extract the digit ROI - invert so digit is black on white for Tesseract
    digit_roi = combined_thresh[y : y + h, x : x + w]
    digit_roi_inverted = cv2.bitwise_not(digit_roi)  # Black digit on white background
    
    # Add padding around the digit ROI
    padding = 10
    padded_digit = cv2.copyMakeBorder(digit_roi_inverted, padding, padding, padding, padding, 
                                      cv2.BORDER_CONSTANT, value=[255,255,255])

    return padded_digit

def recognize_digit_with_pytesseract(cell_image_for_ocr):
    """
    Enhanced digit recognition optimized for blue pen handwriting.
    Recognizes a digit from a preprocessed cell image using Pytesseract.
    Returns the recognized digit (int 1-9) or 0 if not recognized/invalid.
    """
    if cell_image_for_ocr is None:
        return 0

    # Enhanced Pytesseract configuration for better blue pen digit recognition
    # Try multiple configurations and take the best result
    configs = [
        r'--oem 3 --psm 10 -c tessedit_char_whitelist=123456789',  # Single character
        r'--oem 3 --psm 8 -c tessedit_char_whitelist=123456789',   # Single word
        r'--oem 3 --psm 7 -c tessedit_char_whitelist=123456789',   # Single text line
        r'--oem 1 --psm 10 -c tessedit_char_whitelist=123456789',  # LSTM engine
    ]
    
    recognized_digits = []
    
    try:
        # Apply slight morphological operations for better recognition
        kernel = np.ones((2,2), np.uint8)
        enhanced_image = cv2.morphologyEx(cell_image_for_ocr, cv2.MORPH_CLOSE, kernel)
        
        # Try each configuration
        for config in configs:
            try:
                text = pytesseract.image_to_string(enhanced_image, config=config)
                text = text.strip()
                
                # Extract valid digits from the result
                for char in text:
                    if char.isdigit() and '1' <= char <= '9':
                        recognized_digits.append(int(char))
                
                # Also try with original image
                text_orig = pytesseract.image_to_string(cell_image_for_ocr, config=config)
                text_orig = text_orig.strip()
                
                for char in text_orig:
                    if char.isdigit() and '1' <= char <= '9':
                        recognized_digits.append(int(char))
                        
            except Exception:
                continue
        
        # If we have recognized digits, return the most common one
        if recognized_digits:
            from collections import Counter
            counter = Counter(recognized_digits)
            most_common_digit = counter.most_common(1)[0][0]
            return most_common_digit
            
    except pytesseract.TesseractNotFoundError:
        print("Chyba: Tesseract nie je nainštalovaný alebo nie je v PATH.")
        print("Nainštalujte Tesseract OCR a/alebo nastavte pytesseract.tesseract_cmd")
        raise 
    except Exception as e:
        # Silently fail for a single cell
        pass
    
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
    image_file_path = '/home/tomas/PYT/motustom/semestral/app/tests/images/image3.png' 
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

