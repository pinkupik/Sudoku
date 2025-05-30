"""
Comprehensive Sudoku Board Detection and Extraction Module

This module provides robust detection and extraction of Sudoku boards from images,
with special optimization for handwritten content including blue pen writing.

Features:
- Multiple detection methods for different image types
- Blue pen writing detection using multiple color spaces
- Grid structure detection for traditional printed Sudoku
- Adaptive content detection for handwritten digits
- Perspective correction for angled boards
- Robust rectangle detection with fallback options

Usage:
    from sudoku_detector import SudokuBoardDetector
    
    detector = SudokuBoardDetector()
    result = detector.detect_and_extract('path/to/sudoku_image.jpg')
    
    if result['success']:
        extracted_board = result['board']
        # Use the extracted board for further processing
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


class SudokuBoardDetector:
    """
    Advanced Sudoku board detector optimized for various image types
    including handwritten content with blue pen
    """
    
    def __init__(self, output_size=450, debug=False):
        """Initialize the detector
        
        Args:
            output_size (int): Size of the extracted square board
            debug (bool): Enable debug visualizations
        """
        self.output_size = output_size
        self.debug = debug

    def enhance_blue_pen_content(self, image):
        """
        Detect and enhance blue pen writing using multiple color space approaches
        
        Args:
            image: RGB image array
            
        Returns:
            Binary mask highlighting blue pen content
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        h, w = image.shape[:2]
        
        # Method 1: Blue channel enhancement with comparison
        blue = image[:,:,2].astype(np.float32)
        red = image[:,:,0].astype(np.float32)
        green = image[:,:,1].astype(np.float32)
        
        # Enhance blue relative to other channels
        blue_enhanced = blue - 0.3 * (red + green) / 2
        blue_enhanced = np.clip(blue_enhanced, 0, 255).astype(np.uint8)
        
        # Apply OTSU thresholding
        _, blue_thresh = cv2.threshold(blue_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 2: HSV blue detection with multiple ranges
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Range 1: Standard blue
        lower_blue1 = np.array([100, 50, 50])
        upper_blue1 = np.array([130, 255, 255])
        mask_blue1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
        
        # Range 2: Darker blue
        lower_blue2 = np.array([110, 30, 30])
        upper_blue2 = np.array([140, 255, 200])
        mask_blue2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
        
        # Range 3: Light blue
        lower_blue3 = np.array([90, 20, 60])
        upper_blue3 = np.array([120, 255, 255])
        mask_blue3 = cv2.inRange(hsv, lower_blue3, upper_blue3)
        
        # Method 3: LAB color space detection
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        b_channel = lab[:,:,2]
        
        # Blue in LAB has higher b* values (> 128)
        _, lab_blue = cv2.threshold(b_channel, 140, 255, cv2.THRESH_BINARY)
        
        # Method 4: YUV color space
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        u_channel = yuv[:,:,1]
        # Blue has higher U values
        _, yuv_blue = cv2.threshold(u_channel, 135, 255, cv2.THRESH_BINARY)
        
        # Combine all detection methods
        combined_blue = cv2.bitwise_or(blue_thresh, mask_blue1)
        combined_blue = cv2.bitwise_or(combined_blue, mask_blue2)
        combined_blue = cv2.bitwise_or(combined_blue, mask_blue3)
        combined_blue = cv2.bitwise_or(combined_blue, lab_blue)
        combined_blue = cv2.bitwise_or(combined_blue, yuv_blue)
        
        # Clean up noise and small artifacts
        kernel_open = np.ones((2,2), np.uint8)
        combined_blue = cv2.morphologyEx(combined_blue, cv2.MORPH_OPEN, kernel_open)
        
        kernel_close = np.ones((3,3), np.uint8)
        combined_blue = cv2.morphologyEx(combined_blue, cv2.MORPH_CLOSE, kernel_close)
        
        return combined_blue

    def detect_grid_structure(self, gray_image):
        """
        Detect grid lines and structural elements of the Sudoku board
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Tuple of (grid_mask, edges, horizontal_lines, vertical_lines)
        """
        # Preprocessing
        blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
        
        # Edge detection with multiple parameters
        edges1 = cv2.Canny(blurred, 20, 80)
        edges2 = cv2.Canny(blurred, 50, 150)
        edges3 = cv2.Canny(blurred, 80, 200)
        
        # Combine edge results
        edges_combined = cv2.bitwise_or(edges1, edges2)
        edges_combined = cv2.bitwise_or(edges_combined, edges3)
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(edges_combined, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(edges_combined, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine grid lines
        grid_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
        
        # Enhance grid intersections
        kernel_cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        intersections = cv2.morphologyEx(grid_mask, cv2.MORPH_OPEN, kernel_cross)
        
        # Combine grid with intersections
        grid_enhanced = cv2.bitwise_or(grid_mask, intersections)
        
        return grid_enhanced, edges_combined, horizontal_lines, vertical_lines

    def detect_handwritten_content(self, gray_image):
        """
        Detect handwritten content using adaptive thresholding methods
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Binary mask of detected content
        """
        # Multiple adaptive thresholding approaches
        methods = [
            (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11, 2),
            (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 15, 3),
            (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 19, 4),
            (cv2.ADAPTIVE_THRESH_MEAN_C, 11, 2),
            (cv2.ADAPTIVE_THRESH_MEAN_C, 15, 3),
        ]
        
        adaptive_results = []
        for method, block_size, C in methods:
            result = cv2.adaptiveThreshold(gray_image, 255, method, cv2.THRESH_BINARY_INV, block_size, C)
            adaptive_results.append(result)
        
        # Combine adaptive threshold results
        combined_adaptive = adaptive_results[0]
        for result in adaptive_results[1:]:
            combined_adaptive = cv2.bitwise_or(combined_adaptive, result)
        
        # Remove very small noise
        kernel_noise = np.ones((2,2), np.uint8)
        combined_adaptive = cv2.morphologyEx(combined_adaptive, cv2.MORPH_OPEN, kernel_noise)
        
        return combined_adaptive

    def combine_detection_methods(self, image):
        """
        Combine all detection methods for robust board detection
        
        Args:
            image: Input image (RGB)
            
        Returns:
            Tuple of detection results
        """
        # height = image.shape[0]
        # width = image.shape[1]
        # image = image[height//50:-height//50,width//50:-width//50]  # Crop to avoid borders
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply all detection methods
        grid_mask, edges, h_lines, v_lines = self.detect_grid_structure(gray)
        content_mask = self.detect_handwritten_content(gray)
        
        # Create weighted combination
        h, w = gray.shape
        combined = np.zeros((h, w), dtype=np.float32)
        
        # Assign weights based on detection confidence
        combined += grid_mask.astype(np.float32) * 0.30      # Grid structure is important
        combined += content_mask.astype(np.float32) * 0.25   # General handwritten content
        combined += edges.astype(np.float32) * 0.10          # Edge information
        
        # Normalize and convert to binary
        combined = np.clip(combined, 0, 255).astype(np.uint8)
        
        # Apply threshold to create final binary mask
        _, final_mask = cv2.threshold(combined, 60, 255, cv2.THRESH_BINARY)
        
        # Morphological cleanup
        kernel_close = np.ones((5,5), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close)
        
        kernel_open = np.ones((3,3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open)
        
        return {
            'combined_mask': final_mask,
            'grid_structure': grid_mask,
            'content': content_mask,
            'edges': edges
        }

    def find_board_rectangle(self, mask, image_shape):
        """
        Find the best rectangular contour representing the Sudoku board
        
        Args:
            mask: Binary detection mask
            image_shape: Shape of original image
            
        Returns:
            Corner points of detected rectangle or None
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Calculate area constraints
        image_area = image_shape[0] * image_shape[1]
        min_area = image_area * 0.03  # At least 3% of image
        max_area = image_area * 0.90  # At most 90% of image
        
        # Filter contours by area
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                valid_contours.append(contour)
        
        if not valid_contours:
            return None
        
        # Sort by area (largest first)
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
        
        # Try to find a good rectangular approximation
        for contour in valid_contours:
            # Try different epsilon values for polygon approximation
            perimeter = cv2.arcLength(contour, True)
            
            for epsilon_factor in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]:
                epsilon = epsilon_factor * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:
                    # Check if the quadrilateral is reasonably square
                    rect = cv2.minAreaRect(contour)
                    width, height = rect[1]
                    
                    if width > 0 and height > 0:
                        aspect_ratio = max(width, height) / min(width, height)
                        
                        # Accept if reasonably square (allow some tolerance)
                        if aspect_ratio <= 3.0:
                            return approx
        
        # Fallback: use bounding rectangle of largest valid contour
        if valid_contours:
            largest_contour = valid_contours[0]
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Create rectangle points
            rectangle_points = np.array([
                [[x, y]],
                [[x + w, y]],
                [[x + w, y + h]],
                [[x, y + h]]
            ], dtype=np.int32)
            
            return rectangle_points
        
        return None

    def order_corner_points(self, corners):
        """
        Order corner points in consistent order: top-left, top-right, bottom-right, bottom-left
        
        Args:
            corners: Array of 4 corner points
            
        Returns:
            Ordered corner points
        """
        # Reshape to (4, 2)
        points = corners.reshape(4, 2).astype(np.float32)
        
        # Initialize ordered rectangle
        ordered = np.zeros((4, 2), dtype=np.float32)
        
        # Sum of coordinates to find top-left (minimum) and bottom-right (maximum)
        point_sums = points.sum(axis=1)
        ordered[0] = points[np.argmin(point_sums)]  # top-left
        ordered[2] = points[np.argmax(point_sums)]  # bottom-right
        
        # Difference of coordinates to find top-right and bottom-left
        point_diffs = np.diff(points, axis=1)
        ordered[1] = points[np.argmin(point_diffs)]  # top-right
        ordered[3] = points[np.argmax(point_diffs)]  # bottom-left
        
        return ordered

    def extract_board_perspective(self, image, corners):
        """
        Extract the Sudoku board using perspective transformation
        
        Args:
            image: Original image
            corners: Corner points of the board
            
        Returns:
            Warped board image
        """
        # Order the corner points
        ordered_corners = self.order_corner_points(corners)
        
        # Define destination points for a square output
        dst_points = np.array([
            [0, 0],
            [self.output_size - 1, 0],
            [self.output_size - 1, self.output_size - 1],
            [0, self.output_size - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transformation matrix
        transform_matrix = cv2.getPerspectiveTransform(ordered_corners, dst_points)
        
        # Apply perspective transformation
        warped_image = cv2.warpPerspective(image, transform_matrix, 
                                         (self.output_size, self.output_size))
        
        return warped_image

    def detect_and_extract(self, image_path_or_array):
        """
        Main method to detect and extract Sudoku board
        
        Args:
            image_path_or_array: Path to image file or image array
            
        Returns:
            Dictionary with detection results
        """
        # Load image
        if isinstance(image_path_or_array, str):
            if not os.path.exists(image_path_or_array):
                return {'success': False, 'error': 'Image file not found'}
            
            # Load based on file extension
            if image_path_or_array.lower().endswith('.jpg') or image_path_or_array.lower().endswith('.jpeg'):
                image = cv2.imread(image_path_or_array)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                try:
                    import skimage.io
                    image = skimage.io.imread(image_path_or_array)
                except ImportError:
                    image = cv2.imread(image_path_or_array)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if image is None:
                return {'success': False, 'error': 'Could not load image'}
        else:
            image = image_path_or_array.copy()
        
        try:
            # Perform detection
            detection_results = self.combine_detection_methods(image)
            
            # Find board rectangle
            corners = self.find_board_rectangle(detection_results['combined_mask'], image.shape)
            
            result = {
                'success': False,
                'original_image': image,
                'detection_masks': detection_results,
                'corners': corners,
                'board': None,
                'error': None
            }
            
            # Extract board if corners found
            if corners is not None:
                try:
                    warped_board = self.extract_board_perspective(image, corners)
                    result['board'] = warped_board
                    result['success'] = True
                except Exception as e:
                    result['error'] = f'Perspective transformation failed: {str(e)}'
                    result['board'] = image
            else:
                result['error'] = 'No valid board rectangle detected'
                result['board'] = image
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': f'Detection failed: {str(e)}', 'board': image}

    def visualize_detection(self, result, save_path=None, figsize=(16, 12)):
        """
        Visualize the detection process and results
        
        Args:
            result: Result dictionary from detect_and_extract
            save_path: Optional path to save visualization
            figsize: Figure size for matplotlib
        """
        if not result.get('original_image') is not None:
            print("No image data to visualize")
            return
        
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        
        # Original image
        axes[0,0].imshow(result['original_image'])
        axes[0,0].set_title('Original Image')
        axes[0,0].axis('off')
        
        # Detection masks
        if 'detection_masks' in result:
            masks = result['detection_masks']
            
            axes[0,2].imshow(masks['grid_structure'], cmap='gray')
            axes[0,2].set_title('Grid Structure')
            axes[0,2].axis('off')
            
            axes[1,0].imshow(masks['content'], cmap='gray')
            axes[1,0].set_title('Content Detection')
            axes[1,0].axis('off')
            
            axes[1,1].imshow(masks['edges'], cmap='gray')
            axes[1,1].set_title('Edge Detection')
            axes[1,1].axis('off')
            
            axes[1,2].imshow(masks['combined_mask'], cmap='gray')
            axes[1,2].set_title('Combined Mask')
            axes[1,2].axis('off')
        
        # Detected corners visualization
        corner_vis = result['original_image'].copy()
        if result['corners'] is not None:
            cv2.drawContours(corner_vis, [result['corners']], -1, (255, 0, 0), 3)
            # Draw corner points
            for point in result['corners'].reshape(-1, 2):
                cv2.circle(corner_vis, tuple(point.astype(int)), 8, (0, 255, 0), -1)
        
        axes[2,0].imshow(corner_vis)
        axes[2,0].set_title('Detected Board Corners')
        axes[2,0].axis('off')
        
        # Final extracted board
        if result['success'] and result['board'] is not None:
            axes[2,1].imshow(result['board'])
            axes[2,1].set_title('Extracted Sudoku Board')
        else:
            axes[2,1].text(0.5, 0.5, f"Extraction Failed\\n{result.get('error', 'Unknown error')}", 
                          ha='center', va='center', transform=axes[2,1].transAxes, fontsize=12)
            axes[2,1].set_title('Extraction Result')
        axes[2,1].axis('off')
        
        # Remove empty subplot
        axes[2,2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


def demo_sudoku_detection():
    """
    Demonstration function to test the Sudoku detector
    """
    # Initialize detector
    detector = SudokuBoardDetector(output_size=450, debug=True)
    
    # Test directory
    test_dir = Path("/home/tomas/PYT/motustom/semestral/app/tests/images")
    
    if not test_dir.exists():
        print(f"Test directory {test_dir} not found")
        return
    
    # Test images
    test_images = ["image1.jpg", "image2.png", "image3.png", "image4.png"]
    
    print("=== Sudoku Board Detection Demo ===")
    print(f"Output board size: {detector.output_size}x{detector.output_size}")
    print()
    
    for img_name in test_images:
        img_path = test_dir / img_name
        
        if img_path.exists():
            print(f"Processing: {img_name}")
            
            # Detect and extract
            result = detector.detect_and_extract(str(img_path))
            
            if result['success']:
                print(f"  ✓ Successfully extracted Sudoku board")
                print(f"  Board shape: {result['board'].shape}")
            else:
                print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
            
            # Visualize results
            detector.visualize_detection(result)
            
        else:
            print(f"Image not found: {img_name}")
        
        print()
    
    print("Demo completed!")


if __name__ == "__main__":
    demo_sudoku_detection()
