"""
Improved Sudoku Board Detection System
Handles various image types including handwritten sudoku with blue pen
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from pathlib import Path


class SudokuDetector:
    """Enhanced Sudoku board detection with multiple detection methods"""
    
    def __init__(self, output_size=450):
        self.output_size = output_size
    
    def preprocess_image(self, image):
        """Apply preprocessing to enhance detection"""
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Already RGB
            pass
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        return image
    
    def detect_with_edges(self, gray):
        """Edge-based detection for clear grid lines"""
        # Multiple edge detection approaches
        edges1 = cv2.Canny(gray, 50, 150, apertureSize=3)
        edges2 = cv2.Canny(gray, 30, 100, apertureSize=3)
        
        # Combine edges
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Morphological operations to connect lines
        kernel_close = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
        
        # Dilate to strengthen lines
        kernel_dilate = np.ones((2,2), np.uint8)
        edges = cv2.dilate(edges, kernel_dilate, iterations=1)
        
        return edges
    
    def detect_with_adaptive_threshold(self, gray):
        """Adaptive thresholding for handwritten content"""
        # Multiple block sizes for different pen thicknesses
        thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY, 15, 4)
        
        # Combine and invert
        combined = cv2.bitwise_and(thresh1, thresh2)
        return cv2.bitwise_not(combined)
    
    def detect_blue_pen(self, image):
        """Special detection for blue pen writing"""
        if len(image.shape) != 3:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Method 1: Blue channel isolation
        blue_channel = image[:,:,2]
        blue_thresh = cv2.threshold(blue_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Method 2: HSV blue detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # Define range for blue colors
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Method 3: LAB color space for better blue detection
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        # In LAB, 'b' channel represents blue-yellow
        b_channel = lab[:,:,2]
        # Blue has higher b values (>128)
        lab_blue = cv2.threshold(b_channel, 128, 255, cv2.THRESH_BINARY)[1]
        
        # Combine all blue detection methods
        combined_blue = cv2.bitwise_or(blue_thresh, blue_mask)
        combined_blue = cv2.bitwise_or(combined_blue, lab_blue)
        
        return combined_blue
    
    def detect_with_contours(self, gray):
        """Contour-based detection"""
        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Find contours at different threshold levels
        contour_mask = np.zeros_like(gray)
        
        for thresh_val in [127, 100, 150]:
            _, binary = cv2.threshold(filtered, thresh_val, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw significant contours
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Filter small contours
                    cv2.drawContours(contour_mask, [contour], -1, 255, 2)
        
        return contour_mask
    
    def combine_detection_methods(self, image):
        """Combine multiple detection methods for robust detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Get all detection results
        edges = self.detect_with_edges(gray)
        adaptive = self.detect_with_adaptive_threshold(gray)
        blue_pen = self.detect_blue_pen(image)
        contours = self.detect_with_contours(gray)
        
        # Weighted combination - give more weight to certain methods
        combined = np.zeros_like(gray, dtype=np.float32)
        combined += edges.astype(np.float32) * 0.4  # Edges are reliable for grid detection
        combined += adaptive.astype(np.float32) * 0.3  # Good for handwritten content
        combined += blue_pen.astype(np.float32) * 0.2  # Specific for blue pen
        combined += contours.astype(np.float32) * 0.1  # Additional structural info
        
        # Normalize and threshold
        combined = np.clip(combined, 0, 255).astype(np.uint8)
        final_mask = cv2.threshold(combined, 50, 255, cv2.THRESH_BINARY)[1]
        
        # Clean up the mask
        kernel = np.ones((5,5), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        
        return final_mask, edges, adaptive, blue_pen, contours
    
    def find_best_rectangle(self, mask, min_area_ratio=0.1, max_area_ratio=0.9):
        """Find the best rectangular contour representing the Sudoku board"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        image_area = mask.shape[0] * mask.shape[1]
        min_area = image_area * min_area_ratio
        max_area = image_area * max_area_ratio
        
        best_candidates = []
        
        # Sort contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < min_area or area > max_area:
                continue
            
            # Try different approximation levels
            for epsilon_factor in [0.01, 0.02, 0.03, 0.05]:
                epsilon = epsilon_factor * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:
                    # Check if it's roughly square
                    rect = cv2.minAreaRect(contour)
                    width, height = rect[1]
                    aspect_ratio = max(width, height) / min(width, height)
                    
                    if aspect_ratio < 2.0:  # Roughly square
                        best_candidates.append({
                            'points': approx,
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'score': area / (aspect_ratio + 0.1)  # Favor larger, more square shapes
                        })
                        break
        
        if best_candidates:
            # Return the best candidate
            best = max(best_candidates, key=lambda x: x['score'])
            return best['points']
        
        # Fallback: use bounding rectangle of largest contour
        if contours:
            largest_contour = contours[0]
            if cv2.contourArea(largest_contour) > min_area:
                x, y, w, h = cv2.boundingRect(largest_contour)
                return np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)
        
        return None
    
    def order_points(self, pts):
        """Order points in clockwise order: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype="float32")
        pts = pts.reshape(4, 2)
        
        # Sum and difference to find corners
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        
        return rect
    
    def warp_perspective(self, image, corners):
        """Apply perspective transform to get top-down view"""
        ordered_corners = self.order_points(corners)
        
        # Destination points for square output
        dst = np.array([
            [0, 0],
            [self.output_size-1, 0],
            [self.output_size-1, self.output_size-1],
            [0, self.output_size-1]
        ], dtype="float32")
        
        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(ordered_corners, dst)
        warped = cv2.warpPerspective(image, M, (self.output_size, self.output_size))
        
        return warped
    
    def detect_and_extract(self, image):
        """Main method to detect and extract Sudoku board"""
        # Preprocess
        image = self.preprocess_image(image)
        
        # Detect board using combined methods
        combined_mask, edges, adaptive, blue_pen, contours = self.combine_detection_methods(image)
        
        # Find the best rectangle
        sudoku_corners = self.find_best_rectangle(combined_mask)
        
        results = {
            'original': image,
            'combined_mask': combined_mask,
            'edges': edges,
            'adaptive': adaptive,
            'blue_pen': blue_pen,
            'contours': contours,
            'corners': sudoku_corners,
            'warped': None,
            'success': False
        }
        
        if sudoku_corners is not None:
            try:
                warped_board = self.warp_perspective(image, sudoku_corners)
                results['warped'] = warped_board
                results['success'] = True
            except Exception as e:
                print(f"Warning: Could not warp perspective: {e}")
        
        return results
    
    def visualize_results(self, results, save_path=None):
        """Visualize detection results"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        # Original image
        axes[0,0].imshow(results['original'])
        axes[0,0].set_title("Original Image")
        axes[0,0].axis('off')
        
        # Detection methods
        axes[0,1].imshow(results['edges'], cmap='gray')
        axes[0,1].set_title("Edge Detection")
        axes[0,1].axis('off')
        
        axes[0,2].imshow(results['adaptive'], cmap='gray')
        axes[0,2].set_title("Adaptive Threshold")
        axes[0,2].axis('off')
        
        axes[1,0].imshow(results['blue_pen'], cmap='gray')
        axes[1,0].set_title("Blue Pen Detection")
        axes[1,0].axis('off')
        
        axes[1,1].imshow(results['contours'], cmap='gray')
        axes[1,1].set_title("Contour Detection")
        axes[1,1].axis('off')
        
        axes[1,2].imshow(results['combined_mask'], cmap='gray')
        axes[1,2].set_title("Combined Mask")
        axes[1,2].axis('off')
        
        # Detected corners
        visualization = results['original'].copy()
        if results['corners'] is not None:
            cv2.drawContours(visualization, [results['corners']], -1, (255, 0, 0), 3)
            # Draw corner points
            for point in results['corners'].reshape(-1, 2):
                cv2.circle(visualization, tuple(point), 8, (0, 255, 0), -1)
        
        axes[2,0].imshow(visualization)
        axes[2,0].set_title("Detected Board")
        axes[2,0].axis('off')
        
        # Final result
        if results['success']:
            axes[2,1].imshow(results['warped'])
            axes[2,1].set_title("Extracted Sudoku Board")
        else:
            axes[2,1].imshow(results['original'])
            axes[2,1].set_title("Detection Failed")
        axes[2,1].axis('off')
        
        # Remove empty subplot
        axes[2,2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


def test_sudoku_detector():
    """Test the detector on available images"""
    detector = SudokuDetector(output_size=450)
    
    # Test images
    image_dir = Path("/home/tomas/PYT/motustom/semestral/app/tests/images")
    test_images = ["image1.jpg", "image2.png", "image3.png", "image4.png"]
    
    for img_name in test_images:
        img_path = image_dir / img_name
        if img_path.exists():
            print(f"\n=== Testing {img_name} ===")
            
            try:
                # Load image
                image = skimage.io.imread(str(img_path))
                
                # Detect Sudoku board
                results = detector.detect_and_extract(image)
                
                if results['success']:
                    print(f"✓ Successfully detected and extracted Sudoku board from {img_name}")
                else:
                    print(f"✗ Failed to detect Sudoku board in {img_name}")
                
                # Visualize results
                detector.visualize_results(results)
                
            except Exception as e:
                print(f"Error processing {img_name}: {e}")


if __name__ == "__main__":
    test_sudoku_detector()
