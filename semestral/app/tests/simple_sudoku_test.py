"""
Simple Sudoku Detection Test
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def simple_sudoku_detection(image_path):
    """Simple and robust Sudoku board detection"""
    print(f"Loading image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image")
        return None
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print(f"Image shape: {image_rgb.shape}")
    
    # Method 1: Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Method 2: Adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh_inv = cv2.bitwise_not(thresh)
    
    # Combine methods
    combined = cv2.bitwise_or(edges, thresh_inv)
    
    # Clean up
    kernel = np.ones((5,5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No contours found")
        return None
    
    # Sort by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Find largest rectangular contour
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:  # Skip small contours
            continue
            
        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            print(f"Found 4-sided contour with area: {area}")
            
            # Order points
            pts = approx.reshape(4, 2).astype(np.float32)
            
            # Create destination points
            side_length = 450
            dst = np.array([
                [0, 0],
                [side_length-1, 0],
                [side_length-1, side_length-1],
                [0, side_length-1]
            ], dtype=np.float32)
            
            # Calculate perspective transform
            M = cv2.getPerspectiveTransform(pts, dst)
            warped = cv2.warpPerspective(image_rgb, M, (side_length, side_length))
            
            # Show results
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 4, 1)
            plt.imshow(image_rgb)
            plt.title("Original")
            plt.axis('off')
            
            plt.subplot(1, 4, 2)
            plt.imshow(combined, cmap='gray')
            plt.title("Combined Mask")
            plt.axis('off')
            
            plt.subplot(1, 4, 3)
            # Draw contour on original
            vis = image_rgb.copy()
            cv2.drawContours(vis, [approx], -1, (255, 0, 0), 3)
            plt.imshow(vis)
            plt.title("Detected Board")
            plt.axis('off')
            
            plt.subplot(1, 4, 4)
            plt.imshow(warped)
            plt.title("Extracted Board")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            return warped
    
    print("No suitable rectangular contour found")
    return None


# Test the function
if __name__ == "__main__":
    import os
    
    image_dir = "/home/tomas/PYT/motustom/semestral/app/tests/images"
    
    print("Available images:")
    for f in os.listdir(image_dir):
        print(f"  {f}")
    
    # Test on image4.png
    image_path = os.path.join(image_dir, "image4.png")
    result = simple_sudoku_detection(image_path)
    
    if result is not None:
        print("Success! Sudoku board extracted.")
    else:
        print("Failed to extract Sudoku board.")
