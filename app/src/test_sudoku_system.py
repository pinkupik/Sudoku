#!/usr/bin/env python3
"""
Simple test for Sudoku detection
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required imports work"""
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV: {e}")
        return False
        
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False
        
    try:
        import matplotlib.pyplot as plt
        print(f"✓ Matplotlib")
    except ImportError as e:
        print(f"✗ Matplotlib: {e}")
        return False
        
    return True

def test_basic_detection():
    """Test basic Sudoku detection functionality"""
    import cv2
    import numpy as np
    
    # Create a simple test image with a rectangle
    test_image = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Draw a blue rectangle (simulating a Sudoku board)
    cv2.rectangle(test_image, (50, 50), (350, 350), (255, 0, 0), 3)
    
    # Draw some grid lines
    for i in range(1, 9):
        x = 50 + i * (300 // 9)
        y = 50 + i * (300 // 9)
        cv2.line(test_image, (x, 50), (x, 350), (0, 0, 255), 1)
        cv2.line(test_image, (50, y), (350, y), (0, 0, 255), 1)
    
    print("✓ Created test image")
    
    # Basic blue detection
    blue_channel = test_image[:,:,2]
    blue_thresh = cv2.threshold(blue_channel, 200, 255, cv2.THRESH_BINARY)[1]
    
    # Find contours
    contours, _ = cv2.findContours(blue_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        print(f"✓ Found {len(contours)} contours")
        
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        print(f"✓ Largest contour area: {area}")
        
        # Approximate to rectangle
        epsilon = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        
        if len(approx) == 4:
            print("✓ Successfully detected rectangle")
            return True
        else:
            print(f"✗ Rectangle approximation failed: {len(approx)} points")
    else:
        print("✗ No contours found")
    
    return False

def test_image_loading():
    """Test loading actual test images"""
    test_dir = "/home/tomas/PYT/motustom/semestral/app/tests/images"
    
    if not os.path.exists(test_dir):
        print(f"✗ Test directory not found: {test_dir}")
        return False
    
    images = os.listdir(test_dir)
    print(f"✓ Found {len(images)} test images: {images}")
    
    # Try to load one image
    test_image_path = os.path.join(test_dir, "image4.png")
    if os.path.exists(test_image_path):
        try:
            import cv2
            image = cv2.imread(test_image_path)
            if image is not None:
                print(f"✓ Successfully loaded {test_image_path}")
                print(f"  Image shape: {image.shape}")
                return True
            else:
                print(f"✗ Failed to load {test_image_path}")
        except Exception as e:
            print(f"✗ Error loading image: {e}")
    else:
        print(f"✗ Test image not found: {test_image_path}")
    
    return False

def main():
    print("=== Sudoku Detection System Test ===")
    print()
    
    # Test imports
    print("1. Testing imports...")
    if not test_imports():
        print("✗ Import test failed")
        return
    print()
    
    # Test basic detection
    print("2. Testing basic detection...")
    if test_basic_detection():
        print("✓ Basic detection test passed")
    else:
        print("✗ Basic detection test failed")
    print()
    
    # Test image loading
    print("3. Testing image loading...")
    if test_image_loading():
        print("✓ Image loading test passed")
    else:
        print("✗ Image loading test failed")
    print()
    
    print("=== Test Complete ===")

if __name__ == "__main__":
    main()
