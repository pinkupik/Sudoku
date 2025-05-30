#!/usr/bin/env python3
"""
Test script for enhanced Sudoku detection with blue pen support
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_enhanced_sudoku_detection():
    """Test the enhanced Sudoku detection system"""
    try:
        from create_table import extract_sudoku_data_from_image
        import numpy as np
        
        print("=== Enhanced Sudoku Detection Test ===")
        print("Testing improved detection with blue pen support...")
        print()
        
        # Test directory
        test_dir = "/home/tomas/PYT/motustom/semestral/app/tests/images"
        
        if not os.path.exists(test_dir):
            print(f"❌ Test directory not found: {test_dir}")
            return False
        
        # Test images
        test_images = ["image1.jpg", "image2.png", "image3.png", "image4.png"]
        
        results = []
        
        for img_name in test_images:
            img_path = os.path.join(test_dir, img_name)
            
            if not os.path.exists(img_path):
                print(f"⚠️  Image not found: {img_name}")
                continue
                
            print(f"Processing: {img_name}")
            
            try:
                # Test the enhanced detection
                sudoku_matrix = extract_sudoku_data_from_image(img_path, show_steps=False)
                
                if sudoku_matrix is not None:
                    # Count detected digits
                    detected_digits = np.count_nonzero(sudoku_matrix)
                    total_cells = sudoku_matrix.size
                    detection_rate = (detected_digits / total_cells) * 100
                    
                    print(f"  ✅ Successfully processed")
                    print(f"  📊 Detected {detected_digits}/{total_cells} digits ({detection_rate:.1f}%)")
                    
                    if detected_digits > 0:
                        print(f"  🔢 Sample detected digits:")
                        # Show first few detected digits
                        found_digits = []
                        for i in range(9):
                            for j in range(9):
                                if sudoku_matrix[i, j] != 0:
                                    found_digits.append(f"({i},{j})={sudoku_matrix[i, j]}")
                                    if len(found_digits) >= 5:  # Show max 5 examples
                                        break
                            if len(found_digits) >= 5:
                                break
                        print(f"     {', '.join(found_digits)}")
                    
                    results.append((img_name, True, detected_digits, detection_rate))
                else:
                    print(f"  ❌ Failed to detect Sudoku grid")
                    results.append((img_name, False, 0, 0))
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results.append((img_name, False, 0, 0))
            
            print()
        
        # Summary
        print("=== Test Summary ===")
        successful_images = sum(1 for _, success, _, _ in results if success)
        total_images = len(results)
        
        print(f"Successfully processed: {successful_images}/{total_images} images")
        
        if results:
            avg_detection = sum(rate for _, success, _, rate in results if success) / max(1, successful_images)
            print(f"Average digit detection rate: {avg_detection:.1f}%")
            
            print("\nDetailed results:")
            for img_name, success, digits, rate in results:
                status = "✅" if success else "❌"
                print(f"  {status} {img_name}: {digits} digits ({rate:.1f}%)")
        
        return successful_images > 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required dependencies are installed:")
        print("  - opencv-python (cv2)")
        print("  - numpy")
        print("  - pytesseract")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are available"""
    print("=== Dependency Check ===")
    
    dependencies = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'), 
        ('pytesseract', 'pytesseract'),
    ]
    
    all_good = True
    
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - Install with: pip install {package_name}")
            all_good = False
    
    print()
    return all_good

def main():
    """Main test function"""
    print("Enhanced Sudoku Detection System - Test Suite")
    print("=" * 50)
    print()
    
    # Test dependencies first
    if not test_dependencies():
        print("❌ Please install missing dependencies before running tests.")
        return
    
    # Test the enhanced detection
    success = test_enhanced_sudoku_detection()
    
    if success:
        print("\n🎉 Tests completed successfully!")
        print("The enhanced Sudoku detection system is working.")
    else:
        print("\n⚠️  Tests completed with issues.")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main()
