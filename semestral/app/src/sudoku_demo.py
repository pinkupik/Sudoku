#!/usr/bin/env python3
"""
Enhanced Sudoku Detection Demo
Demonstrates the improved detection capabilities for blue pen handwriting
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def demo_sudoku_detection():
    """Demonstrate the enhanced Sudoku detection with visual output"""
    try:
        from create_table import extract_sudoku_data_from_image
        import cv2
        import numpy as np
        
        print("🔍 Enhanced Sudoku Detection Demo")
        print("=" * 40)
        print()
        print("Features:")
        print("✅ Blue pen handwriting detection")
        print("✅ Multiple color space analysis (HSV, LAB, RGB)")
        print("✅ Enhanced grid detection with edge detection")
        print("✅ Improved digit recognition with multiple OCR configurations")
        print("✅ Robust contour detection with fallback options")
        print()
        
        # Test directory
        test_dir = "/home/tomas/PYT/motustom/semestral/app/tests/images"
        
        if not os.path.exists(test_dir):
            print(f"❌ Test directory not found: {test_dir}")
            return
        
        # Available test images
        available_images = []
        for filename in os.listdir(test_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                available_images.append(filename)
        
        print(f"📁 Found {len(available_images)} test images:")
        for i, img in enumerate(available_images, 1):
            print(f"   {i}. {img}")
        print()
        
        # Process each image
        for img_name in available_images:
            img_path = os.path.join(test_dir, img_name)
            
            print(f"🔄 Processing: {img_name}")
            print("-" * 30)
            
            try:
                # Load and show image info
                image = cv2.imread(img_path)
                if image is not None:
                    h, w = image.shape[:2]
                    print(f"📐 Image size: {w}x{h} pixels")
                    
                    # Process with enhanced detection
                    print("🧠 Running enhanced detection...")
                    sudoku_matrix = extract_sudoku_data_from_image(img_path, show_steps=False)
                    
                    if sudoku_matrix is not None:
                        print("✅ Grid detection successful!")
                        
                        # Analyze results
                        detected_digits = np.count_nonzero(sudoku_matrix)
                        total_cells = 81
                        empty_cells = total_cells - detected_digits
                        
                        print(f"📊 Detection Statistics:")
                        print(f"   • Detected digits: {detected_digits}/81 ({(detected_digits/81)*100:.1f}%)")
                        print(f"   • Empty cells: {empty_cells}")
                        
                        # Show the detected Sudoku grid
                        print(f"🔢 Detected Sudoku Grid:")
                        print("   " + "─" * 25)
                        for i in range(9):
                            row_str = "   │ "
                            for j in range(9):
                                digit = sudoku_matrix[i, j]
                                if digit == 0:
                                    row_str += "· "
                                else:
                                    row_str += f"{digit} "
                                if j == 2 or j == 5:
                                    row_str += "│ "
                            row_str += "│"
                            print(row_str)
                            if i == 2 or i == 5:
                                print("   ├" + "─" * 7 + "┼" + "─" * 7 + "┼" + "─" * 7 + "┤")
                        print("   " + "─" * 25)
                        
                        # Digit distribution
                        if detected_digits > 0:
                            digit_counts = {}
                            for i in range(1, 10):
                                count = np.sum(sudoku_matrix == i)
                                if count > 0:
                                    digit_counts[i] = count
                            
                            if digit_counts:
                                print(f"📈 Digit distribution:")
                                for digit, count in sorted(digit_counts.items()):
                                    print(f"   • Digit {digit}: {count} times")
                    else:
                        print("❌ Failed to detect Sudoku grid")
                        print("   Possible issues:")
                        print("   • Grid not clearly visible")
                        print("   • Image too blurry or low contrast")
                        print("   • Unexpected image format")
                
                else:
                    print(f"❌ Could not load image: {img_name}")
                    
            except Exception as e:
                print(f"❌ Error processing {img_name}: {e}")
            
            print()
            print("=" * 50)
            print()
        
        print("🏁 Demo completed!")
        print()
        print("💡 Tips for better detection:")
        print("   • Ensure good lighting and contrast")
        print("   • Use clear, legible handwriting")
        print("   • Keep the camera steady to avoid blur")
        print("   • Make sure the entire Sudoku grid is visible")
        print("   • Blue pen often works better than pencil")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required dependencies:")
        print("   pip install opencv-python numpy pytesseract")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def save_detection_results():
    """Save detection results to files for analysis"""
    try:
        from create_table import extract_sudoku_data_from_image
        import cv2
        import numpy as np
        
        print("💾 Saving detection results...")
        
        test_dir = "/home/tomas/PYT/motustom/semestral/app/tests/images"
        output_dir = "/home/tomas/PYT/motustom/semestral/app/output"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process test images
        for filename in os.listdir(test_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(test_dir, filename)
                
                try:
                    sudoku_matrix = extract_sudoku_data_from_image(img_path, show_steps=False)
                    
                    if sudoku_matrix is not None:
                        # Save as text file
                        base_name = os.path.splitext(filename)[0]
                        output_file = os.path.join(output_dir, f"{base_name}_sudoku_detected.txt")
                        
                        with open(output_file, 'w') as f:
                            f.write(f"Sudoku detection results for: {filename}\\n")
                            f.write(f"Detected digits: {np.count_nonzero(sudoku_matrix)}/81\\n")
                            f.write("\\nGrid:\\n")
                            
                            for i in range(9):
                                row = ""
                                for j in range(9):
                                    digit = sudoku_matrix[i, j]
                                    row += str(digit) if digit != 0 else "0"
                                    if j < 8:
                                        row += " "
                                f.write(row + "\\n")
                        
                        print(f"✅ Saved: {output_file}")
                    
                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")
        
        print("💾 Results saved to output directory")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--save":
        save_detection_results()
    else:
        demo_sudoku_detection()
