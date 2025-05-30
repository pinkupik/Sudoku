# Sudoku Board Detection - Environment Setup

This document describes how to set up the environment for the enhanced Sudoku board detection system that supports blue pen handwriting recognition.

## Quick Setup

### Using Conda (Recommended)

```bash
# Create the environment from the environment.yml file
conda env create -f environment.yml

# Activate the environment
conda activate sudoku-detection

# Verify the installation
python -c "import cv2, numpy, matplotlib, pytesseract, imutils, skimage; print('✅ All core dependencies installed successfully')"
```

### Alternative: Using pip

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0
pip install matplotlib>=3.7.0
pip install scikit-image>=0.21.0
pip install pytesseract>=0.3.10
pip install imutils>=0.5.4
pip install paddleocr>=2.7.0
pip install jupyter>=1.0.0

# Install additional dependencies
pip install -r requirements.txt  # if you create one from the environment.yml
```

## Platform-Specific Setup

### Windows
1. **Install Tesseract OCR:**
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Add Tesseract to your PATH or set `pytesseract.pytesseract.tesseract_cmd` in your code
   
2. **Verify installation:**
   ```cmd
   tesseract --version
   ```

### macOS
1. **Install Tesseract using Homebrew:**
   ```bash
   brew install tesseract
   ```

2. **For Apple Silicon Macs, you might need:**
   ```bash
   export TESSDATA_PREFIX=/opt/homebrew/share/tessdata
   ```

### Linux (Ubuntu/Debian)
1. **Install Tesseract:**
   ```bash
   sudo apt-get update
   sudo apt-get install tesseract-ocr tesseract-ocr-eng
   ```

2. **Install additional language packs if needed:**
   ```bash
   sudo apt-get install tesseract-ocr-all
   ```

## Verification

Run this verification script to ensure everything is working:

```python
#!/usr/bin/env python3
"""Environment verification script"""

def verify_environment():
    print("🔍 Verifying Sudoku Detection Environment...")
    print("=" * 50)
    
    # Test core imports
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib: {e}")
        return False
    
    try:
        import skimage
        print(f"✅ Scikit-image: {skimage.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-image: {e}")
        return False
    
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ Pytesseract: {version}")
    except Exception as e:
        print(f"❌ Pytesseract: {e}")
        return False
    
    try:
        import imutils
        print(f"✅ Imutils: Available")
    except ImportError as e:
        print(f"❌ Imutils: {e}")
        return False
    
    try:
        from paddleocr import PaddleOCR
        print(f"✅ PaddleOCR: Available")
    except ImportError as e:
        print(f"⚠️  PaddleOCR: {e} (Optional, but recommended)")
    
    print("\n🎉 Environment verification completed!")
    return True

if __name__ == "__main__":
    verify_environment()
```

## Dependencies Overview

### Core Dependencies (Required)
- **opencv-python**: Computer vision and image processing
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **scikit-image**: Advanced image processing
- **pytesseract**: OCR engine interface
- **imutils**: Computer vision utilities

### Enhanced Features (Recommended)
- **paddleocr**: Advanced OCR with better handwriting support
- **jupyter**: Interactive development and testing
- **pandas**: Data manipulation and analysis

### Development Tools (Optional)
- **pytest**: Testing framework
- **sphinx**: Documentation generation
- **albumentations**: Image augmentation for training

## Project Structure

```
semestral/
├── environment.yml          # Main environment file (corrected spelling)
├── enviroment.yml          # Original file (typo, but kept for compatibility)
├── app/
│   ├── src/
│   │   ├── create_table.py          # Enhanced detection with blue pen support
│   │   ├── sudoku_detector.py       # Comprehensive detection class
│   │   ├── sudoku_demo.py          # Interactive demonstration
│   │   ├── test_enhanced_detection.py  # Test suite
│   │   └── test_sudoku_system.py   # Basic system validation
│   └── tests/
│       ├── images/                 # Test images
│       ├── testorss.ipynb         # Jupyter notebook tests
│       └── improved_sudoku_detection.py  # Advanced detection methods
└── README_ENVIRONMENT.md          # This file
```

## Troubleshooting

### Common Issues

1. **Tesseract not found:**
   ```python
   # Add this to your Python code if Tesseract is not in PATH
   import pytesseract
   pytesseract.pytesseract.tesseract_cmd = r'/path/to/tesseract'
   ```

2. **OpenCV installation issues:**
   ```bash
   # Try installing with specific backend
   pip install opencv-python-headless  # For servers without GUI
   ```

3. **PaddleOCR installation issues:**
   ```bash
   # Install CPU version if GPU version fails
   pip install paddlepaddle-cpu
   pip install paddleocr
   ```

4. **Import errors:**
   ```bash
   # Ensure you're in the correct environment
   conda activate sudoku-detection
   # Or activate your virtual environment
   ```

### Performance Optimization

For better performance with large images:
- Install OpenCV with optimized builds
- Use `numba` for numerical computations
- Consider GPU acceleration for PaddleOCR

## Usage

After setting up the environment, you can run the enhanced Sudoku detection:

```python
from src.create_table import extract_sudoku_data_from_image

# Extract Sudoku from image with blue pen handwriting
sudoku_matrix = extract_sudoku_data_from_image('path/to/sudoku_image.jpg', show_steps=True)
print(sudoku_matrix)
```

Or use the comprehensive detector:

```python
from src.sudoku_detector import SudokuBoardDetector

detector = SudokuBoardDetector(debug=True)
result = detector.detect_and_extract('path/to/sudoku_image.jpg')

if result['success']:
    print("Sudoku board detected successfully!")
    extracted_board = result['board']
else:
    print(f"Detection failed: {result['error']}")
```

## Support

If you encounter any issues with the environment setup:
1. Check the verification script output
2. Ensure all platform-specific dependencies are installed
3. Verify your Python version is 3.8 or higher
4. Check that your conda/pip environment is activated

For project-specific issues, refer to the main project documentation.
