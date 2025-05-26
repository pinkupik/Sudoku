import sys
import os
# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import sudscan as sscan
import numpy as np
import pytest

def test_scan_table_returns_correct_shape():
    """Test that scan_table returns a 9x9 numpy array"""
    test_dir = "/home/tomas/PYT/motustom/semestral/app/tests/images"
    img_path = os.path.join(test_dir, "image1.jpg")
    
    if os.path.exists(img_path):
        result = sscan.scan_table(img_path)
        assert result.shape == (9, 9), f"Expected shape (9, 9), got {result.shape}"
        assert isinstance(result, np.ndarray), "Result should be a numpy array"

def test_scan_table_returns_valid_values():
    """Test that scan_table returns only valid Sudoku values (0-9)"""
    test_dir = "/home/tomas/PYT/motustom/semestral/app/tests/images"
    img_path = os.path.join(test_dir, "image1.jpg")
    
    if os.path.exists(img_path):
        result = sscan.scan_table(img_path)
        assert np.all((result >= 0) & (result <= 9)), "All values should be between 0 and 9"
        assert result.dtype == int, "Result should contain integers"

def test_scan_table_image1():
    """Test scan_table with image1.jpg"""
    test_dir = "/home/tomas/PYT/motustom/semestral/app/tests/images"
    img_path = os.path.join(test_dir, "image1.jpg")
    
    if not os.path.exists(img_path):
        pytest.skip(f"Test image not found: {img_path}")
    
    result = sscan.scan_table(img_path)
    assert result is not None, "scan_table should not return None"
    assert result.shape == (9, 9), "Result should be 9x9 matrix"

def test_scan_table_image2():
    """Test scan_table with image2.png"""
    test_dir = "/home/tomas/PYT/motustom/semestral/app/tests/images"
    img_path = os.path.join(test_dir, "image2.png")
    
    if not os.path.exists(img_path):
        pytest.skip(f"Test image not found: {img_path}")
    
    result = sscan.scan_table(img_path)
    assert result is not None, "scan_table should not return None"
    assert result.shape == (9, 9), "Result should be 9x9 matrix"

def test_scan_table_image3():
    """Test scan_table with image3.png"""
    test_dir = "/home/tomas/PYT/motustom/semestral/app/tests/images"
    img_path = os.path.join(test_dir, "image3.png")
    
    if not os.path.exists(img_path):
        pytest.skip(f"Test image not found: {img_path}")
    
    result = sscan.scan_table(img_path)
    assert result is not None, "scan_table should not return None"
    assert result.shape == (9, 9), "Result should be 9x9 matrix"
    result_ref = np.array([[5, 3, 0, 0, 7, 0, 0, 0, 0],
                                                [6, 0, 0, 1, 9, 5, 0, 0, 0],
                                                [0, 9, 8, 0, 0, 0, 0, 6, 0],
                                                [8, 0, 0, 0, 6, 0, 0, 0, 3],
                                                [4, 0, 0, 8, 0, 3, 0, 0, 1],
                                                [7, 0, 0, 0, 2, 0, 0, 0, 6],
                                                [0, 6, 0, 0, 0, 0, 2, 8, 0],
                                                [0, 0, 0, 4, 1, 9, 0, 0, 5],
                                                [0, 0, 0, 0, 8, 0, 0, 7, 9]])
    assert np.array_equal(result, result_ref), "Result does not match reference"

def test_scan_table_image4():
    """Test scan_table with image4.png"""
    test_dir = "/home/tomas/PYT/motustom/semestral/app/tests/images"
    img_path = os.path.join(test_dir, "image4.png")
    
    if not os.path.exists(img_path):
        pytest.skip(f"Test image not found: {img_path}")
    
    result = sscan.scan_table(img_path)
    assert result is not None, "scan_table should not return None"
    assert result.shape == (9, 9), "Result should be 9x9 matrix"

def test_scan_table_nonexistent_file():
    """Test scan_table with non-existent file"""
    with pytest.raises(Exception):
        sscan.scan_table("nonexistent_file.jpg")