#!/usr/bin/env python3
"""
Comprehensive Environment Verification Script for Sudoku Detection System
Checks all dependencies and their versions for compatibility
"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Python Version Check")
    print(f"   Python: {sys.version}")
    
    if sys.version_info < (3, 8):
        print("   ❌ Python 3.8+ required")
        return False
    elif sys.version_info >= (3, 12):
        print("   ✅ Python version excellent (3.12+)")
    else:
        print("   ✅ Python version compatible")
    return True

def check_dependency(module_name, package_name, min_version=None):
    """Check if a dependency is available and optionally verify version"""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        
        status = "✅"
        note = ""
        
        if min_version and version != 'unknown':
            try:
                from packaging import version as pkg_version
                if pkg_version.parse(version) < pkg_version.parse(min_version):
                    status = "⚠️"
                    note = f" (min: {min_version})"
            except:
                pass
        
        print(f"   {status} {package_name}: {version}{note}")
        return True
        
    except ImportError as e:
        print(f"   ❌ {package_name}: Not installed ({e})")
        return False

def check_tesseract():
    """Check Tesseract OCR installation"""
    print("\n🔍 Tesseract OCR Check")
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"   ✅ {version_line}")
            return True
        else:
            print("   ❌ Tesseract command failed")
            return False
    except subprocess.TimeoutExpired:
        print("   ❌ Tesseract command timed out")
        return False
    except FileNotFoundError:
        print("   ❌ Tesseract not found in PATH")
        print("   💡 Install: sudo apt-get install tesseract-ocr (Linux)")
        print("   💡 Install: brew install tesseract (macOS)")
        print("   💡 Download: https://github.com/UB-Mannheim/tesseract/wiki (Windows)")
        return False

def test_sudoku_detection():
    """Test basic Sudoku detection functionality"""
    print("\n🎯 Sudoku Detection Test")
    try:
        import cv2
        import numpy as np
        
        # Create a simple test image
        test_image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.rectangle(test_image, (50, 50), (350, 350), (0, 0, 0), 2)
        
        # Test basic OpenCV operations
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            print("   ✅ Basic image processing works")
            return True
        else:
            print("   ⚠️ Basic image processing issue")
            return False
            
    except Exception as e:
        print(f"   ❌ Sudoku detection test failed: {e}")
        return False

def check_project_files():
    """Check if project files are accessible"""
    print("\n📁 Project Files Check")
    
    project_root = Path("/home/tomas/PYT/motustom/semestral")
    required_files = [
        "environment.yml",
        "app/src/create_table.py",
        "app/src/sudoku_detector.py",
        "app/tests/images"
    ]
    
    all_found = True
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - Not found")
            all_found = False
    
    return all_found

def main():
    """Main verification function"""
    print("🔍 Enhanced Sudoku Detection Environment Verification")
    print("=" * 60)
    
    checks = []
    
    # Python version
    checks.append(check_python_version())
    
    # Core dependencies
    print("\n📦 Core Dependencies")
    dependencies = [
        ('cv2', 'opencv-python', '4.6.0'),
        ('numpy', 'numpy', '1.21.0'),
        ('matplotlib', 'matplotlib', '3.5.0'),
        ('skimage', 'scikit-image', '0.19.0'),
        ('pytesseract', 'pytesseract', '0.3.8'),
        ('imutils', 'imutils', '0.5.0'),
    ]
    
    for module, package, min_ver in dependencies:
        checks.append(check_dependency(module, package, min_ver))
    
    # Optional dependencies
    print("\n📦 Optional Dependencies")
    optional_deps = [
        ('paddleocr', 'paddleocr'),
        ('jupyter', 'jupyter'),
        ('pandas', 'pandas'),
    ]
    
    for module, package in optional_deps:
        check_dependency(module, package)  # Don't fail on optional deps
    
    # External tools
    checks.append(check_tesseract())
    
    # Functional tests
    checks.append(test_sudoku_detection())
    checks.append(check_project_files())
    
    # Summary
    print("\n" + "=" * 60)
    if all(checks):
        print("🎉 Environment verification PASSED!")
        print("✅ All required dependencies are installed and working")
        print("✅ Sudoku detection system is ready to use")
        print("\n💡 Next steps:")
        print("   • Run: python app/src/sudoku_demo.py")
        print("   • Or: python app/src/test_enhanced_detection.py")
        return True
    else:
        failed_checks = sum(1 for check in checks if not check)
        print(f"⚠️ Environment verification completed with {failed_checks} issues")
        print("❌ Please fix the issues above before running the system")
        print("\n💡 For help, see:")
        print("   • README_ENVIRONMENT.md")
        print("   • environment.yml")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
