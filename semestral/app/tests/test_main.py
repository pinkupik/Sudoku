import sys
import os
# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Now you can import the modules
from src import filter
from gui import kuskus
import kokos

def test_main():
    print ("Hello, World!")
    print (filter.add(1, 2))
    print (filter.subtract(2, 1))
    kuskus.randoms()
    print (kokos.kokosak(1, 2))
    
def test_filter():
    assert filter.add(1, 2) == 3
    assert filter.subtract(2, 1) == 1
    assert filter.multiply(2, 3) == 6
    assert filter.divide(6, 3) == 2
    assert filter.divide(6, 0) == "Cannot divide by zero"
    
def test_kuskus():
    assert kuskus.randoms() == 0
    assert filter.add(1, 2) == 3
    assert filter.subtract(2, 1) == 1
    assert filter.multiply(2, 3) == 6
    assert filter.divide(6, 3) == 2
    assert filter.divide(6, 0) == "Cannot divide by zero"