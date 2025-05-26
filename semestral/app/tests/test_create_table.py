import sys
import os
# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Now you can import the modules
import cv2
from imutils import contours
import numpy as np
from src import create_table as ct
import pytesseract as tess
from paddleocr import PaddleOCR
import skimage
ocr = PaddleOCR(use_angle_cls=False, lang='en') # need to run only once to download and load model into memory
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
# to switch the language model in order.

def test_create_table():
    image = cv2.imread('/home/tomas/PYT/motustom/semestral/app/tests/images/image2.png')
    thresh = ct.preprocess_image(image)
    print(ct.create(image, thresh))
    img_path = '/home/tomas/PYT/motustom/semestral/app/tests/images/image2.png'
    result = ocr.predict(img_path)
    for res in result:
        res.save_to_img("output")
    img_path = '/home/tomas/PYT/motustom/semestral/app/tests/images/image1.jpg'
    result = ocr.predict(img_path)
    for res in result:
        res.save_to_img("output")
    image = cv2.imread('/home/tomas/PYT/motustom/semestral/app/tests/images/image3.png')
    thresh = ct.preprocess_image(image)
    result = ct.create(image, thresh)
    np.set_printoptions(threshold=sys.maxsize)
    print(result)
    img_path = '/home/tomas/PYT/motustom/semestral/app/tests/images/image3.png'
    result = ocr.predict(img_path)
    for res in result:
        res.save_to_img("output")
    img_path = '/home/tomas/PYT/motustom/semestral/app/tests/images/image4.png'
    result = ocr.predict(img_path)
    for res in result:
        res.save_to_img("output")

def test_preprocess_image():
    image = cv2.imread('/home/tomas/PYT/motustom/semestral/app/tests/images/image1.jpg')
    thresh = ct.preprocess_image(image)
    assert thresh is not None
    assert isinstance(thresh, np.ndarray)
    assert thresh.shape[0] == image.shape[0]
    assert thresh.shape[1] == image.shape[1]
    assert len(thresh.shape) == 2  # Check if the output is a grayscale image

def test_detect_digit():
    image = cv2.imread('/home/tomas/PYT/motustom/semestral/app/tests/images/digit1.png')
    thresh = ct.preprocess_image(image)
    digit = ct.detect_digit(thresh)
    assert digit is not None
    assert isinstance(digit, int)
    print(f"Detected digit: {digit}")
    image = cv2.imread('/home/tomas/PYT/motustom/semestral/app/tests/images/digit2.png')
    print("detected", tess.image_to_string(image))
    thresh = ct.preprocess_image(image)
    digit = ct.detect_digit(thresh)
    assert digit is not None
    assert isinstance(digit, int)
    print(f"Detected digit: {digit}")
    result = ocr.predict('/home/tomas/PYT/motustom/semestral/app/tests/images/digit2.png', use_doc_orientation_classify=False)
    for res in result:
        res.save_to_img("output")