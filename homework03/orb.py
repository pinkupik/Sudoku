# @generated "[partially]" github-copilot-autocomplete:

# @generated "[all]" github-copilot-gpt-4.1:
"""
Implements a modified FAST (Features from Accelerated Segment Test) corner detector.

This module provides functions to detect keypoints in an image using a variation
of the FAST algorithm combined with Harris corner response for filtering. It includes:
- Image pyramid creation (`create_pyramid`) for multi-scale detection.
- FAST corner detection tests (`get_first_test_mask`, `get_second_test_mask`).
- Calculation of FAST scores (`calculate_kp_scores`).
- Initial keypoint detection per pyramid level (`detect_keypoints`).
- Calculation of image derivatives (`get_x_derivative`, `get_y_derivative`).
- Calculation of Harris corner response (`get_harris_response`).
- Filtering of keypoints based on Harris response (`filter_keypoints`).
- The main `fast` function orchestrating the detection process across pyramid levels,
    distributing the desired number of features across levels and applying filtering.

Constants related to the FAST algorithm's Bresenham circle are also defined.
Note: Some functions (`get_first_test_mask`, `get_second_test_mask`) might be
simplified or marked as not necessary to implement based on external requirements (README).
"""
from typing import List
from typing import Tuple
import math
import cv2
import scipy
import numpy as np
from utils import apply_gaussian_2d


FAST_CIRCLE_RADIUS = 3
FAST_ROW_OFFSETS = [-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3]
FAST_COL_OFFSETS = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1]
FAST_FIRST_TEST_INDICES = [0, 4, 8, 12]
FAST_FIRST_TEST_THRESHOLD = 3
FAST_SECOND_TEST_THRESHOLD = 12

# @generated "[partially]" github-copilot-gemini-2.5-pro:
def create_pyramid(
    img: np.ndarray, n_pyr_layers: int, downscale_factor: float = 1.2
) -> List[np.ndarray]:
    """
    Creates multi-scale image pyramid.

    Parameters
    ----------
    img : np.ndarray
        Gray-scaled input image.
    n_pyr_layers : int
        Number of layers in the pyramid.
    downscale_factor: float
        Downscaling performed between successive pyramid layers.

    Returns
    -------
    pyr : List[np.ndarray]
        Pyramid of scaled images.
    """
    pyr = []
    current_img = img.copy()
    pyr.append(current_img)
    for _ in range(1, n_pyr_layers):
        new_height = math.ceil(current_img.shape[0] / downscale_factor)
        new_width = math.ceil(current_img.shape[1] / downscale_factor)
        # ;size expects (width, height)
        downscaled_img = cv2.resize(current_img, (new_width, new_height))
        pyr.append(downscaled_img)
        current_img = downscaled_img
    return pyr


# not necessary to implement, see README
# @generated "[partially]" github-copilot-gemini-2.5-pro:
def get_first_test_mask(
    img_level: np.ndarray, threshold: int, border: int
) -> np.ndarray:
    """
    Returns the mask from the first FAST test (FAST_FIRST_TEST_INDICES).

    Parameters
    ----------
    img_level : np.ndarray
        Image at the given level of the image pyramid.
    threshold : int
        Intensity by which tested pixel should differ from the pixels on its Bresenham circle.
    border: int
        Number of rows/columns at the image border where no keypoints should be reported.

    Returns
    -------
    mask : np.ndarray
        Boolean mask with True values at pixels which pass the first FAST test.
    """
    img_level = img_level.astype(int)
    mask = np.zeros_like(img_level, dtype=bool)
    # Ensure border is large enough for FAST circle
    border = max(border, FAST_CIRCLE_RADIUS)
    for row in range(border, img_level.shape[0] - border):
        for col in range(border, img_level.shape[1] - border):
            center_pixel = img_level[row, col]
            first_test_pixels = [img_level[row + FAST_ROW_OFFSETS[i],
                    col + FAST_COL_OFFSETS[i]] for i in FAST_FIRST_TEST_INDICES]
            brighter_count = sum(
                1 for p in first_test_pixels if p > center_pixel + threshold)
            darker_count = sum(
                1 for p in first_test_pixels if p < center_pixel - threshold)
            if brighter_count >= FAST_FIRST_TEST_THRESHOLD or \
            darker_count >= FAST_FIRST_TEST_THRESHOLD:
                mask[row, col] = True
    return mask


# not necessary to implement, see README
# @generated "[partially]" github-copilot-gemini-2.5-pro:
def get_second_test_mask(
    img_level: np.ndarray,
    first_test_mask: np.ndarray,
    threshold: int,
) -> np.ndarray:
    """
    Returns the mask from the second FAST test (FAST_FIRST_TEST_INDICES).
    HINT: test only at those points which already passed the first test (first_test_mask).
    (Simplified logic: count total brighter/darker, not contiguous)

    Parameters
    ----------
    img_level : np.ndarray
        Image at the given level of the image pyramid.
    first_test_mask: np.ndarray
        Boolean mask for the first test, which was created by get_first_test_mask().
    threshold : int
        Intensity by which tested pixel should differ from the pixels on its Bresenham circle.

    Returns
    -------
    mask : np.ndarray
        Boolean mask with True values at pixels which pass the second FAST test.
    """
    img_level = img_level.astype(int)
    mask = np.zeros_like(img_level, dtype=bool)
    # Ensure loops don't go out of bounds for FAST circle offsets
    rows, cols = np.where(first_test_mask)
    for row, col in zip(rows, cols):
        # Check if the current point is far enough from the border to apply offsets
        if (FAST_CIRCLE_RADIUS <= row < img_level.shape[0] - FAST_CIRCLE_RADIUS and
                FAST_CIRCLE_RADIUS <= col < img_level.shape[1] - FAST_CIRCLE_RADIUS):
            center_pixel = img_level[row, col]
            # Use list comprehension for efficiency
            circle_pixels = [img_level[row + dr, col + dc]
                             for dr, dc in zip(FAST_ROW_OFFSETS, FAST_COL_OFFSETS)]

            # --- Simplified Check ---
            # Count total number of brighter and darker pixels in the whole circle
            brighter_count = sum(
                1 for p in circle_pixels if p > center_pixel + threshold)
            darker_count = sum(1 for p in circle_pixels if p <
                               center_pixel - threshold)

            # Check if the count meets the second threshold
            if brighter_count >= FAST_SECOND_TEST_THRESHOLD or darker_count >= FAST_SECOND_TEST_THRESHOLD:
                mask[row, col] = True
            # --- End Simplified Check ---
    return mask

# @generated "[partially]" github-copilot-gemini-2.5-pro:
def calculate_kp_scores(
    img_level: np.ndarray,
    keypoints: List[Tuple[int, int]],
) -> List[int]:
    """
    Calculates FAST score for initial keypoints.

    Parameters
    ----------
    img_level : np.ndarray
        Image at the given level of the image pyramid.
    keypoints: List[Tuple[int, int]]
        Tentative keypoints detected by FAST algorithm.

    Returns
    -------
    scores : List[int]
        Scores for the tentative keypoints.
    """
    img_level = img_level.astype(int)
    scores = []
    for row, col in keypoints:
        center_pixel = img_level[row, col]
        circle_pixels = [
            img_level[row + FAST_ROW_OFFSETS[i], col + FAST_COL_OFFSETS[i]]
            for i in range(len(FAST_ROW_OFFSETS))
        ]

        max_of_min_abs_diff = 0
        # Iterate through all 16 possible starting points for the 9 consecutive pixels
        for start_idx in range(len(FAST_ROW_OFFSETS)):
            min_abs_diff_group = float('inf')
            # Consider 9 consecutive pixels starting from start_idx
            for i in range(9):
                current_idx = (start_idx + i) % len(FAST_ROW_OFFSETS)
                abs_diff = abs(center_pixel - circle_pixels[current_idx])
                min_abs_diff_group = min(min_abs_diff_group, abs_diff)

            # Update the maximum of the minimums found so far
            max_of_min_abs_diff = max(max_of_min_abs_diff, min_abs_diff_group)

        scores.append(int(max_of_min_abs_diff))
    return scores

# @generated "[partially]" github-copilot-gemini-2.5-pro:
def detect_keypoints(
    img_level: np.ndarray,
    threshold: int,
    border: int = 0,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Creates the initial keypoints list.

    Scans the image at the given pyramid level and detects the unfiltered FAST keypoints,
    which are upscaled according to the current level index.

    Parameters
    ----------
    img_level : np.ndarray
        Image at the given level of the image pyramid.
    threshold : int
        Intensity by which tested pixel should differ from the pixels on its Bresenham circle.
    border: int
        Number of rows/columns at the image border where no keypoints should be reported.

    Returns
    -------
    keypoints : List[Tuple[int, int]]
        Initial FAST keypoints as tuples of (row_idx, col_idx).
    scores: List[int]
        Corresponding scores calculate with calculate_kp_scores().
    """
    border = max(border, FAST_CIRCLE_RADIUS)
    # Keep the optimized mask generation and keypoint creation
    first_test_mask = get_first_test_mask(img_level, threshold, border)
    # Use the simplified second test mask function
    second_test_mask = get_second_test_mask(
        img_level, first_test_mask, threshold)

    # Find coordinates where the second test passed
    rows, cols = np.where(second_test_mask)
    keypoints = list(zip(rows, cols))  # Directly create list of tuples

    # Calculate scores only for the keypoints that passed the second test
    scores = calculate_kp_scores(img_level, keypoints)

    # No redundant filtering needed here

    return keypoints, scores

# @generated "[partially]" github-copilot-gemini-2.5-pro:
def get_x_derivative(img: np.ndarray) -> np.ndarray:
    """
    Calculates x-derivative by applying separable Sobel filter.
    HINT: np.pad()

    Parameters
    ----------
    img : np.ndarray
        Gray-scaled input image.

    Returns
    -------
    result : np.ndarray
        X-derivative of the input image.
    """
    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    result = scipy.signal.convolve2d(img, sobel_x, mode='same', boundary='wrap')
    result = result[1:-1, 1:-1]
    result = np.pad(result, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    return result

# @generated "[partially]" github-copilot-gemini-2.5-pro:
def get_y_derivative(img: np.ndarray) -> np.ndarray:
    """
    Calculates y-derivative by applying separable Sobel filter.
    HINT: np.pad()

    Parameters
    ----------
    img : np.ndarray
        Gray-scaled input image.

    Returns
    -------
    result : np.ndarray
        Y-derivative of the input image.
    """
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    result = scipy.signal.convolve2d(img, sobel_y, mode='same', boundary='wrap')
    result = result[1:-1, 1:-1]
    result = np.pad(result, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    return result

# @generated "[partially]" github-copilot-gemini-2.5-pro:
def get_harris_response(img: np.ndarray) -> np.ndarray:
    """
    Calculates the Harris response.

    Calculates ixx, ixy and iyy from x and y-derivatives with Gaussian
    windowing (apply_gaussian_2d(data=..., sigma=1.0) from the utils.py). Then, uses the
    computed matrices to calculate the determinant and trace of the second-
    moment matrix. From it, calculates the final Harris response.

    Parameters
    ----------
    img : np.ndarray
        Gray-scaled input image.

    Returns
    -------
    harris_response : np.ndarray
        Harris response of the input image.
    """
    ix = get_x_derivative(img) / 255
    iy = get_y_derivative(img) / 255

    sxx = apply_gaussian_2d(data=ix * ix, sigma=1.0)
    syy = apply_gaussian_2d(data=iy * iy, sigma=1.0)
    sxy = apply_gaussian_2d(data=ix * iy, sigma=1.0)

    # compute determinant and trace elementwise from the smoothed second‐moment matrices
    det = sxx * syy - sxy * sxy
    trace = sxx + syy
    k = 0.05

    harris_response = det - (k * (trace ** 2))
    return harris_response

# @generated "[partially]" github-copilot-gemini-2.5-pro:
def filter_keypoints(
    img: np.ndarray, keypoints: List[Tuple[int, int]], n_max_level: int
) -> List[Tuple[int, int]]:
    """
    Filters keypoints by Harris response.

    Iterates the detected keypoints for the given level. Sorts those keypoints
    by their Harris response in the descending order. Then, takes only the
    n_max_level top keypoints.

     Parameters
    ----------
    img : np.ndarray
        Gray-scaled input image.
    keypoints : List[Tuple[int, int]]
        Initial FAST keypoints.
    n_max_level : int
        Maximal number of keypoints for a single pyramid level.

    Returns
    -------
    filtered_keypoints : List[Tuple[int, int]]
        Filtered FAST keypoints.
    """
    if not keypoints:
        return []

    harris_response = get_harris_response(img)

    # Get Harris scores for each keypoint
    keypoint_scores = []
    for r, c in keypoints:
        # Ensure keypoint coordinates are within bounds of harris_response
        if 0 <= r < harris_response.shape[0] and 0 <= c < harris_response.shape[1]:
            score = harris_response[r, c]
            keypoint_scores.append((score, (r, c)))

    # Sort keypoints by Harris score in descending order
    keypoint_scores.sort(key=lambda x: x[0], reverse=True)

    # Select the top n_max_level keypoints
    filtered_keypoints = [kp for score, kp in keypoint_scores[:n_max_level]]

    return filtered_keypoints


# def fast(
#     img: np.ndarray,
#     threshold: int = 20,
#     n_pyr_levels: int = 8,
#     downscale_factor: float = 1.2,
#     n_max_features: int = 500,
#     border: int = 0,
# ) -> List[List[Tuple[int, int]]]:
#     """
#     Applies the modified FAST detector.

#     Parameters
#     ----------
#     img : np.ndarray
#         Gray-scaled input image.
#     threshold: int
#         Absolute intensity threshold for FAST detector.
#     n_pyr_levels : int
#         Number of layers in the image pyramid.
#     downscale_factor: float
#         Downscaling performed between successive pyramid layers.
#     n_max_features : int
#         Total maximal number of keypoints.
#     """
#     pyr = create_pyramid(img, n_pyr_levels, downscale_factor)
#     keypoints_pyr = []
#     # Adapt Nmax for each level
#     factor = 1.0 / downscale_factor
#     n_max_level, n_sum_levels = [], 0
#     n_per_level = n_max_features * (1 - factor) / (1 - factor**n_pyr_levels)
#     for level in range(n_pyr_levels):
#         n_max_level.append(int(n_per_level))
#         n_sum_levels += n_max_level[-1]
#         n_per_level *= factor
#     n_max_level[-1] = max(n_max_features - n_sum_levels, 0)
#     for level, img_level in enumerate(pyr):
#         keypoints, scores = detect_keypoints(img_level, threshold, border=border)
#         idxs = np.argsort(scores)[::-1]
#         keypoints = np.asarray(keypoints)[idxs][: 2 * n_max_level[level]].tolist()
#         keypoints = filter_keypoints(img_level, keypoints, n_max_level[level])
#         upscale_factor = downscale_factor**level
#         keypoints = [
#             (int(x * upscale_factor), int(y * upscale_factor)) for (x, y) in keypoints
#         ]
#         keypoints_pyr.append(keypoints)
#     return keypoints_pyr
