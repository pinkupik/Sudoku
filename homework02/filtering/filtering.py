# -*- coding: utf-8 -*-
"""Filtering module for image processing.
This module provides functions to apply various filters to images using
convolution operations. It supports both grayscale and RGB images and
implements different filtering techniques commonly used in image processing.
This module is part of the homework02 package and is intended for educational
purposes.
"""

import numpy as np


def apply_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply a filter to an image using convolution.
    
    This function applies a filter to an image using convolution. The filter is 
    applied to each pixel in the image, and the result is stored in a new image. 
    The function supports both grayscale and RGB images.

    Args:
        image (np.ndarray): The input image to which the filter will be applied.
        kernel (np.ndarray): The filter kernel to be used for convolution.

    Returns:
        np.ndarray: The filtered image after applying the convolution.
    """
    # Validate input dimensions
    assert image.ndim in [2, 3], "Image must be either grayscale (2D) or RGB (3D)"
    assert kernel.ndim == 2, "Kernel must be 2-dimensional"
    assert kernel.shape[0] == kernel.shape[1], "Kernel must be square"

    # For convolution, we keep the kernel as is (no flipping)
    # as the tests expect correlation behavior not convolution

    # Create output image with same shape as input
    filtered_image = np.zeros_like(image, dtype=np.int16)

    # Calculate padding size
    pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2

    # Process RGB image
    if image.ndim == 3:
        padded_image = np.pad(
            image.astype(np.int16),
            ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode='constant'
        )
        # Create a sliding window view into the padded image
        shape = (image.shape[0], image.shape[1], kernel.shape[0], kernel.shape[1], image.shape[2])
        strides = (
            padded_image.strides[0], # pylint: disable=unsubscriptable-object
            padded_image.strides[1], # pylint: disable=unsubscriptable-object
            padded_image.strides[0], # pylint: disable=unsubscriptable-object
            padded_image.strides[1], # pylint: disable=unsubscriptable-object
            padded_image.strides[2]  # pylint: disable=unsubscriptable-object
        )
        sub_matrices = np.lib.stride_tricks.as_strided(
            padded_image, shape=shape, strides=strides
        )
        # Perform the correlation using tensordot over the kernel dimensions
        filtered_image = np.tensordot(sub_matrices, kernel, axes=([2, 3], [0, 1]))
    # Process grayscale image
    else:
        padded_image = np.pad(
            image.astype(np.int16),
            ((pad_h, pad_h), (pad_w, pad_w)),
            mode='constant'
        )
        # Create sliding window view for grayscale image
        shape = (image.shape[0], image.shape[1], kernel.shape[0], kernel.shape[1])
        strides = (
            padded_image.strides[0], # pylint: disable=unsubscriptable-object
            padded_image.strides[1], # pylint: disable=unsubscriptable-object
            padded_image.strides[0], # pylint: disable=unsubscriptable-object
            padded_image.strides[1]  # pylint: disable=unsubscriptable-object
        )
        sub_matrices = np.lib.stride_tricks.as_strided(
            padded_image, shape=shape, strides=strides
        )
        filtered_image = np.tensordot(sub_matrices, kernel, axes=([2, 3], [0, 1]))

    # Clip values to valid range and convert back to original dtype
    return np.clip(filtered_image, 0, 255).astype(image.dtype)
