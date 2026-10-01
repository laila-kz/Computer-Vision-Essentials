"""Global, adaptive, and statistical thresholding methods for image binarization.

Theoretical algorithms:
- Global Fixed Thresholding: $g(x,y) = 255$ if $f(x,y) \\ge T$ else $0$.
- Otsu's Optimal Thresholding: Maximizes inter-class variance $\\sigma_B^2(T) = \\omega_0(T)\\omega_1(T)[\\mu_0(T) - \\mu_1(T)]^2$.
- Adaptive Local Thresholding (Mean and Gaussian): $T(x,y) = \\mu_{\\mathrm{local}}(x,y) - C$.
- Triangle Thresholding: Geometric histogram peak-to-corner binarization.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from ..utils.conversions import ensure_uint8, to_gray


def manual_threshold(
    image: np.ndarray,
    threshold_value: int = 128,
    inverse: bool = False,
) -> np.ndarray:
    """Apply global hard thresholding with a specified scalar value.

    Args:
        image: Input grayscale or color image.
        threshold_value: Pixel cutoff intensity $T \\in [0, 255]$.
        inverse: If True, pixels above $T$ become 0 (black) and below become 255 (white).

    Returns:
        np.ndarray: Binary mask (0 and 255) in uint8.
    """
    gray = to_gray(image)
    flag = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    _, binary = cv2.threshold(ensure_uint8(gray), threshold_value, 255, flag)
    return binary


def otsu_threshold(image: np.ndarray, inverse: bool = False) -> Tuple[float, np.ndarray]:
    """Calculate optimal bimodal threshold using Otsu's method.

    Otsu's algorithm exhaustively searches for threshold $T^*$ that maximizes between-class variance:
    $$\\sigma_B^2(T) = \\omega_0(T) \\omega_1(T) [\\mu_0(T) - \\mu_1(T)]^2$$
    where $\\omega_0, \\omega_1$ are class probabilities and $\\mu_0, \\mu_1$ are class means.

    Args:
        image: Input image array.
        inverse: If True, produces an inverted binary mask.

    Returns:
        tuple containing:
            - float: Optimal threshold $T^*$.
            - np.ndarray: Binarized mask in uint8.
    """
    gray = to_gray(image)
    flag = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    threshold_val, binary = cv2.threshold(ensure_uint8(gray), 0, 255, flag + cv2.THRESH_OTSU)
    return float(threshold_val), binary


def adaptive_threshold_mean(
    image: np.ndarray,
    block_size: int = 11,
    c_offset: float = 2.0,
    inverse: bool = False,
) -> np.ndarray:
    """Apply local adaptive thresholding using mean neighborhood intensity.

    Threshold per pixel:
    $$T(x, y) = \\frac{1}{B^2} \\sum_{i,j \\in \\mathrm{Block}} I(x+i, y+j) - C$$

    Args:
        image: Input image array.
        block_size: Neighborhood window diameter (odd integer $\\ge 3$).
        c_offset: Constant subtracted from the calculated mean.
        inverse: If True, inverts output polarity.

    Returns:
        np.ndarray: Adaptively binarized mask.
    """
    gray = to_gray(image)
    b = max(3, block_size if block_size % 2 == 1 else block_size + 1)
    flag = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(
        ensure_uint8(gray), 255, cv2.ADAPTIVE_THRESH_MEAN_C, flag, b, c_offset
    )


def adaptive_threshold_gaussian(
    image: np.ndarray,
    block_size: int = 11,
    c_offset: float = 2.0,
    inverse: bool = False,
) -> np.ndarray:
    """Apply local adaptive thresholding using Gaussian-weighted neighborhood sum.

    Robust against uneven illumination, shadows, and lighting gradients across the image.

    Args:
        image: Input image array.
        block_size: Neighborhood window size (odd integer $\\ge 3$).
        c_offset: Constant offset subtracted from weighted mean.
        inverse: If True, inverts output polarity.

    Returns:
        np.ndarray: Adaptively binarized mask.
    """
    gray = to_gray(image)
    b = max(3, block_size if block_size % 2 == 1 else block_size + 1)
    flag = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(
        ensure_uint8(gray), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, flag, b, c_offset
    )
